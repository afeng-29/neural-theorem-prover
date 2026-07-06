"""
Tree-search proof search for DeepSeek-Prover.

Uses batch lake-build verification (no live REPL needed) to prune bad
tactic prefixes early, then expand only valid branches.

Algorithm (BFS tree search):
  1. Generate k tactic prefixes (depth 1: first tactic only)
  2. Batch-verify → keep valid prefixes (those with only "unsolved goals")
  3. For each valid prefix at depth d, generate k next tactics
  4. Verify each (prefix + new_tactic) → prune
  5. Repeat until a prefix has NO errors (proof complete)
  6. Fall back to whole-proof generation if tree search times out

Verification approach (no sorry):
  Compile proof WITHOUT sorry. Classify each candidate by error type:
  - No errors:                    complete proof     (True, True)
  - Only "unsolved goals" error:  valid partial      (True, False)
  - Other errors:                 invalid tactic     (False, False)

Each candidate gets its own temp file and is verified via 'lake env lean'
in parallel — wall time ≈ one build, zero line-range attribution errors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""


@dataclass
class SearchNode:
    """Node in the proof search tree."""
    tactics: list[str]          # tactics applied from root to here
    depth: int
    parent: Optional["SearchNode"] = None
    is_complete: bool = False   # no remaining goals (proved!)


@dataclass
class TreeSearchResult:
    proof: Optional[str]        # newline-joined tactic sequence, or None
    proved: bool
    nodes_expanded: int
    elapsed_seconds: float
    error: str = ""


class TreeSearchProver:
    """
    BFS tree-search proof search using batch lake build for verification.

    Args:
        lean_project_path:  Path to the Lean 4 project (has lakefile.lean).
        width:              Tactic candidates to generate per node expansion.
        max_depth:          Maximum proof depth (number of tactics).
        batch_timeout:      Seconds budget for each lake build call.
        tree_timeout:       Wall-clock budget for the full tree search.
    """

    def __init__(
        self,
        lean_project_path: str | Path,
        width: int = 8,
        max_depth: int = 6,
        batch_timeout: int = 90,
        tree_timeout: float = 240.0,
        preamble: str | None = None,
    ):
        self.lean_project = Path(lean_project_path).resolve()
        self.width = width
        self.max_depth = max_depth
        self.batch_timeout = batch_timeout
        self.tree_timeout = tree_timeout
        self._preamble = preamble if preamble is not None else _PREAMBLE
        self._elan_env = {
            **os.environ,
            "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
        }

    def search(
        self,
        model,                   # DeepSeekProverModel
        formal_statement: str,   # 'theorem NAME PARAMS : GOAL := sorry'
    ) -> TreeSearchResult:
        """
        Run BFS tree search to prove formal_statement.
        Returns TreeSearchResult with proof tactics if found.
        """
        t_start = time.monotonic()
        nodes_expanded = 0

        # BFS frontier: list of SearchNode at current depth
        frontier = [SearchNode(tactics=[], depth=0)]
        base_stmt = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())

        for depth in range(self.max_depth):
            if not frontier:
                break
            if time.monotonic() - t_start > self.tree_timeout:
                break

            logger.info(
                "Tree search depth=%d, frontier=%d nodes (%.1fs elapsed)",
                depth, len(frontier), time.monotonic() - t_start,
            )

            # Expand all nodes at this depth in parallel
            next_nodes: list[SearchNode] = []
            all_candidates: list[tuple[SearchNode, str]] = []  # (parent, tactic)

            for node in frontier:
                elapsed = time.monotonic() - t_start
                if elapsed > self.tree_timeout:
                    break

                # Generate k next-tactic candidates for this node
                tactics = model.generate_next_tactics(
                    formal_statement=formal_statement,
                    applied_tactics=node.tactics,
                    n=self.width,
                )
                seen = set(node.tactics)
                for tac in tactics:
                    t = tac.strip()
                    if t and t not in seen:
                        all_candidates.append((node, t))
                nodes_expanded += 1

            if not all_candidates:
                break

            # De-duplicate candidates by (tactics_so_far + new_tactic).
            # Also drop candidates where the model regenerated the theorem header
            # (e.g. "theorem foo :") — these are not valid tactics and cause false
            # positives because Lean closes the `by` block early on keyword errors.
            seen_keys: set[str] = set()
            deduped: list[tuple[SearchNode, str]] = []
            for node, tac in all_candidates:
                key = "|".join(node.tactics + [tac])
                if key not in seen_keys and not re.match(r"^\s*theorem\s+\S", tac):
                    seen_keys.add(key)
                    deduped.append((node, tac))

            # Batch verify (no sorry): classify by error type.
            # "unsolved goals" only → valid partial; no errors → complete; other → invalid.
            verify_bodies = [
                "\n".join([f"  {t}" for t in node.tactics] + [f"  {tac}"])
                for node, tac in deduped
            ]

            verify_results = self._batch_verify(
                base_stmt=base_stmt,
                tactic_bodies=verify_bodies,
                timeout=self.batch_timeout,
            )

            for (node, tac), (valid, complete) in zip(deduped, verify_results):
                child = SearchNode(
                    tactics=node.tactics + [tac],
                    depth=depth + 1,
                    parent=node,
                    is_complete=complete,
                )
                if complete:
                    logger.info(
                        "Tree search: PROVED at depth=%d, tactics=%s",
                        depth + 1, child.tactics,
                    )
                    return TreeSearchResult(
                        proof="\n".join(child.tactics),
                        proved=True,
                        nodes_expanded=nodes_expanded,
                        elapsed_seconds=time.monotonic() - t_start,
                    )
                if valid and depth + 1 < self.max_depth:
                    next_nodes.append(child)

            frontier = next_nodes

            if time.monotonic() - t_start > self.tree_timeout:
                break

        return TreeSearchResult(
            proof=None,
            proved=False,
            nodes_expanded=nodes_expanded,
            elapsed_seconds=time.monotonic() - t_start,
        )

    def _batch_verify(
        self,
        base_stmt: str,
        tactic_bodies: list[str],
        timeout: int = 90,
    ) -> list[tuple[bool, bool]]:
        """
        Verify each tactic body via a parallel 'lake env lean <file>' call.

        Returns list of (valid, complete) pairs:
          (True, True):   no errors — proof is complete
          (True, False):  only "unsolved goals" error — valid tactics, proof incomplete
          (False, False): other errors — tactic is invalid

        Each candidate gets its own temp file (ProofGoals_bN.lean) so errors
        are never attributed to the wrong theorem via line-range arithmetic.
        All N files are elaborated in parallel; wall time ≈ one build.
        """
        if not tactic_bodies:
            return []

        from concurrent.futures import ThreadPoolExecutor

        def _verify_one(i: int, body: str) -> tuple[bool, bool]:
            unique_stmt = re.sub(r"(theorem\s+\S+)", rf"\1_b{i}", base_stmt, count=1)
            stmt_with_by = f"{unique_stmt} := by"
            file_lines: list[str] = self._preamble.rstrip().splitlines() + [""]
            for line in stmt_with_by.splitlines():
                file_lines.append(line)
            for raw_line in body.splitlines():
                stripped = raw_line.strip()
                file_lines.append(f"  {stripped}" if stripped else "")
            src = "\n".join(file_lines) + "\n"

            fpath = self.lean_project / f"ProofGoals_b{i}.lean"
            try:
                fpath.write_text(src)
                result = subprocess.run(
                    ["lake", "env", "lean", fpath.name],
                    cwd=self.lean_project,
                    capture_output=True, text=True,
                    timeout=timeout,
                    env=self._elan_env,
                )
                out = result.stdout + result.stderr

                if result.returncode == 0 and "uses 'sorry'" not in out:
                    return (True, True)

                has_unsolved = False
                has_other = False
                for line in out.splitlines():
                    if "error:" in line:
                        if "unsolved goals" in line:
                            has_unsolved = True
                        else:
                            has_other = True

                if has_unsolved and not has_other:
                    return (True, False)
                return (False, False)

            except subprocess.TimeoutExpired:
                return (False, False)
            except Exception as e:
                logger.warning("_verify_one[%d] error: %s", i, e)
                return (False, False)
            finally:
                try:
                    fpath.unlink(missing_ok=True)
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=len(tactic_bodies)) as executor:
            results = list(executor.map(
                lambda args: _verify_one(*args),
                enumerate(tactic_bodies),
            ))

        return results
