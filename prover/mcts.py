"""
Tree-search proof search for DeepSeek-Prover.

Uses batch lake-build verification (no live REPL needed) to prune bad
tactic prefixes early, then expand only valid branches.

Algorithm (BFS tree search):
  1. Generate k tactic prefixes (depth 1: first tactic only)
  2. Append `sorry` and batch-verify → keep valid prefixes
  3. For each valid prefix at depth d, generate k next tactics
  4. Verify each (prefix + new_tactic + sorry) → prune
  5. Repeat until a prefix has NO remaining goals (proof complete)
  6. Fall back to whole-proof generation if tree search times out

Why this works without a live REPL:
  A Lean file like:
      theorem foo : P := by
        intro h   -- valid
        sorry     -- remaining goal (treated as warning, not error)
  compiles successfully. We detect "valid partial proof" = compiles
  without error (sorry warning is OK).

The key efficiency gain: we verify k branches in ONE lake build call
(~35s) instead of k separate calls (k × 35s).
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
    ):
        self.lean_project = Path(lean_project_path).resolve()
        self.width = width
        self.max_depth = max_depth
        self.batch_timeout = batch_timeout
        self.tree_timeout = tree_timeout
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

            # De-duplicate candidates by (tactics_so_far + new_tactic)
            seen_keys: set[str] = set()
            deduped: list[tuple[SearchNode, str]] = []
            for node, tac in all_candidates:
                key = "|".join(node.tactics + [tac])
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append((node, tac))

            # Batch verify: (tactics_so_far + new_tactic + sorry) for each candidate
            # A result with ONLY sorry-warnings (no errors) means the tactic sequence
            # is valid so far and has remaining goals.
            # A result with NO warnings/errors at all means proof is COMPLETE.
            verify_bodies = [
                "\n".join(node.tactics + [tac, "  sorry"])
                if not (node.tactics + [tac])  # empty — shouldn't happen
                else "\n".join([f"  {t}" for t in node.tactics] + [f"  {tac}", "  sorry"])
                for node, tac in deduped
            ]

            verify_results = self._batch_verify_with_sorry(
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

    def _batch_verify_with_sorry(
        self,
        base_stmt: str,
        tactic_bodies: list[str],
        timeout: int = 90,
    ) -> list[tuple[bool, bool]]:
        """
        For each tactic body, check whether it's a valid (possibly partial) proof.

        Returns list of (valid, complete) pairs:
          valid:    the tactic sequence compiled without Lean errors
          complete: the proof is complete (no sorry needed / no remaining goals)

        We write all candidates to one file and parse Lean's output once.
        """
        if not tactic_bodies:
            return []

        goals_path = self.lean_project / "ProofGoals.lean"
        original = goals_path.read_text() if goals_path.exists() else None

        file_lines = [_PREAMBLE.rstrip(), ""]
        ranges: list[tuple[int, int]] = []

        for i, body in enumerate(tactic_bodies):
            start = len(file_lines) + 1
            file_lines.append(f"{base_stmt} := by")
            for raw_line in body.splitlines():
                stripped = raw_line.strip()
                file_lines.append(f"  {stripped}" if stripped else "")
            end = len(file_lines)
            ranges.append((start, end))
            file_lines.append("")

        src = "\n".join(file_lines)
        try:
            goals_path.write_text(src)
            result = subprocess.run(
                ["lake", "build", "TheoremProver"],
                cwd=self.lean_project,
                capture_output=True, text=True, timeout=timeout,
                env=self._elan_env,
            )
            out = result.stdout + result.stderr

            # Collect error lines (genuine Lean errors, not sorry-warnings)
            # Lean format: "ProofGoals.lean:N:M: error: ..."
            error_lines: set[int] = set()
            for m in re.finditer(r"ProofGoals\.lean:(\d+):\d+:.*?error:", out):
                error_lines.add(int(m.group(1)))

            # Collect sorry-warning lines
            # Lean format: "ProofGoals.lean:N:M: warning: declaration uses 'sorry'"
            sorry_lines: set[int] = set()
            for m in re.finditer(r"ProofGoals\.lean:(\d+):\d+:.*?warning:.*sorry", out):
                sorry_lines.add(int(m.group(1)))

            results = []
            for start, end in ranges:
                has_error = any(start <= ln <= end for ln in error_lines)
                has_sorry = any(start <= ln <= end for ln in sorry_lines)
                if has_error:
                    results.append((False, False))
                elif has_sorry:
                    results.append((True, False))   # valid partial proof
                else:
                    # No error and no sorry → proof is complete
                    results.append((True, True))
            return results

        except subprocess.TimeoutExpired:
            logger.warning("Batch verify timed out after %ds", timeout)
            return [(False, False)] * len(tactic_bodies)
        except Exception as e:
            logger.warning("Batch verify error: %s", e)
            return [(False, False)] * len(tactic_bodies)
        finally:
            if original is not None:
                goals_path.write_text(original)
            elif goals_path.exists():
                goals_path.unlink()
