"""
Fast Lean 4 REPL for interactive theorem proving.

Keeps a single Lean subprocess alive (Mathlib loaded once ~30s startup),
then accepts tactic commands via stdin/stdout JSON protocol.

This wraps the lean4-repl JSON protocol:
  → {"cmd": "tactic", "tactic": "intro h", "proofState": <int>}
  ← {"proofState": <int>, "goals": ["h : P ⊢ Q"], "error": null}

Usage:
    with LeanREPL(lean_project) as repl:
        sid0 = repl.init_proof(formal_statement)
        result = repl.apply_tactic(sid0, "intro h")
        if result.success:
            result2 = repl.apply_tactic(result.state_id, "exact h")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""

_REPL_LEAN = """\
-- Lean 4 REPL for neural-theorem-prover.
-- Reads JSON commands from stdin, writes JSON responses to stdout.
-- Protocol:
--   {"cmd": "proof", "stmt": "theorem foo : P := by"}
--   {"cmd": "tactic", "sid": 0, "tactic": "intro h"}
-- Responses:
--   {"ok": true, "sid": 1, "goals": ["h : P ⊢ Q"]}
--   {"ok": false, "error": "unknown tactic"}
import Mathlib
import Aesop

open BigOperators Real Nat Topology Finset

set_option maxHeartbeats 400000

-- REPL state: a map from sid to TacticState
private opaque State : Type := Unit

#eval (do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  -- Simple line-by-line JSON protocol
  -- Each line is a command, each response is a line
  pure () : IO Unit)
"""


@dataclass
class TacticResult:
    success: bool
    state_id: Optional[int] = None   # new proof state id (for branching)
    goals: list[str] = None          # remaining goals (empty = proof done)
    error: str = ""
    is_complete: bool = False        # no remaining goals


class BatchVerifier:
    """
    Verify proof candidates via lake build (no persistent process).
    Slower (30-40s per batch) but reliable. Used when REPL is unavailable.
    """

    def __init__(self, lean_project_path: str | Path):
        self.lean_project = Path(lean_project_path).resolve()
        self._elan_env = {
            **os.environ,
            "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
        }

    def verify_batch(
        self,
        formal_statement: str,
        proof_bodies: list[str],
        timeout: int = 120,
    ) -> list[bool]:
        """
        Try each proof body for formal_statement.
        formal_statement: 'theorem NAME PARAMS : GOAL := sorry'
        proof_bodies: list of proof body strings (indented tactics)
        Returns: list of bool (True = Lean accepted)
        """
        if not proof_bodies:
            return []

        # Build a Lean file with all proof candidates
        header = _PREAMBLE.rstrip()
        base = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
        file_lines = [header, ""]
        ranges: list[tuple[int, int]] = []

        for i, body in enumerate(proof_bodies):
            start = len(file_lines) + 1
            file_lines.append(f"{base} := by")
            # Indent the body
            for line in body.splitlines():
                file_lines.append(f"  {line.strip()}" if line.strip() else "")
            end = len(file_lines)
            ranges.append((start, end))
            file_lines.append("")

        src = "\n".join(file_lines)
        goals_path = self.lean_project / "ProofGoals.lean"
        original = goals_path.read_text() if goals_path.exists() else None

        try:
            goals_path.write_text(src)
            result = subprocess.run(
                ["lake", "build", "TheoremProver"],
                cwd=self.lean_project,
                capture_output=True, text=True, timeout=timeout,
                env=self._elan_env,
            )
            out = result.stdout + result.stderr
            if result.returncode == 0 and "error:" not in out.lower():
                return [True] * len(proof_bodies)

            # Parse error line numbers
            error_lines: set[int] = set()
            for m in re.finditer(r"error:.*?ProofGoals\.lean:(\d+):\d+:", out):
                error_lines.add(int(m.group(1)))

            return [
                not any(s <= ln <= e for ln in error_lines)
                for s, e in ranges
            ]
        except subprocess.TimeoutExpired:
            logger.warning("lake build timed out after %ds", timeout)
            return [False] * len(proof_bodies)
        finally:
            if original is not None:
                goals_path.write_text(original)
            elif goals_path.exists():
                goals_path.unlink()
