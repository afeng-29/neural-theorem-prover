#!/user/af3698/neural-theorem-prover/venv/bin/python
"""Smoke-test: verify that 'lake env lean file.lean' works for parallel individual builds."""
import os, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

LEAN_PROJECT = Path("/user/af3698/neural-theorem-prover/lean_project").resolve()
ELAN_ENV = {
    **os.environ,
    "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
}

PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""

# Test cases: (name, proof_tactic, expected_result)
TESTS = [
    ("trivial_norm_num",   "norm_num",                    (True, True)),   # complete
    ("trivial_ring",       "ring",                        (False, False)), # ring can't prove 1+1=2
    ("incomplete_intro",   "intro h",                     (True, False)),  # valid partial (unsolved goals)
    ("bad_tactic",         "exact blahblah",              (False, False)), # unknown id
]

STMT_TEMPLATE = "theorem test_{name}_b0 : (1 : ℕ) + 1 = 2 := by"
STMT_INCOMPLETE = "theorem test_{name}_b0 : ∀ (n : ℕ), n = n := by"

def run_one(i: int, name: str, stmt: str, body: str) -> tuple[bool, bool]:
    fpath = LEAN_PROJECT / f"ProofGoals_b{i}.lean"
    file_lines = PREAMBLE.rstrip().splitlines() + [""]
    for line in stmt.splitlines():
        file_lines.append(line)
    for raw_line in body.splitlines():
        s = raw_line.strip()
        file_lines.append(f"  {s}" if s else "")
    src = "\n".join(file_lines) + "\n"

    try:
        fpath.write_text(src)
        result = subprocess.run(
            ["lake", "env", "lean", fpath.name],
            cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=120,
            env=ELAN_ENV,
        )
        out = result.stdout + result.stderr
        if result.returncode == 0 and "uses 'sorry'" not in out:
            return (True, True)
        has_unsolved = any("unsolved goals" in ln for ln in out.splitlines() if "error:" in ln)
        has_other    = any("unsolved goals" not in ln for ln in out.splitlines() if "error:" in ln)
        if has_unsolved and not has_other:
            return (True, False)
        return (False, False)
    except subprocess.TimeoutExpired:
        return (False, False)
    finally:
        fpath.unlink(missing_ok=True)

def main():
    cases = [
        (0, "trivial_norm_num", STMT_TEMPLATE.format(name="trivial_norm_num"), "norm_num"),
        (1, "incomplete_intro", STMT_INCOMPLETE.format(name="incomplete_intro"), "intro h\nexact h"),
        (2, "bad_tactic",       STMT_TEMPLATE.format(name="bad_tactic"),       "exact blahblah"),
        (3, "incomplete_goals", STMT_INCOMPLETE.format(name="incomplete_goals"), ""),  # empty body
    ]

    expected = [(True, True), (True, True), (False, False), (True, False)]

    print("Running 4 parallel 'lake env lean' calls...")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda c: run_one(*c), cases))
    elapsed = time.monotonic() - t0

    print(f"Elapsed: {elapsed:.1f}s\n")
    all_ok = True
    for (i, name, _, body), got, exp in zip(cases, results, expected):
        ok = got == exp
        all_ok = all_ok and ok
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}: got={got} expected={exp}")

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
