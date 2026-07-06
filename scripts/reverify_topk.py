#!/user/af3698/neural-theorem-prover/venv/bin/python
"""
Re-verify whole-proof (top_k) results via individual lake build calls.
Usage: reverify_topk.py results/minif2f_deepseek_base_topk256_test.json
"""
import json, logging, os, re, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

LEAN_PROJECT = Path("/user/af3698/neural-theorem-prover/lean_project").resolve()
PREAMBLE = """\
import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Finset

"""
ELAN_ENV = {
    **os.environ,
    "PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}",
    "LEAN_NUM_THREADS": "1",
}


def verify_one(pid: str, formal_statement: str, proof: str) -> bool:
    base = re.sub(r":=\s*sorry\s*$", "", formal_statement.strip())
    file_lines = PREAMBLE.rstrip().splitlines() + [""]
    stmt_with_by = f"{base} := by"
    for line in stmt_with_by.splitlines():
        file_lines.append(line)
    for ln in proof.splitlines():
        s = ln.strip()
        file_lines.append(f"  {s}" if s else "")
    src = "\n".join(file_lines) + "\n"

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", pid)[:30]
    fpath = LEAN_PROJECT / f"ProofGoals_rv_{safe}.lean"
    try:
        fpath.write_text(src)
        result = subprocess.run(
            ["lake", "env", "lean", fpath.name],
            cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=300,
            env=ELAN_ENV,
        )
        out = result.stdout + result.stderr
        return result.returncode == 0 and "uses 'sorry'" not in out
    except subprocess.TimeoutExpired:
        logger.warning("Timeout on %s", pid)
        return False
    except Exception as e:
        logger.warning("Error on %s: %s", pid, e)
        return False
    finally:
        fpath.unlink(missing_ok=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: reverify_topk.py <results_json>")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    out_file = results_file.with_name(results_file.stem + "_reverified.json")

    data = json.loads(results_file.read_text())
    results = data.get("results", data)
    proved_items = [(k, v) for k, v in results.items() if isinstance(v, dict) and v.get("proved")]
    logger.info("Re-verifying %d claimed proofs from %s", len(proved_items), results_file.name)

    from datasets import load_dataset
    ds = load_dataset("cat-searcher/minif2f-lean4", split="test")
    stmt_map = {row.get("id", row.get("problem_name", "")): row["formal_statement"] for row in ds}

    confirmed = 0
    rejected = 0
    skipped = 0
    corrected = dict(results)

    # Verify up to 8 in parallel (GPU node can handle it)
    PARALLEL = 8
    items_with_formal = []
    for pid, v in proved_items:
        formal = stmt_map.get(pid)
        if not formal:
            logger.warning("%s — no formal statement, SKIPPING", pid)
            skipped += 1
            continue
        items_with_formal.append((pid, formal, v.get("proof", "")))

    def check(args):
        pid, formal, proof = args
        return pid, verify_one(pid, formal, proof)

    total = len(items_with_formal)
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for pid, ok in ex.map(check, items_with_formal):
            done += 1
            v = results[pid]
            if ok:
                confirmed += 1
                status = "OK"
            else:
                rejected += 1
                status = "FALSE POSITIVE"
                corrected[pid] = {**v, "proved": False, "reverify": "false_positive"}
            proof_preview = repr(results[pid].get("proof", "")[:80])
            logger.info("[%d/%d] %s — %s | proof: %s", done, total, pid, status, proof_preview)
            if done % 10 == 0:
                print(f"=== [{done}/{total}] confirmed={confirmed} rejected={rejected} ===", flush=True)

    total_items = len([v for v in corrected.values() if isinstance(v, dict)])
    real_proved = sum(1 for v in corrected.values() if isinstance(v, dict) and v.get("proved"))

    print(f"\n=== RE-VERIFICATION COMPLETE ===")
    print(f"Claimed:   {len(proved_items)}/244")
    print(f"Confirmed: {confirmed}/244 = {100*confirmed/244:.1f}%")
    print(f"Rejected:  {rejected} false positives")
    print(f"Skipped:   {skipped} (no formal statement)")

    out_data = {
        "summary": {"proved": real_proved, "total": total_items,
                    "pct": round(100 * real_proved / max(total_items, 1), 2)},
        "results": corrected,
    }
    out_file.write_text(json.dumps(out_data, indent=2))
    logger.info("Written to %s", out_file)


if __name__ == "__main__":
    main()
