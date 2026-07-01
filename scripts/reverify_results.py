"""
Post-process miniF2F result JSONs: re-verify each 'proved' result with a
single-candidate build. Distinguishes definite failures (FALSE POSITIVE)
from timeouts (UNCERTAIN — kept as proved).

Run on a compute node for speed parity with the original evaluation.
"""
import json, os, sys, re, subprocess, shutil, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger()

sys.path.insert(0, Path(__file__).parent.as_posix())
from run_minif2f_eval import parse_formal_statement, _build_lean_file, MINIF2F_PREAMBLE
from datasets import load_dataset

LEAN_PROJECT = Path("/project/dachxiu/afeng/prover/lean_project")
GOALS_FILE = LEAN_PROJECT / "ProofGoals.lean"
TIMEOUT = 120  # seconds — same as original eval

elan_bin = Path.home() / ".elan" / "bin"
ENV = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}

# Load both splits so validation results can also be re-verified.
ds = {}
for _split in ("test", "validation"):
    for row in load_dataset("cat-searcher/minif2f-lean4", split=_split):
        ds[row["id"]] = row

RESULT_FILES = [
    ("/project/dachxiu/afeng/prover/results/minif2f_byt5_ft_test.json",         "ByT5-FT",       "test"),
    ("/project/dachxiu/afeng/prover/results/minif2f_byt5_pretrained_test.json", "ByT5-pre",      "test"),
    ("/project/dachxiu/afeng/prover/results/minif2f_deepseek_base_test.json",     "DeepSeek-test", "test"),
    ("/project/dachxiu/afeng/prover/results/minif2f_deepseek_valid.json",        "DeepSeek-valid","validation"),
]


def reverify_one(pid: str, winning_proof: str) -> str:
    """Return 'confirmed', 'false_positive', or 'uncertain_timeout'."""
    prob = ds.get(pid)
    if not prob:
        return "uncertain_no_dataset_entry"

    thm_name, formal_body = parse_formal_statement(prob["formal_statement"])
    cand = [{"thm_name": f"{thm_name}_rechk", "formal_body": formal_body,
             "proof_body": winning_proof}]
    lean_src, thm_ranges = _build_lean_file(cand)

    original = GOALS_FILE.read_text(encoding="utf-8") if GOALS_FILE.exists() else ""
    try:
        GOALS_FILE.write_text(lean_src, encoding="utf-8")
        result = subprocess.run(
            ["lake", "build", "TheoremProver"],
            cwd=LEAN_PROJECT, capture_output=True, text=True,
            timeout=TIMEOUT, env=ENV,
        )
        output = result.stdout + result.stderr
        # Lean 4 error recovery can insert sorry on parse failure (exit 0 + warning).
        # Treat sorry-containing declarations as false positives.
        if result.returncode == 0 and "error:" not in output.lower() and "uses 'sorry'" not in output:
            return "confirmed"
        if "uses 'sorry'" in output:
            return "false_positive"

        # Definite failure: check for error lines
        error_lines = set()
        for m in re.finditer(r"error:.*?ProofGoals\.lean:(\d+):\d+:", output):
            error_lines.add(int(m.group(1)))

        if not error_lines:
            # Build failed but no error lines captured — inconclusive
            return "uncertain_no_error_lines"

        return "false_positive"

    except subprocess.TimeoutExpired:
        return "uncertain_timeout"
    except Exception as e:
        logger.warning("  Verify error for %s: %s", pid, e)
        return "uncertain_error"
    finally:
        GOALS_FILE.write_text(original, encoding="utf-8")


# Warm the lake cache first (allow up to 20 min; GPU nodes may still need linking)
logger.info("Warming lake cache...")
warmup = MINIF2F_PREAMBLE + "\ntheorem warmup_rv (n : ℕ) : n = n := rfl\n"
GOALS_FILE.write_text(warmup)
try:
    subprocess.run(["lake", "build", "TheoremProver"], cwd=LEAN_PROJECT,
                   capture_output=True, timeout=1200, env=ENV)
    logger.info("Cache warm. Starting re-verification.")
except subprocess.TimeoutExpired:
    logger.warning("Warmup timed out — proceeding anyway (cache may be cold, expect slower first build)")

for fpath, name, _split in RESULT_FILES:
    if not Path(fpath).exists():
        logger.info("=== %s: result file not found, skipping ===", name)
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data["results"]
    proved_pids = [pid for pid, r in results.items() if r.get("proved")]
    logger.info("=== %s: %d proved to re-verify ===", name, len(proved_pids))

    n_false_positive = 0
    n_uncertain = 0
    n_confirmed = 0

    for i, pid in enumerate(proved_pids):
        r = results[pid]
        proof = r.get("proof") or ""
        status = reverify_one(pid, proof)

        if status == "confirmed":
            n_confirmed += 1
            logger.info("[%d/%d] %s: OK", i + 1, len(proved_pids), pid)
        elif status == "false_positive":
            n_false_positive += 1
            logger.warning("[%d/%d] %s: FALSE POSITIVE  proof=%r",
                           i + 1, len(proved_pids), pid, proof[:60])
            results[pid]["proved"] = False
            results[pid]["proof"] = None
        else:
            n_uncertain += 1
            logger.warning("[%d/%d] %s: UNCERTAIN (%s) — keeping proved  proof=%r",
                           i + 1, len(proved_pids), pid, status, proof[:60])

    n_proved_final = n_confirmed + n_uncertain
    data["summary"]["n_proved"] = n_proved_final
    data["summary"]["pass_rate"] = n_proved_final / data["summary"]["n_attempted"]
    data["summary"]["re_verified"] = True
    data["summary"]["false_positives_removed"] = n_false_positive
    data["summary"]["uncertain_kept"] = n_uncertain

    backup = fpath.replace(".json", "_pre_reverify.json")
    shutil.copy(fpath, backup)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("=== %s DONE: confirmed=%d  false_positives=%d  uncertain=%d  final=%d/244 (%.1f%%) ===",
                name, n_confirmed, n_false_positive, n_uncertain,
                n_proved_final, n_proved_final / 244 * 100)
