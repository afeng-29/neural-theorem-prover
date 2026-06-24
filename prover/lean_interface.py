"""
LeanDojo wrapper for interactive tactic-by-tactic proof search.

Design:
  LeanDojo 4.x requires theorems to live in a *traced* git repository.
  This module handles:
    1. Registering custom theorems in a ProofGoals.lean file inside lean_project/
    2. Committing that file and tracing the project (once per batch of theorems)
    3. Wrapping Dojo for interactive tactic application
    4. Using check_proof() for fast whole-proof verification

Environment variables:
  GITHUB_ACCESS_TOKEN  — optional; only needed when tracing remote GitHub repos.
                         NOT needed for local-repo tracing (our workflow). A stale/
                         expired token is silently hidden during module import.
  LEAN_DOJO_CACHE_PATH — optional, overrides default cache dir (~/.cache/leandojo)

Gotchas:
  - lean_project/ must be a git repository (setup.sh initialises it).
  - Tracing is cached on disk; the first trace of a file takes ~30s because
    it re-elaborates the theorem declarations.
  - Always use _ProofSession as a context manager — it guarantees Lean
    subprocess cleanup even on exception.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Suppress harmless SWIG DeprecationWarnings from lean_dojo's C extensions.
warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*swigvarlink.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*builtin type.*no __module__.*", category=DeprecationWarning)

logger = logging.getLogger(__name__)

_lean_dojo_available: Optional[bool] = None
_lean_dojo_patched: bool = False


def _patch_leandojo_local_repo() -> None:
    """
    Monkey-patch LeanDojo to avoid copying the large .lake directory for local repos.

    LeanDojo's default flow for LOCAL repos:
      url_to_repo()        → shutil.copytree(lean_project, tmp_dir/lean_project)  [7.5 GB]
      repo_cache.store()   → shutil.copytree(tmp_dir/lean_project, cache_path)    [7.5 GB again]

    Both copies fail on a nearly-full disk (need 15+ GB free).  We patch:
      url_to_repo       → open the repo in-place via git.Repo(url), no copy
      repo_cache.store  → create a symlink to the original instead of copying
    """
    global _lean_dojo_patched
    if _lean_dojo_patched:
        return
    try:
        import functools
        # lean.py runs `GITHUB.get_user().login` at module-level if a token is set.
        # A stale/expired token raises a 401 and aborts this function.  We use
        # LOCAL repos that don't need GitHub auth at all, so permanently remove
        # the token here.  Child processes (e.g., launch_progressbar's _monitor)
        # inherit os.environ, so they must NOT see the bad token either.
        os.environ.pop("GITHUB_ACCESS_TOKEN", None)
        # NUM_PROCS=1 → NUM_WORKERS=0 → TracedRepo.from_traced_files skips Ray
        # (ray.init fails on this macOS: raylet times out waiting for dashboard
        # agent metrics port file). Single-threaded is fine for our 2-file repo.
        os.environ["NUM_PROCS"] = "1"
        import lean_dojo.data_extraction.lean as _lean_mod
        import lean_dojo.data_extraction.cache as _cache_mod
        # Also patch the already-imported constants module so it takes effect
        # even if lean_dojo.constants was imported earlier.
        import lean_dojo.data_extraction.traced_data as _traced_mod
        _traced_mod.NUM_WORKERS = 0
        # Path.glob("**/*.ast.json") doesn't follow symlinks on macOS Python 3.12.
        # Our cache uses .lake/build → symlink to original's build dir, so glob
        # finds 0 JSON files. Patch from_traced_files to use os.walk(followlinks=True).
        import os as _os, random as _random, pathlib as _pathlib
        _orig_from_traced = _traced_mod.TracedRepo.from_traced_files.__func__
        @classmethod
        def _from_traced_patched(cls, root_dir, build_deps=True):
            root_dir = _pathlib.Path(root_dir).resolve()
            json_paths = [
                _pathlib.Path(dirp) / f
                for dirp, _, files in _os.walk(str(root_dir), followlinks=True)
                for f in files if f.endswith(".ast.json")
            ]
            _random.shuffle(json_paths)
            # Temporarily override the module-level glob so the rest of the
            # original function works. Simplest: just call original with root_dir
            # patched so glob finds the files via a saved json_paths injection.
            # Instead, replicate the non-Ray branch directly.
            from lean_dojo.data_extraction.traced_data import (
                TracedFile, _build_dependency_graph,
            )
            from lean_dojo.data_extraction.lean import LeanGitRepo, is_git_repo
            if not is_git_repo(root_dir):
                raise RuntimeError(f"{root_dir} is not a Git repo.")
            repo = LeanGitRepo.from_path(root_dir)
            from tqdm import tqdm
            traced_files = [
                TracedFile.from_traced_file(root_dir, p, repo)
                for p in tqdm(json_paths)
            ]
            # get_dependencies() queries GitHub API — skip for build_deps=False
            # (dependencies only needed for cross-file dependency graph).
            dependencies = repo.get_dependencies(root_dir) if build_deps else {}
            if build_deps:
                traced_files_graph = _build_dependency_graph(traced_files, root_dir, repo)
            else:
                traced_files_graph = None
            traced_repo = cls(repo, dependencies, root_dir, traced_files, traced_files_graph)
            traced_repo._update_traced_files()
            return traced_repo
        _traced_mod.TracedRepo.from_traced_files = _from_traced_patched

        # load_from_disk has the same symlink-glob issue with *.trace.xml.
        @classmethod
        def _load_from_disk_patched(cls, root_dir, build_deps=True):
            root_dir = _pathlib.Path(root_dir).resolve()
            from lean_dojo.data_extraction.traced_data import TracedFile, _build_dependency_graph
            from lean_dojo.data_extraction.lean import LeanGitRepo, is_git_repo
            from lean_dojo.constants import LOAD_USED_PACKAGES_ONLY
            if not is_git_repo(root_dir):
                raise RuntimeError(f"{root_dir} is not a Git repo.")
            repo = LeanGitRepo.from_path(root_dir)
            xml_paths = [
                _pathlib.Path(dirp) / f
                for dirp, _, files in _os.walk(str(root_dir), followlinks=True)
                for f in files if f.endswith(".trace.xml")
            ]
            if LOAD_USED_PACKAGES_ONLY:
                xml_paths = [
                    p for p in xml_paths
                    if "lake-packages/" not in str(p) and ".lake/packages" not in str(p)
                ]
            from tqdm import tqdm
            traced_files = [
                TracedFile.from_xml(root_dir, p, repo) for p in tqdm(xml_paths)
            ]
            # get_dependencies() queries GitHub API — skip for build_deps=False.
            dependencies = repo.get_dependencies(root_dir) if build_deps else {}
            if build_deps:
                traced_files_graph = _build_dependency_graph(traced_files, root_dir, repo)
            else:
                traced_files_graph = None
            traced_repo = cls(repo, dependencies, root_dir, traced_files, traced_files_graph)
            traced_repo._update_traced_files()
            return traced_repo
        _traced_mod.TracedRepo.load_from_disk = _load_from_disk_patched

        from git import Repo as _GitRepo

        _orig_fn = _lean_mod.url_to_repo.__wrapped__   # unwrap functools.cache

        def _url_to_repo_patched(url, num_retries=2, repo_type=None, tmp_dir=None):
            from lean_dojo.data_extraction.lean import RepoType, normalize_url, get_repo_type
            url_n = normalize_url(url)
            rt = repo_type or get_repo_type(url_n)
            if rt == RepoType.LOCAL:
                return _GitRepo(url_n)   # open in-place — no 7.5 GB copy
            return _orig_fn(url, num_retries=num_retries, repo_type=repo_type, tmp_dir=tmp_dir)

        _lean_mod.url_to_repo = functools.cache(_url_to_repo_patched)

        import shutil as _shutil

        def _store_patched(self, src, rel_cache_dir):
            src = Path(src)  # GitPython's working_dir is a str, not Path
            cache_path = self.cache_dir / rel_cache_dir

            # LeanDojo calls store() in two contexts:
            # 1. LeanGitRepo.__post_init__ (rel_cache_dir starts with "repos/"):
            #    caches the raw git repo. For LOCAL repos we just return src
            #    in-place — no 10 GB copy.
            # 2. get_traced_repo_path (no "repos/" prefix):
            #    caches the traced repo (AST files, symlinks).
            if str(rel_cache_dir).startswith("repos/"):
                # Return src directly so gitpython opens the repo from its real
                # path. A symlink would break relative gitdir resolution: the
                # lean_project .git file says "gitdir: ../.git/modules/lean_project"
                # which only resolves correctly when opened from lean_project itself,
                # not from a symlink at an arbitrary cache location.
                return src

            if not cache_path.parent.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                # The trace copies lean4 stdlib (2.6 GB) as a real dir.
                # Before caching, replace it with a symlink to the toolchain path
                # so that copytree(symlinks=True) preserves it as a symlink.
                lean4_in_trace = src / ".lake" / "packages" / "lean4"
                if lean4_in_trace.exists() and not lean4_in_trace.is_symlink():
                    lean4_real = lean4_in_trace.resolve()
                    _shutil.rmtree(lean4_in_trace)
                    os.symlink(lean4_real, lean4_in_trace)
                # Copy trace to persistent cache; symlinks=True preserves package
                # symlinks, so the cache stores only AST files (small).
                _shutil.copytree(src, cache_path, symlinks=True)
            return cache_path

        _cache_mod.Cache.store = _store_patched

        # Patch 3: clone_and_checkout — after git-cloning a LOCAL repo,
        # symlink .lake from the original so lake build finds pre-built oleans.
        # Without this, lake would try to re-download 4+ GB of mathlib packages.
        _orig_clone = _lean_mod.LeanGitRepo.clone_and_checkout

        def _clone_and_checkout_patched(self):
            from lean_dojo.data_extraction.lean import RepoType
            if self.repo_type == RepoType.LOCAL:
                from git import Repo as _GitRepo
                r = _GitRepo.clone_from(self.url, Path(self.name), no_checkout=True)
                r.git.checkout(self.commit)
                # DON'T do recursive submodule_update — embedded .lake/packages are
                # not real submodules; updating them would re-clone mathlib (4+ GB).
                # Instead, create .lake/ structure and symlink each PACKAGE individually
                # (excluding 'lean4' so the trace can copy it fresh via lean --print-prefix).
                # We cannot symlink the whole .lake/ dir because the trace calls
                # shutil.copytree(lean_prefix, ".lake/packages/lean4") which fails if
                # that dir already exists via a symlink.
                lake_dir = Path(self.name) / ".lake"
                original_lake = Path(self.url) / ".lake"
                if original_lake.exists() and not lake_dir.exists():
                    lake_dir.mkdir(exist_ok=True)
                    # Symlink .lake/build (project oleans — avoids re-compile).
                    orig_build = original_lake / "build"
                    if orig_build.exists():
                        os.symlink(orig_build, lake_dir / "build")
                    # Symlink each package except 'lean4' (trace creates that from lean --print-prefix).
                    orig_packages = original_lake / "packages"
                    if orig_packages.exists():
                        (lake_dir / "packages").mkdir(exist_ok=True)
                        for pkg in orig_packages.iterdir():
                            if pkg.name != "lean4":
                                os.symlink(pkg, lake_dir / "packages" / pkg.name)
            else:
                _orig_clone(self)

        _lean_mod.LeanGitRepo.clone_and_checkout = _clone_and_checkout_patched

        # Patch 4: _get_modified_proof — stop appending lean_file[proof_end:] which
        # adds all other theorems (with sorry) AFTER the target theorem.  Lean 4.31
        # elaborates those concurrently with lean_dojo_repl; when they finish first
        # the process exits before the REPL can send its response, causing pexpect EOF.
        import lean_dojo.interaction.dojo as _dojo_mod
        from lean_dojo.interaction.dojo import DojoInitError as _DojoInitError
        from lean_dojo.data_extraction.traced_data import get_code_without_comments as _gcwc

        def _get_modified_proof_patched(self, traced_file):
            assert isinstance(self.entry, _dojo_mod.Theorem)
            traced_theorem = traced_file.get_traced_theorem(self.entry)
            if traced_theorem is None:
                raise _DojoInitError(
                    f"Failed to locate the theorem with `{self.entry.full_name}` as its fully qualified name."
                )
            proof_start, proof_end = traced_theorem.locate_proof()
            lean_file = traced_file.lean_file
            code_proof = "by\n  lean_dojo_repl\n  sorry\n"
            code_before_theorem = _gcwc(
                lean_file, lean_file.start_pos, traced_theorem.start, traced_file.comments
            )
            code_thereom = _gcwc(
                lean_file, traced_theorem.start, proof_start, traced_file.comments
            ).strip()
            if code_thereom.endswith(" where"):
                raise _DojoInitError("Cannot interact with theorems with the `where` keyword.")
            if not code_thereom.endswith(":="):
                code_thereom += " := "
            # Drop lean_file[proof_end:] — that would add all subsequent theorems
            # (marked sorry) which Lean elaborates concurrently and whose sorry-warnings
            # cause the elaboration to finish before the REPL responds.
            return str(
                self._get_imports()
                + code_before_theorem
                + "\n\nset_option maxHeartbeats 0 in\n"
                + code_thereom
                + code_proof
            )

        _dojo_mod.Dojo._get_modified_proof = _get_modified_proof_patched

        # Patch 5: use PopenSpawn (subprocess with pipes) instead of pty_spawn for Dojo.
        #
        # Root cause: pexpect.spawn uses pty.fork() so lean's stdin is a PTY slave.
        # On macOS + Lean 4.31.0, lean's runtime detects the TTY and drains stdin
        # before our lean_dojo_repl tactic's IO.getStdin.getLine can read it.
        # Result: getLine always returns "" → REPL loop never receives any command.
        #
        # Fix: replace pexpect.spawn with pexpect.popen_spawn.PopenSpawn which uses
        # subprocess.Popen(stdin=PIPE, stdout=PIPE, stderr=STDOUT). Lean's stdin is a
        # pipe — getLine blocks correctly and reads each command when we sendline().

        def _patched_dojo_enter(self_dojo):
            from pexpect.popen_spawn import PopenSpawn as _PopenSpawn
            import time as _time2
            import json as _json2
            import threading as _threading
            import signal as _signal
            from pathlib import Path as _Path2
            from lean_dojo.constants import TACTIC_CPU_LIMIT as _cpu, TACTIC_MEMORY_LIMIT as _mem
            from lean_dojo.data_extraction.trace import get_traced_repo_path as _gtrp2
            from lean_dojo.interaction.dojo import (
                DojoInitError as _DII2, DojoTacticTimeoutError as _DTT2,
                TacticState as _TState2, CommandState as _CState2,
                kill_descendants as _kd2,
            )
            from loguru import logger as _log2
            _log2.debug(f"Initializing Dojo (PopenSpawn/pipe-stdin) for {self_dojo.entry}")

            traced_repo_path = _gtrp2(self_dojo.repo, getattr(self_dojo, "build_deps", False))
            repl_path = traced_repo_path / "Lean4Repl.lean"
            assert repl_path.exists(), (
                "Unable to find Lean4Repl.lean in the traced repo. "
                "See https://github.com/lean-dojo/LeanDojo/releases/tag/v2.0.0."
            )

            try:
                traced_file = self_dojo._locate_traced_file(traced_repo_path)
            except FileNotFoundError:
                raise _DII2(
                    f"Cannot find the *.ast.json file for {self_dojo.entry} in {traced_repo_path}."
                )

            self_dojo._modify_file(traced_file)

            memory_limit = 1024 * int(_mem[:-1])
            modified_path = _Path2(self_dojo.modified_file.name).relative_to(traced_repo_path)
            cmd = f"lake env lean --threads={_cpu} --memory={memory_limit} {modified_path}"

            # PopenSpawn: stdin=PIPE, stdout+stderr=PIPE. Lean sees a pipe for stdin,
            # so IO.getStdin.getLine blocks correctly and receives each sendline() call.
            self_dojo.proc = _PopenSpawn(
                cmd,
                timeout=self_dojo.timeout,
                maxread=1,
                encoding="utf-8",
                cwd=str(traced_repo_path),
            )

            # PopenSpawn lacks isalive() which _check_alive() calls internally.
            # Patch both on the instance to use subprocess.Popen.poll() instead.
            _popen_inner = self_dojo.proc.proc  # the subprocess.Popen object
            self_dojo.proc.isalive = lambda: _popen_inner.poll() is None

            import types as _types

            def _check_alive_popen(self_inner):
                from lean_dojo.interaction.dojo import DojoCrashError as _DCE2
                rc = _popen_inner.poll()
                if rc is None:
                    return  # process still running — ok
                if rc in (-9, 137):
                    raise _DCE2("OOM")
                elif rc != 0:
                    raise _DCE2(f"Unexpected exit code: {rc}")
                # rc == 0: lean exited cleanly; let caller handle pipe EOF naturally

            self_dojo._check_alive = _types.MethodType(_check_alive_popen, self_dojo)

            # Hard wall-clock deadline for init: pexpect's per-expect timeout resets on
            # every newline (sorry warnings, build output), so can greatly exceed
            # self_dojo.timeout when Lean prints many lines. A threading.Timer fires
            # SIGALRM on the main thread after INIT_HARD_LIMIT seconds regardless.
            INIT_HARD_LIMIT = 300  # 5 min max for Lean to print first REPL> response
            _init_timed_out = [False]

            def _hard_kill():
                _init_timed_out[0] = True
                try:
                    _kd2(_popen_inner.pid)
                except Exception:
                    pass

            _hard_timer = _threading.Timer(INIT_HARD_LIMIT, _hard_kill)
            _hard_timer.daemon = True
            _hard_timer.start()

            # Read initial tactic state
            try:
                res = _json2.loads(self_dojo._read_next_line()[0])
            except (_DTT2, EOFError) as _ex:
                _hard_timer.cancel()
                try:
                    _kd2(_popen_inner.pid)
                except Exception:
                    pass
                self_dojo.modified_file.__exit__(None, None, None)
                msg = "Timeout during initialization" if _init_timed_out[0] or isinstance(_ex, _DTT2) else "Unexpected EOF"
                raise _DII2(msg)
            except Exception as ex:
                _hard_timer.cancel()
                try:
                    _kd2(_popen_inner.pid)
                except Exception:
                    pass
                self_dojo.modified_file.__exit__(None, None, None)
                if hasattr(traced_file, 'has_prelude') and traced_file.has_prelude:
                    raise _DII2(
                        "Currently LeanDojo does not support interacting with proofs in prelude files."
                    )
                raise ex

            _hard_timer.cancel()

            assert res["error"] is None

            if self_dojo.uses_tactics:
                assert res["tacticState"] != "no goals"
                init_state = _TState2(
                    self_dojo._post_process(res["tacticState"]),
                    res["sid"],
                )
            else:
                assert self_dojo.uses_commands
                init_state = _CState2(int(res["sid"]))

            self_dojo.start_time = _time2.monotonic()
            return self_dojo, init_state

        def _patched_dojo_exit(self_dojo, exc_type, exc_val, exc_tb):
            from lean_dojo.interaction.dojo import kill_descendants as _kd
            from loguru import logger as _log2
            _log2.debug("Cleaning up (PopenSpawn patch).")
            try:
                _kd(self_dojo.proc.proc.pid)
            except Exception:
                pass
            self_dojo.modified_file.__exit__(exc_type, exc_val, exc_tb)

        _dojo_mod.Dojo.__enter__ = _patched_dojo_enter
        _dojo_mod.Dojo.__exit__ = _patched_dojo_exit

        _lean_dojo_patched = True
        logger.info(
            "LeanDojo patched: LOCAL repos cloned with per-package symlinks "
            "(no 7.5 GB mathlib copy); cache uses symlinks=True."
        )
    except Exception as exc:
        logger.warning("LeanDojo local-repo patch failed (will proceed without it): %s", exc)


def _check_lean_dojo() -> bool:
    global _lean_dojo_available
    if _lean_dojo_available is not None:
        return _lean_dojo_available
    try:
        import lean_dojo  # noqa: F401
        _lean_dojo_available = True
    except ImportError:
        _lean_dojo_available = False
    return _lean_dojo_available


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of applying a single tactic."""
    success: bool
    next_state: str   # proof state string after the tactic ("" if proof complete or failed)
    error_message: str = ""
    is_complete: bool = False


# ── Main interface ─────────────────────────────────────────────────────────────

class LeanInterface:
    """
    High-level interface for interactive Lean 4 proving via LeanDojo.

    Usage:
        iface = LeanInterface("./lean_project")
        with iface.open_proof("n + 0 = n", ["n : ℕ"], name="nat_add_zero") as session:
            r = session.apply_tactic("simp")
            print(r.is_complete)   # True

    The lean_project must be a git repository — run setup.sh which calls
    `git init && git commit` for you.
    """

    # Relative path (inside lean_project) where we write custom theorems.
    PROOF_GOALS_FILE = "ProofGoals.lean"

    def __init__(self, lean_project_path: str | Path):
        self.lean_project_path = Path(lean_project_path).resolve()
        if not (self.lean_project_path / "lakefile.lean").exists():
            raise FileNotFoundError(
                f"No lakefile.lean at {self.lean_project_path}. "
                "Is this a Lean 4 project? Run `lake build` first."
            )
        # Inject elan/lake into os.environ['PATH'] so every subprocess spawned by
        # LeanDojo (which calls subprocess.run("lake build", shell=True) without an
        # explicit env= argument) can find the `lake` binary.
        elan_bin = str(Path.home() / ".elan" / "bin")
        current_path = os.environ.get("PATH", "")
        if elan_bin not in current_path:
            os.environ["PATH"] = f"{elan_bin}:{current_path}"
            logger.debug("Added %s to PATH for LeanDojo subprocesses", elan_bin)
        if os.environ.get("GITHUB_ACCESS_TOKEN"):
            logger.warning(
                "GITHUB_ACCESS_TOKEN is set — if stale/expired it will cause a 401 "
                "on import. Unset it for local-repo tracing: unset GITHUB_ACCESS_TOKEN"
            )
        # Patch LeanDojo to avoid copying 7.5 GB .lake when tracing local repos.
        _patch_leandojo_local_repo()
        # Ensure the project is a git repo (needed by LeanDojo).
        self._ensure_git_repo()

    def open_proof(
        self,
        theorem: str,
        hypotheses: list[str],
        name: str | None = None,
        tactic_timeout: int = 600,
    ) -> "_ProofSession":
        """
        Return a context-manager session for interactive proving.

        theorem:         Goal string, e.g. "n + 0 = n"
        hypotheses:      Hypothesis list, e.g. ["n : ℕ", "h : n > 0"]
        name:            Optional theorem name (auto-generated if None)
        tactic_timeout:  Per-tactic Lean subprocess timeout in seconds
        """
        thm_name = name or self._auto_name(theorem)
        return _ProofSession(
            theorem=theorem,
            hypotheses=hypotheses,
            thm_name=thm_name,
            lean_project_path=self.lean_project_path,
            proof_goals_file=self.PROOF_GOALS_FILE,
            tactic_timeout=tactic_timeout,
        )

    def verify_proof(
        self,
        theorem: str,
        hypotheses: list[str],
        proof_tactics: list[str],
        name: str | None = None,
    ) -> bool:
        """
        Return True if proof_tactics form a complete, accepted proof.
        Uses LeanDojo's check_proof() when the theorem has been traced,
        otherwise falls back to a subprocess check.
        """
        thm_name = name or self._auto_name(theorem)
        lean_src = _build_lean_source(theorem, hypotheses, thm_name,
                                      proof_tactics=proof_tactics)
        return self._subprocess_check(lean_src)

    def verify_proofs_batch(
        self,
        items: list[tuple[str, list[str], list[str], str]],
    ) -> list[bool]:
        """
        Verify multiple proofs in ONE lake build call (much faster than
        calling verify_proof() N times — Lean loads Mathlib interfaces once).

        items: [(theorem, hypotheses, proof_tactics, thm_name), ...]
        Returns: list of bool, same order as items.

        If the batch build succeeds, all theorems passed.
        If it fails, falls back to individual checks per theorem.
        """
        # Build one file with all theorems (skip per-item import headers).
        parts = ["import Mathlib\nimport Aesop\n"]
        for theorem, hypotheses, proof_tactics, thm_name in items:
            hyp_str = (" " + " ".join(f"({h})" for h in hypotheses)) if hypotheses else ""
            indented = "\n".join(f"  {t}" for t in (proof_tactics or ["sorry"]))
            parts.append(f"\ntheorem {thm_name}{hyp_str} : {theorem} := by\n{indented}\n")
        combined_src = "\n".join(parts)

        goals_path = self.lean_project_path / self.PROOF_GOALS_FILE
        elan_bin = Path.home() / ".elan" / "bin"
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}
        original = goals_path.read_text(encoding="utf-8") if goals_path.exists() else None
        try:
            goals_path.write_text(combined_src, encoding="utf-8")
            logger.info("Batch build: %d theorems in one lake build...", len(items))
            result = subprocess.run(
                ["lake", "build", "TheoremProver"],
                cwd=self.lean_project_path,
                capture_output=True, text=True, timeout=2400, env=env,
            )
            combined_out = result.stdout + result.stderr
            if result.returncode == 0 and "error:" not in combined_out.lower():
                return [True] * len(items)
            # Log first 600 chars of output to help diagnose the failure.
            logger.debug("Batch build output:\n%s", combined_out[:600])
            logger.info(
                "Batch build failed (rc=%d, has_error=%s) — retrying individually",
                result.returncode, "error:" in combined_out.lower(),
            )
        except Exception as e:
            logger.warning("Batch build failed: %s — falling back to individual checks", e)
        finally:
            if original is not None:
                goals_path.write_text(original, encoding="utf-8")
            elif goals_path.exists():
                goals_path.unlink()

        # Batch failed — retry individually to identify which theorems passed.
        logger.info("Batch failed; retrying %d theorems individually...", len(items))
        results = []
        for theorem, hypotheses, proof_tactics, thm_name in items:
            lean_src = _build_lean_source(theorem, hypotheses, thm_name,
                                          proof_tactics=proof_tactics)
            results.append(self._subprocess_check(lean_src))
        return results

    def prepare_theorem_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> str:
        """
        Write all theorems (with sorry proofs) into ProofGoals.lean in one shot
        and commit once. Returns the commit hash shared by all theorems.

        Call this BEFORE any open_proof() sessions so they all share one cached
        trace — otherwise each session creates a new commit and re-traces (~26 min each).

        items: [(theorem, hypotheses), ...]
        Theorem names are auto-generated using the same _auto_name() scheme as
        open_proof() so that _ProofSession finds theorems without re-writing the file.
        """
        goals_path = self.lean_project_path / self.PROOF_GOALS_FILE
        parts = ["import Mathlib\nimport Aesop\n"]
        for theorem, hypotheses in items:
            thm_name = self._auto_name(theorem)
            hyp_str = (
                (" " + " ".join(f"({h})" for h in hypotheses)) if hypotheses else ""
            )
            parts.append(
                f"\ntheorem {thm_name}{hyp_str} : {theorem} := by\n  sorry\n"
            )
        goals_path.write_text("\n".join(parts), encoding="utf-8")
        elan_bin = str(Path.home() / ".elan" / "bin")
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}
        subprocess.run(
            ["git", "add", self.PROOF_GOALS_FILE],
            cwd=self.lean_project_path, check=True, capture_output=True, env=env,
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.lean_project_path, capture_output=True,
        )
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", f"batch: {len(items)} theorems"],
                cwd=self.lean_project_path, check=True, capture_output=True, env=env,
            )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.lean_project_path, capture_output=True, text=True, check=True,
        )
        commit_hash = result.stdout.strip()
        logger.info(
            "Prepared %d theorems in ProofGoals.lean, commit %s",
            len(items), commit_hash[:8],
        )
        return commit_hash

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_git_repo(self):
        git_dir = self.lean_project_path / ".git"
        if not git_dir.exists():
            logger.info("Initialising git repo in %s", self.lean_project_path)
            subprocess.run(
                ["git", "init"], cwd=self.lean_project_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "lean@prover.local"],
                cwd=self.lean_project_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Lean Prover"],
                cwd=self.lean_project_path, check=True, capture_output=True
            )
            # Initial commit so HEAD is valid.
            subprocess.run(
                ["git", "add", "-A"], cwd=self.lean_project_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=self.lean_project_path, check=True, capture_output=True
            )
            logger.info("Git repo initialised with initial commit")

    def _subprocess_check(self, lean_src: str, timeout: int = 2400) -> bool:
        """
        Verify a proof by writing ProofGoals.lean and running `lake build TheoremProver`.

        TIMING: each call takes ~26 min (the Lean runtime must load all Mathlib
        module interfaces from disk, even with prebuilt oleans). This is a hard
        lower bound imposed by Lean 4's module loading, not something we can optimize.

        This path exists solely as a correctness check when GITHUB_ACCESS_TOKEN is
        unavailable. For interactive proof search, use LeanDojo's Dojo — it keeps
        the Lean process alive and pays this startup cost only once per session.
        """
        goals_path = self.lean_project_path / "ProofGoals.lean"
        elan_bin = Path.home() / ".elan" / "bin"
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}
        original = goals_path.read_text(encoding="utf-8") if goals_path.exists() else None
        try:
            goals_path.write_text(lean_src, encoding="utf-8")
            logger.info("Running lake build TheoremProver (timeout=%ds)...", timeout)
            result = subprocess.run(
                ["lake", "build", "TheoremProver"],
                cwd=self.lean_project_path,
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            if result.returncode != 0:
                logger.debug("lake build failed:\n%s", (result.stdout + result.stderr)[:600])
                return False
            combined = (result.stdout + result.stderr).lower()
            return "error:" not in combined
        except subprocess.TimeoutExpired:
            logger.warning("lake build timed out after %ds", timeout)
            return False
        except Exception as e:
            logger.debug("Subprocess check error: %s", e)
            return False
        finally:
            # Restore ProofGoals.lean to its previous state (or delete if new).
            if original is not None:
                goals_path.write_text(original, encoding="utf-8")
            elif goals_path.exists():
                goals_path.unlink()

    @staticmethod
    def _auto_name(theorem: str) -> str:
        """Generate a stable theorem name from the goal string."""
        h = hashlib.sha1(theorem.encode()).hexdigest()[:8]
        return f"goal_{h}"


# ── Proof session ──────────────────────────────────────────────────────────────

class _ProofSession:
    """
    Context manager wrapping a single Dojo tactic session.

    Workflow on __enter__:
      1. Write (or update) ProofGoals.lean with the theorem declaration.
      2. git add + commit so LeanDojo can see the file at a known commit.
      3. Call lean_dojo.trace() on the local repo (uses disk cache — fast on repeat).
      4. Retrieve the TracedTheorem and open a Dojo context.
    """

    def __init__(
        self,
        theorem: str,
        hypotheses: list[str],
        thm_name: str,
        lean_project_path: Path,
        proof_goals_file: str,
        tactic_timeout: int,
    ):
        self._theorem = theorem
        self._hypotheses = hypotheses
        self._thm_name = thm_name
        self._lean_project_path = lean_project_path
        self._proof_goals_file = proof_goals_file
        self._tactic_timeout = tactic_timeout
        self._dojo = None
        self._state = None
        self.is_complete = False

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "_ProofSession":
        if not _check_lean_dojo():
            raise ImportError("lean_dojo not installed. Run: pip install lean-dojo")
        from lean_dojo import LeanGitRepo, trace, Theorem, ProofFinished

        # 1. Ensure theorem is in ProofGoals.lean.
        # If prepare_theorem_batch() was called before this session, the file
        # already contains all theorems and we skip the write (reusing the commit).
        # Otherwise write just this theorem (single-theorem flow).
        goals_path = self._lean_project_path / self._proof_goals_file
        thm_decl = f"theorem {self._thm_name}"
        current_text = goals_path.read_text(encoding="utf-8") if goals_path.exists() else ""
        if thm_decl not in current_text:
            lean_src = _build_lean_source(
                self._theorem, self._hypotheses, self._thm_name
            )
            goals_path.write_text(lean_src, encoding="utf-8")
            logger.debug("Wrote %s:\n%s", goals_path, lean_src)

        # 2. Commit so LeanDojo can reference a specific SHA.
        # If nothing changed (theorem already committed), this is a no-op and
        # returns the existing HEAD hash — reusing the cached trace.
        commit_hash = self._commit_file(self._proof_goals_file)

        # 3. Trace the project (cached after first run).
        # Pass the absolute path directly — LeanDojo normalizes this as RepoType.LOCAL.
        # Do NOT use "file://..." prefix: normalize_url() calls os.path.abspath() on it,
        # which prepends cwd (since "file:..." doesn't start with "/"), then normpath
        # collapses "///" → "/", corrupting the path.
        elan_bin = str(Path.home() / ".elan" / "bin")
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}
        repo_url = str(self._lean_project_path)
        repo = LeanGitRepo(url=repo_url, commit=commit_hash)
        logger.info("Tracing repo %s @ %s", repo_url, commit_hash[:8])
        # build_deps=False: only trace ProofGoals.lean (not all 8500 mathlib files).
        # For premise retrieval, use pre-traced mathlib from the LeanDojo cache instead.
        traced_repo = trace(repo, build_deps=False)

        # 4. Get the traced theorem.
        # get_traced_file() requires traced_files_graph (only built with build_deps=True).
        # Since we use build_deps=False, look up directly from traced_repo.traced_files.
        rel_file = Path(self._proof_goals_file)
        traced_file = next(
            (tf for tf in traced_repo.traced_files
             if Path(tf.path) == rel_file or Path(tf.path).name == rel_file.name),
            None,
        )
        if traced_file is None:
            available = [str(tf.path) for tf in traced_repo.traced_files]
            raise RuntimeError(
                f"LeanDojo could not find traced file: {rel_file}. "
                f"Available: {available}"
            )

        traced_thm = next(
            (t for t in traced_file.get_traced_theorems()
             if t.theorem.full_name == self._thm_name),
            None,
        )
        if traced_thm is None:
            raise RuntimeError(
                f"Theorem '{self._thm_name}' not found in traced file. "
                f"Available: {[t.theorem.full_name for t in traced_file.get_traced_theorems()]}"
            )

        # 5. Open Dojo.
        import lean_dojo as ld
        dojo_ctx = ld.Dojo(traced_thm.theorem, timeout=self._tactic_timeout)
        self._dojo, self._state = dojo_ctx.__enter__()
        self._dojo_ctx = dojo_ctx

        self.is_complete = isinstance(self._state, ld.ProofFinished)
        return self

    def __exit__(self, *args):
        if hasattr(self, "_dojo_ctx") and self._dojo_ctx is not None:
            try:
                self._dojo_ctx.__exit__(*args)
            except Exception as e:
                logger.debug("Dojo cleanup error: %s", e)

    # ── Tactic application ─────────────────────────────────────────────────────

    def apply_tactic(self, tactic: str) -> StepResult:
        """Apply a tactic and return the result."""
        if self._dojo is None:
            raise RuntimeError("Session not entered — use as context manager.")

        from lean_dojo import ProofFinished, LeanError, ProofGivenUp

        try:
            new_state = self._dojo.run_tac(self._state, tactic)
        except Exception as e:
            return StepResult(success=False, next_state="", error_message=str(e))

        if isinstance(new_state, ProofFinished):
            self._state = new_state
            self.is_complete = True
            return StepResult(success=True, next_state="", is_complete=True)

        if isinstance(new_state, (LeanError, ProofGivenUp)):
            return StepResult(
                success=False, next_state="", error_message=str(new_state)
            )

        # Successful tactic — update state and return new proof state string.
        self._state = new_state
        return StepResult(success=True, next_state=new_state.pp)

    def current_state_str(self) -> str:
        """Return the current proof state as a string."""
        if self._state is None or self.is_complete:
            return "No goals"
        return getattr(self._state, "pp", str(self._state))

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _commit_file(self, relative_path: str) -> str:
        """git add + commit the file; return the commit hash."""
        elan_bin = str(Path.home() / ".elan" / "bin")
        env = {**os.environ, "PATH": f"{elan_bin}:{os.environ.get('PATH', '')}"}
        cwd = self._lean_project_path

        subprocess.run(["git", "add", relative_path], cwd=cwd, check=True,
                       capture_output=True, env=env)
        # Only commit if there are staged changes.
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=cwd, capture_output=True
        )
        if diff.returncode != 0:  # returncode 1 = there are staged changes
            subprocess.run(
                ["git", "commit", "-m", f"theorem: {self._thm_name}"],
                cwd=cwd, check=True, capture_output=True, env=env,
            )

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()


# ── Lean source builder ────────────────────────────────────────────────────────

def _build_lean_source(
    theorem: str,
    hypotheses: list[str],
    name: str,
    proof_tactics: list[str] | None = None,
) -> str:
    """
    Build a complete .lean file containing the theorem declaration.

    If proof_tactics is None/empty, uses `sorry` as the placeholder proof
    (LeanDojo tracing requires a tactic proof — Dojo replaces it with its REPL tactic).
    If proof_tactics is provided, emits them as the actual proof (for subprocess verification).
    """
    hyp_str = ""
    if hypotheses:
        hyp_str = " " + " ".join(f"({h})" for h in hypotheses)

    tactic_lines = proof_tactics if proof_tactics else ["sorry"]
    # Indent each tactic line by 2 spaces so they sit inside the `by` block.
    indented = "\n".join(f"  {t}" for t in tactic_lines)

    # Build without textwrap.dedent to preserve indentation exactly.
    return (
        "import Mathlib\n"
        "import Aesop\n"
        "\n"
        f"theorem {name}{hyp_str} : {theorem} := by\n"
        f"{indented}\n"
    )


# ── Proof state formatter ──────────────────────────────────────────────────────

def format_state_for_model(
    state_str: str,
    retrieved_premises: list[str] | None = None,
) -> str:
    """
    Format a proof state string for input to the ReProver tactic model.

    Without retrieved premises (default): return state_str unchanged.
    With premises: prepend <a>premise</a> tokens (ReProver retrieval format).
    """
    state = state_str.strip()
    if not retrieved_premises:
        return state
    premise_tokens = "".join(f"<a>{p}</a>" for p in retrieved_premises)
    return f"{premise_tokens}<s>{state}</s>"
