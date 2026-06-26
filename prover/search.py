"""
Best-first proof search.

Algorithm:
  - Maintain a priority queue of (neg_log_prob, SearchNode).
  - Each node holds the current proof state and the tactic sequence used to reach it.
  - At each step: pop the best node, query the tactic model for top-k candidates,
    submit each to Lean via LeanDojo, push successful states back onto the queue.
  - Terminate when a node has no remaining goals (proof complete) or timeout/depth exceeded.

Two operating modes:
  Interactive (full):  Uses LeanDojo's Dojo for tactic-by-tactic feedback.
                       Requires GITHUB_ACCESS_TOKEN. Lean process stays alive
                       between tactics so startup cost (~2-3 min) is paid once.
  Whole-proof (fast):  Generates complete proof candidates with beam search /
                       sampling, verifies each candidate with a subprocess call.
                       No token required; each Lean call takes 2-5 min cold-start.
                       Best for short proofs where top-k=1 often works.

This is greedy best-first search, not MCTS. MCTS can be added later.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .lean_interface import LeanInterface, StepResult, format_state_for_model
from .tactic_model import TacticModel, TacticCandidate, CausalLMTacticModel, DeepSeekProverModel

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    """Returned by ProofSearch.prove()."""
    proof: Optional[str]     # tactic proof string (e.g. "intro n\nsimp"), or None
    verified: bool           # did Lean accept the proof?
    steps: list[tuple[str, str]]  # list of (state, tactic) pairs
    search_nodes_expanded: int
    elapsed_seconds: float
    error: str = ""          # set if something failed unexpectedly
    root_tactics: list[dict] = field(default_factory=list)
    # Each entry: {tactic, log_prob, elaboration: "complete"|"success"|"error",
    #              next_state (if success), error_message (if error)}


@dataclass(order=True)
class _SearchNode:
    """Node in the proof search tree."""
    neg_log_prob: float             # priority (lower = better, for min-heap)
    state: str = field(compare=False)
    tactic_history: list[str] = field(compare=False, default_factory=list)
    state_history: list[str] = field(compare=False, default_factory=list)
    depth: int = field(compare=False, default=0)


class ProofSearch:
    """
    End-to-end proof search: model + search + Lean verification.

    Args:
        model_path:    Path to a local HuggingFace checkpoint, or a HuggingFace
                       model id. Defaults to the small ReProver checkpoint.
        lean_project:  Path to the Lean 4 project directory (must have lakefile.lean).
        top_k:         Number of tactic candidates to generate per step.
        device:        Torch device override ('cpu', 'cuda', 'mps').

    Example:
        prover = ProofSearch(
            model_path="models/pretrained/leandojo-lean4-tacgen-byt5-small",
            lean_project="./lean_project",
        )
        result = prover.prove("∀ n : ℕ, n + 0 = n", hypotheses=[])
        print(result.proof)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        lean_project: str | Path = "./lean_project",
        top_k: int = 32,
        device: Optional[str] = None,
        model=None,
        model_type: str = "byt5",
        load_in_4bit: bool = False,
    ):
        """
        model:        Pre-constructed model object (TacticModel, DeepSeekProverModel, etc.).
                      If provided, model_path and model_type are ignored.
        model_type:   "byt5"    — ByT5-small ReProver (default)
                      "deepseek" — DeepSeek-Prover-V1.5-RL (7B, needs GPU)
                      "causal"   — generic CausalLM (set model_path to HF model id)
        load_in_4bit: Load DeepSeek model in 4-bit (bitsandbytes). Reduces VRAM from
                      ~14GB to ~3.5GB; allows running on 16GB V100.
        """
        self.top_k = top_k
        if model is not None:
            self._model = model
        elif model_type == "deepseek":
            self._model = DeepSeekProverModel(model_id=model_path, device=device,
                                               load_in_4bit=load_in_4bit)
        elif model_type == "causal":
            self._model = CausalLMTacticModel(model_id=str(model_path), device=device)
        else:
            self._model = TacticModel(model_path=model_path, device=device)
        self._lean = LeanInterface(lean_project_path=lean_project)

    # ── Public API ─────────────────────────────────────────────────────────────

    def prove(
        self,
        theorem: str,
        hypotheses: list[str] | None = None,
        timeout: float = 60.0,
        max_depth: int = 20,
        retrieved_premises: list[str] | None = None,
    ) -> ProofResult:
        """
        Attempt to prove `theorem` using best-first tactic search.

        theorem:             Goal string in Lean 4 syntax, e.g. "∀ n : ℕ, n + 0 = n"
        hypotheses:          List of hypothesis strings, e.g. ["n : ℕ", "h : n > 0"]
        timeout:             Wall-clock timeout in seconds.
        max_depth:           Maximum tactic sequence length.
        retrieved_premises:  Optional retrieved mathlib premises (used by ReProver retriever).
                             Leave None to skip retrieval-augmentation.
        """
        hypotheses = hypotheses or []
        if isinstance(self._model, DeepSeekProverModel):
            return self._prove_deepseek_whole_proof(theorem, hypotheses, timeout, max_depth)
        return self._prove_best_first(theorem, hypotheses, timeout, max_depth, retrieved_premises)

    def _prove_deepseek_whole_proof(
        self,
        theorem: str,
        hypotheses: list[str],
        timeout: float,
        max_depth: int,
    ) -> ProofResult:
        """
        Whole-proof generation mode for DeepSeek-Prover-V1.5-RL.

        DeepSeek was trained on Lean 4 file completion, not step-by-step chat.
        Strategy: generate top_k complete proof scripts BEFORE opening the REPL,
        then try each script sequentially inside one Dojo session (each rejected
        tactic leaves the REPL state unchanged).

        For each depth level, we only try tactics from scripts whose prefix
        matches what has been applied so far — this keeps the search coherent
        across multi-tactic proofs without requiring multiple REPL sessions.
        """
        t_total_start = time.monotonic()
        _root_tactics: list[dict] = []
        nodes_expanded = 0

        # Generate complete proof scripts (GPU inference, before REPL starts)
        proof_scripts = self._model.generate_proofs(
            theorem=theorem,
            hypotheses=hypotheses,
            n=self.top_k,
            max_new_tokens=256,
        )
        logger.info("DeepSeek generated %d scripts for '%s...'",
                    len(proof_scripts), theorem[:50])
        for i, s in enumerate(proof_scripts[:5]):
            logger.debug("  script %d: %s", i, s)

        if not proof_scripts:
            return ProofResult(proof=None, verified=False, steps=[],
                               search_nodes_expanded=0,
                               elapsed_seconds=time.monotonic() - t_total_start,
                               error="DeepSeek generated no proof scripts")

        try:
            with self._lean.open_proof(theorem, hypotheses) as session:
                # Reset timer AFTER REPL opens — timeout counts only tactic search,
                # not model inference and Lean trace initialization (first theorem
                # takes 3-5 min cold-cache; that cost is not the model's fault).
                t_start = time.monotonic()

                # Scripts that are still "alive" (prefix so far has been valid)
                active_scripts: list[list[str]] = list(proof_scripts)
                applied: list[str] = []
                state_history: list[str] = [session.current_state_str()]

                for depth in range(max_depth):
                    if time.monotonic() - t_start > timeout:
                        break

                    # Collect the next tactic from each active script at this depth
                    candidates_at_depth: list[str] = []
                    seen_at_depth: set[str] = set()
                    for script in active_scripts:
                        if depth < len(script):
                            t = script[depth]
                            if t not in seen_at_depth:
                                seen_at_depth.add(t)
                                candidates_at_depth.append(t)

                    if not candidates_at_depth:
                        break

                    nodes_expanded += 1
                    is_root_node = depth == 0
                    succeeded_tactic: str | None = None

                    for tactic in candidates_at_depth:
                        if time.monotonic() - t_start > timeout:
                            break

                        result = session.apply_tactic(tactic)

                        if is_root_node:
                            entry: dict = {"tactic": tactic, "log_prob": -float(candidates_at_depth.index(tactic))}
                            if result.is_complete:
                                entry["elaboration"] = "complete"
                            elif result.success:
                                entry["elaboration"] = "success"
                                entry["next_state"] = result.next_state
                            else:
                                entry["elaboration"] = "error"
                                entry["error_message"] = result.error_message
                            _root_tactics.append(entry)

                        if result.is_complete:
                            tactic_seq = applied + [tactic]
                            return ProofResult(
                                proof="\n".join(tactic_seq),
                                verified=True,
                                steps=list(zip(state_history, tactic_seq)),
                                search_nodes_expanded=nodes_expanded,
                                elapsed_seconds=time.monotonic() - t_total_start,
                                root_tactics=_root_tactics,
                            )

                        if result.success:
                            # First successful tactic at this depth wins;
                            # filter active scripts to those using this tactic.
                            succeeded_tactic = tactic
                            applied.append(tactic)
                            state_history.append(result.next_state)
                            active_scripts = [s for s in active_scripts
                                              if depth < len(s) and s[depth] == tactic]
                            break  # move to next depth

                    if succeeded_tactic is None:
                        # No tactic at this depth advanced the proof
                        break

        except Exception as e:
            logger.exception("Unexpected error during DeepSeek proof search")
            return ProofResult(proof=None, verified=False, steps=[],
                               search_nodes_expanded=nodes_expanded,
                               elapsed_seconds=time.monotonic() - t_total_start,
                               error=str(e), root_tactics=_root_tactics)

        return ProofResult(proof=None, verified=False, steps=[],
                           search_nodes_expanded=nodes_expanded,
                           elapsed_seconds=time.monotonic() - t_total_start,
                           root_tactics=_root_tactics)

    def _prove_best_first(
        self,
        theorem: str,
        hypotheses: list[str],
        timeout: float,
        max_depth: int,
        retrieved_premises: list[str] | None,
    ) -> ProofResult:
        """Best-first proof search for ByT5-small and other step-by-step models."""
        t_open_start = time.monotonic()
        nodes_expanded = 0
        _root_tactics: list[dict] = []

        try:
            with self._lean.open_proof(theorem, hypotheses) as session:
                initial_state = session.current_state_str()
                # Reset timer AFTER Dojo opens so the timeout counts only tactic
                # search, not the ~90s Lean REPL startup (Mathlib olean loading).
                t_start = time.monotonic()

                if session.is_complete:
                    return ProofResult(
                        proof="",
                        verified=True,
                        steps=[],
                        search_nodes_expanded=0,
                        elapsed_seconds=time.monotonic() - t_open_start,
                    )

                heap: list[_SearchNode] = []
                root = _SearchNode(
                    neg_log_prob=0.0,
                    state=initial_state,
                    tactic_history=[],
                    state_history=[initial_state],
                    depth=0,
                )
                heapq.heappush(heap, root)
                while heap:
                    if time.monotonic() - t_start > timeout:
                        logger.info("Proof search timed out after %.1fs", timeout)
                        break

                    node = heapq.heappop(heap)

                    if node.depth >= max_depth:
                        continue

                    nodes_expanded += 1
                    is_root_node = nodes_expanded == 1
                    logger.debug(
                        "Expanding node depth=%d, nodes_so_far=%d\n  state: %s",
                        node.depth, nodes_expanded, node.state[:120],
                    )

                    model_input = format_state_for_model(node.state, retrieved_premises)
                    candidates = self._model.predict_tactics(model_input, top_k=self.top_k)

                    for cand in candidates:
                        if time.monotonic() - t_start > timeout:
                            break

                        result = session.apply_tactic(cand.tactic)

                        if is_root_node:
                            entry: dict = {"tactic": cand.tactic, "log_prob": cand.log_prob}
                            if result.is_complete:
                                entry["elaboration"] = "complete"
                            elif result.success:
                                entry["elaboration"] = "success"
                                entry["next_state"] = result.next_state
                            else:
                                entry["elaboration"] = "error"
                                entry["error_message"] = result.error_message
                            _root_tactics.append(entry)

                        if result.is_complete:
                            tactic_seq = node.tactic_history + [cand.tactic]
                            state_seq = list(zip(node.state_history, tactic_seq))
                            proof_str = "\n".join(tactic_seq)
                            return ProofResult(
                                proof=proof_str,
                                verified=True,
                                steps=state_seq,
                                search_nodes_expanded=nodes_expanded,
                                elapsed_seconds=time.monotonic() - t_open_start,
                                root_tactics=_root_tactics,
                            )

                        if result.success:
                            child = _SearchNode(
                                neg_log_prob=node.neg_log_prob + (-cand.log_prob),
                                state=result.next_state,
                                tactic_history=node.tactic_history + [cand.tactic],
                                state_history=node.state_history + [result.next_state],
                                depth=node.depth + 1,
                            )
                            heapq.heappush(heap, child)

        except Exception as e:
            logger.exception("Unexpected error during proof search")
            return ProofResult(
                proof=None,
                verified=False,
                steps=[],
                search_nodes_expanded=nodes_expanded,
                elapsed_seconds=time.monotonic() - t_open_start,
                error=str(e),
                root_tactics=_root_tactics,
            )

        return ProofResult(
            proof=None,
            verified=False,
            steps=[],
            search_nodes_expanded=nodes_expanded,
            elapsed_seconds=time.monotonic() - t_open_start,
            root_tactics=_root_tactics,
        )

    def verify_proof(
        self,
        theorem: str,
        hypotheses: list[str] | None = None,
        proof_tactics: list[str] = [],
        verify_timeout: int = 2400,
    ) -> bool:
        """
        Verify that `proof_tactics` constitutes a complete, accepted proof.

        Uses `lake build TheoremProver` subprocess — no GITHUB_ACCESS_TOKEN required.
        Each call takes ~26 min because Lean must load all ~8,500 Mathlib module
        interfaces from disk (unavoidable with subprocess approach).

        For practical proof search use prove() with GITHUB_ACCESS_TOKEN set —
        LeanDojo's Dojo keeps Lean alive and pays this cost once per session.
        """
        return self._lean.verify_proof(
            theorem=theorem,
            hypotheses=hypotheses or [],
            proof_tactics=proof_tactics,
            name=None,
        )

    def verify_proofs_batch(
        self,
        items: list[tuple[str, list[str], list[str], str]],
    ) -> list[bool]:
        """
        Verify multiple proofs in one lake build call.
        items: [(theorem, hypotheses, proof_tactics, thm_name), ...]
        """
        return self._lean.verify_proofs_batch(items)

    def prepare_theorem_batch(
        self,
        items: list[tuple[str, list[str]]],
    ) -> str:
        """
        Write all theorems to ProofGoals.lean and commit once so subsequent
        prove() calls share one cached LeanDojo trace (avoids re-trace per theorem).
        Returns the shared commit hash.

        items: [(theorem, hypotheses), ...]
        """
        return self._lean.prepare_theorem_batch(items)

    def batch_prove(
        self,
        theorems: list[tuple[str, list[str]]],
        timeout: float = 60.0,
        max_depth: int = 20,
    ) -> list[ProofResult]:
        """Prove a list of (theorem, hypotheses) pairs sequentially."""
        results = []
        for thm, hyps in theorems:
            logger.info("Proving: %s", thm)
            r = self.prove(thm, hyps, timeout=timeout, max_depth=max_depth)
            results.append(r)
            status = "SUCCESS" if r.verified else "FAILED"
            logger.info("  %s (%.1fs, %d nodes)", status, r.elapsed_seconds, r.search_nodes_expanded)
        return results
