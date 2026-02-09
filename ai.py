#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import gradio as gr
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable, Set, Optional, Any

# ================================================================
# Utils
# ================================================================
def softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    z = x - np.max(x)
    ez = np.exp(z)
    s = ez.sum()
    return ez / (s if s > 0 else 1.0)

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def resolve_gradio_file_to_path(infile: Any) -> str:
    # Supports Gradio temp file objects and plain strings
    if infile is None:
        raise ValueError("No input file provided.")
    if isinstance(infile, str):
        return infile
    if hasattr(infile, "name") and isinstance(infile.name, str):
        return infile.name
    if isinstance(infile, dict) and "path" in infile:
        return str(infile["path"])
    if hasattr(infile, "path"):
        return str(infile.path)
    raise ValueError(f"Unsupported infile type: {type(infile)}")

# ================================================================
# Quine–McCluskey + Petrick
# API:
#   minimize_sop(n_bits, minterms, dontcares) -> implicants (bitstrings with '-' wildcards)
#   implicants_to_expr(implicants, varnames) -> string SOP
#   make_sop_evaluator(implicants, varorder) -> callable(bits[List[int]]) -> 0/1
# ================================================================
def _bits(n: int, width: int) -> str:
    return format(n, "0{}b".format(width))

def _can_combine(a: str, b: str) -> bool:
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            if x != "-" and y != "-":
                diff += 1
            else:
                return False
    return diff == 1

def _combine(a: str, b: str) -> str:
    out = []
    for x, y in zip(a, b):
        out.append(x if x == y else "-")
    return "".join(out)

def _hamming_ones(s: str) -> int:
    return sum(c == "1" for c in s)

def _covers(imp: str, m: str) -> bool:
    return all(x == "-" or x == y for x, y in zip(imp, m))

def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _iterate_combine(terms: List[str]) -> List[str]:
    current = _unique(terms)
    while True:
        buckets: Dict[int, List[str]] = {}
        for t in current:
            k = _hamming_ones(t.replace("-", "0"))
            buckets.setdefault(k, []).append(t)

        used = set()
        next_terms = []
        keys = sorted(buckets.keys())
        for k in keys:
            a_bucket = buckets.get(k, [])
            b_bucket = buckets.get(k + 1, [])
            for a in a_bucket:
                for b in b_bucket:
                    if _can_combine(a, b):
                        next_terms.append(_combine(a, b))
                        used.add(a)
                        used.add(b)

        primes = [t for t in current if t not in used]
        if not next_terms:
            return _unique(primes)
        current = _unique(next_terms + primes)

def _build_pi_chart(primes: List[str], minterms: List[str]) -> Dict[str, Set[str]]:
    cov: Dict[str, Set[str]] = {}
    for p in primes:
        covered = {m for m in minterms if _covers(p, m)}
        if covered:
            cov[p] = covered
    return cov

def _essential_primes(pi_chart: Dict[str, Set[str]], mset: Set[str]) -> Tuple[Set[str], Set[str]]:
    essentials = set()
    remaining = set(mset)
    while True:
        unique_map = defaultdict(list)
        for p, cols in pi_chart.items():
            for m in cols:
                if m in remaining:
                    unique_map[m].append(p)

        added = False
        for m, plist in unique_map.items():
            if len(plist) == 1:
                ep = plist[0]
                if ep not in essentials:
                    essentials.add(ep)
                    remaining -= pi_chart.get(ep, set())
                    added = True
        if not added:
            break
    return essentials, remaining

def _absorb_minimal(sets_: Set[frozenset]) -> Set[frozenset]:
    out = set(sets_)
    lst = list(sets_)
    for a in lst:
        for b in lst:
            if a != b and a.issuperset(b):
                out.discard(a)
                break
    return out

def _petrick(pi_chart: Dict[str, Set[str]], remaining: Set[str]) -> Set[str]:
    sums: List[Set[frozenset]] = []
    for m in remaining:
        choices = {frozenset([p]) for p, cols in pi_chart.items() if m in cols}
        sums.append(choices)

    if not sums:
        return set()

    prod: Set[frozenset] = {frozenset()}
    for s in sums:
        new_prod: Set[frozenset] = set()
        for term in prod:
            for choice in s:
                new_prod.add(term | choice)  # union
        prod = _absorb_minimal(new_prod)

    min_size = min(len(t) for t in prod)
    candidates = [t for t in prod if len(t) == min_size]

    def literal_cost(ps: Iterable[str]) -> int:
        return sum(sum(ch != "-" for ch in p) for p in ps)

    best = min(candidates, key=lambda s: literal_cost(s))
    return set(best)

def minimize_sop(n_bits: int, minterms: List[int], dontcares: List[int]) -> List[str]:
    on = sorted(set(minterms))
    dc = sorted(set(dontcares))
    if not (on or dc):
        return []

    base_terms = [_bits(m, n_bits) for m in (on + dc)]
    primes = _iterate_combine(base_terms)

    on_bits = [_bits(m, n_bits) for m in on]
    chart = _build_pi_chart(primes, on_bits)
    essentials, remaining = _essential_primes(chart, set(on_bits))
    cover_rest = _petrick(chart, remaining) if remaining else set()
    chosen = essentials | cover_rest
    return [p for p in primes if p in chosen]

def implicants_to_expr(implicants: List[str], varnames: List[str]) -> str:
    def cube_to_term(cube: str) -> str:
        lits = []
        for bit, name in zip(cube, varnames):
            if bit == "1":
                lits.append(name)
            elif bit == "0":
                lits.append(f"~{name}")
        return " & ".join(lits) if lits else "1"
    return " | ".join(cube_to_term(c) for c in implicants) if implicants else "0"

def make_sop_evaluator(implicants: List[str], varorder: List[str]):
    compiled = []
    for cube in implicants:
        req = []
        for i, b in enumerate(cube):
            if b == "-":
                continue
            req.append((i, b == "1"))
        compiled.append(req)

    def eval_row(bits: List[int]) -> int:
        for req in compiled:
            ok = True
            for idx, need_one in req:
                if (bits[idx] == 1) != need_one:
                    ok = False
                    break
            if ok:
                return 1
        return 0

    return eval_row

# ================================================================
# Automorphic Surjection Generator (instrumented)
# ================================================================
class SurjectionField:
    def __init__(self):
        self.map: Dict[Tuple[str, str], float] = {}
    def register(self, a, b, v): self.map[(a, b)] = float(v)
    def lookup(self, a, b): return self.map.get((a, b), None)
    def automorph(self):
        new_map = {}
        for (a, b), v in self.map.items():
            new_map[(b, a)] = v
            new_map[(a, b)] = v
        self.map = new_map
        return self

class SurjectionOps:
    def __init__(self, field=None):
        self.field = field or SurjectionField()
    def surject(self, u, v, a=None, b=None):
        u = np.asarray(u, float); v = np.asarray(v, float)
        n = min(len(u), len(v))
        if n == 0: return 0.5
        dot = float(np.dot(u[:n], v[:n]))
        nv2 = float(np.dot(v[:n], v[:n]) + 1e-9)
        corr = 1.0
        if a and b:
            val = self.field.lookup(a, b)
            if val is not None:
                corr = 0.7 + 0.6 * np.tanh(val)
        result = float(np.clip(0.5 * (np.tanh(corr * dot / nv2) + 1.0), 0, 1))
        return self.automorph_scalar(result)
    def automorph_scalar(self, x):
        return 1.0 - x if x > 0.5 else x

class WordFeatures:
    def __init__(self, tokens):
        self.freq = Counter(tokens)
        self.total = max(1, len(tokens))
        self.feature_cache: Dict[str, np.ndarray] = {}
    def vec(self, w):
        if w in self.feature_cache:
            return self.feature_cache[w]
        L = len(w)
        f = self.freq.get(w, 1)
        vec = np.array([
            L / 10.0,
            sum(c.isalpha() for c in w) / (L + 1.0),
            sum(c in "aeiou" for c in w) / (L + 1.0),
            np.log(f + 1.0) / np.log(self.total + 1.0),
            1.0 / (f + 1.0),
        ], float)
        vec = self.automorph_vector(vec)
        self.feature_cache[w] = vec
        return vec
    def automorph_vector(self, v):
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return v
        normalized = v / norm
        reflected = 2.0 * normalized - v
        return reflected

class SurjectionGenerator:
    def __init__(self, tokens, model):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())

        self.field = SurjectionField()
        self.ops = SurjectionOps(self.field)
        self.feat = WordFeatures(tokens)
        self._auto_pairs()

        self.generation_state: List[str] = []

        self._build_codomain_anchors(k=18)
        self.anchor_hits = np.zeros(len(self.anchors), dtype=int)

        self.alt_period = 14
        self.alpha_linear = 0.35
        self.beta_onto = 0.45

        self.qmc_logs: List[Dict] = []
        self.sim_thresh = 0.45
        self.align_thresh = 0.05
        self.pmin = 1e-12

    def _auto_pairs(self):
        big = Counter(zip(self.tokens[:-1], self.tokens[1:]))
        if not big:
            return
        m = max(big.values())
        for (a, b), c in big.items():
            self.field.register(a, b, c / m)
        self.field.automorph()

    def _build_codomain_anchors(self, k=8):
        counts = Counter(self.tokens)
        top = [w for w, _ in counts.most_common(max(2 * k, k + 4))]
        feats = []
        chosen = []
        for w in top:
            v = self.feat.vec(w)
            v = v / (np.linalg.norm(v) + 1e-9)
            if not feats:
                feats.append(v); chosen.append(w)
            else:
                dmin = min(np.linalg.norm(v - u) for u in feats)
                if dmin > 0.35:
                    feats.append(v); chosen.append(w)
            if len(feats) >= k:
                break
        while len(feats) < k and top:
            w = top[np.random.randint(len(top))]
            v = self.feat.vec(w)
            v = v / (np.linalg.norm(v) + 1e-9)
            feats.append(v); chosen.append(w)
        self.anchors = np.stack(feats, axis=0)
        self.anchor_tokens = chosen

    def _candidate_feat_matrix(self, cands: List[str]) -> np.ndarray:
        V = np.array([self.feat.vec(c) for c in cands], float)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        return V / norms

    def _nearest_anchor_idx(self, vec: np.ndarray) -> int:
        v = vec / (np.linalg.norm(vec) + 1e-9)
        sims = self.anchors @ v
        return int(np.argmax(sims))

    def _anchor_alignment_dist(self, cands: List[str], anchor_idx: int) -> np.ndarray:
        A = self.anchors[anchor_idx]
        C = self._candidate_feat_matrix(cands)
        sims = C @ A
        sims = np.maximum(sims, 0.0)
        if sims.max() < 1e-12:
            sims = np.ones_like(sims)
        sims = sims / (sims.sum() + 1e-9)
        return sims

    def _onto_reweight(self, cands: List[str]) -> Tuple[np.ndarray, int]:
        min_hits = self.anchor_hits.min()
        under = np.where(self.anchor_hits == min_hits)[0]
        aidx = int(under[len(self.generation_state) % len(under)])
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    def _linearize_toward_context(self, cands: List[str], context_words: Tuple[str, str]) -> Tuple[np.ndarray, int]:
        v1 = self.feat.vec(context_words[0])
        v2 = self.feat.vec(context_words[1])
        ctx = (v1 + v2) / 2.0
        aidx = self._nearest_anchor_idx(ctx)
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    def surjection_similarity(self, a, b):
        va, vb = self.feat.vec(a), self.feat.vec(b)
        score = self.ops.surject(va, vb, a, b)
        return self.automorph_similarity(score)

    def automorph_similarity(self, s):
        return float(s + np.sin(2 * np.pi * s) / 4.0)

    def automorph_state(self):
        if len(self.generation_state) < 2:
            return
        self.generation_state[-2], self.generation_state[-1] = (
            self.generation_state[-1],
            self.generation_state[-2],
        )

    def _bool_features_for_candidate(
        self,
        c: str,
        sim_norm: List[float],
        q_lin: np.ndarray,
        step: int,
        p_final: np.ndarray,
        cands: List[str],
    ) -> Dict[str, int]:
        idx = cands.index(c)
        s_norm = sim_norm[idx] if sim_norm else 0.0
        X0 = 1 if s_norm >= self.sim_thresh else 0

        top_idx = int(np.argmax(q_lin))
        X1 = 1 if (idx == top_idx or (q_lin[top_idx] - q_lin[idx]) <= self.align_thresh) else 0

        X2 = 1 if ((step + 1) % self.alt_period == 0) else 0

        min_hits = self.anchor_hits.min()
        under = np.where(self.anchor_hits == min_hits)[0]
        a_c = self._nearest_anchor_idx(self.feat.vec(c))
        X3 = 1 if a_c in under else 0

        X4 = 1 if p_final[idx] >= self.pmin else 0
        X5 = 1

        return {"X0": X0, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}

    def _log_qmc_row(self, X: Dict[str, int], accept: int):
        self.qmc_logs.append({"X": X, "Y": accept})

    def generate(self, seed: str, length=80, enable_qmc_logging=False):
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.tokens[len(words) % len(self.tokens)])
        seed_pair = tuple(words)
        if seed_pair not in self.model:
            seed_pair = self.keys[np.random.randint(len(self.keys))]

        out = list(seed_pair)
        self.generation_state = list(seed_pair)

        for step in range(int(length)):
            cands = self.model.get(seed_pair, [])
            if not cands:
                seed_pair = self.keys[np.random.randint(len(self.keys))]
                continue

            sim_scores = [self.surjection_similarity(out[-2], c) for c in cands]
            if not sim_scores:
                continue

            norm = (max(sim_scores) + 1e-9)
            sim_norm = [s / norm for s in sim_scores]
            base = softmax_np(np.array(sim_norm, dtype=np.float64))

            q_lin, _a_lin = self._linearize_toward_context(cands, (out[-2], out[-1]))
            p_lin = (1.0 - self.alpha_linear) * base + self.alpha_linear * q_lin

            if (step + 1) % self.alt_period == 0:
                q_onto, _a_onto = self._onto_reweight(cands)
                p = (1.0 - self.beta_onto) * p_lin + self.beta_onto * q_onto
            else:
                p = p_lin

            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()

            if enable_qmc_logging:
                for c in cands:
                    X = self._bool_features_for_candidate(c, sim_norm, q_lin, step, p, cands)
                    accept = int((X["X0"] and X["X4"]) and (X["X1"] or (X["X2"] and X["X3"])))
                    self._log_qmc_row(X, accept)

            next_word = np.random.choice(cands, p=p)

            v_next = self.feat.vec(next_word)
            a_chosen = self._nearest_anchor_idx(v_next)
            self.anchor_hits[a_chosen] += 1

            self.generation_state.append(next_word)
            if (step + 1) % 5 == 0:
                self.automorph_state()

            out.append(next_word)
            seed_pair = tuple(out[-2:])

        return " ".join(out)

# ================================================================
# QMC training + gated generation
# ================================================================
def learn_minimized_gate_from_logs(logs: List[Dict], varorder: List[str]) -> Tuple[List[str], str]:
    on: List[int] = []
    dc: List[int] = []
    for row in logs:
        X = row["X"]; y = row["Y"]
        bits = "".join(str(int(X[v])) for v in varorder)
        mi = int(bits, 2)
        if y == 1:
            on.append(mi)
    implicants = minimize_sop(len(varorder), on, dc)
    expr = implicants_to_expr(implicants, varorder)
    return implicants, expr

def generate_with_implicants(gen: SurjectionGenerator, seed: str, length=80, gate=None, varorder=None):
    words = seed.split()[:2]
    while len(words) < 2:
        words.append(gen.tokens[len(words) % len(gen.tokens)])
    seed_pair = tuple(words)
    if seed_pair not in gen.model:
        seed_pair = gen.keys[np.random.randint(len(gen.keys))]

    out = list(seed_pair)
    gen.generation_state = list(seed_pair)

    for step in range(int(length)):
        cands = gen.model.get(seed_pair, [])
        if not cands:
            seed_pair = gen.keys[np.random.randint(len(gen.keys))]
            continue

        sim_scores = [gen.surjection_similarity(out[-2], c) for c in cands]
        if not sim_scores:
            continue

        norm = (max(sim_scores) + 1e-9)
        sim_norm = [s / norm for s in sim_scores]
        base = softmax_np(np.array(sim_norm, dtype=np.float64))

        q_lin, _a_lin = gen._linearize_toward_context(cands, (out[-2], out[-1]))
        p_lin = (1.0 - gen.alpha_linear) * base + gen.alpha_linear * q_lin

        if (step + 1) % gen.alt_period == 0:
            q_onto, _a_onto = gen._onto_reweight(cands)
            p = (1.0 - gen.beta_onto) * p_lin + gen.beta_onto * q_onto
        else:
            p = p_lin

        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()

        mask = np.ones(len(cands), dtype=np.float64)
        if gate is not None and varorder is not None:
            for ci, c in enumerate(cands):
                X = gen._bool_features_for_candidate(c, sim_norm, q_lin, step, p, cands)
                bits = [int(X[v]) for v in varorder]
                mask[ci] = float(gate(bits))

        if mask.sum() <= 0:
            mask[np.argmax(p)] = 1.0

        p_masked = p * mask
        p_masked = p_masked / p_masked.sum()

        next_word = np.random.choice(cands, p=p_masked)

        v_next = gen.feat.vec(next_word)
        a_chosen = gen._nearest_anchor_idx(v_next)
        gen.anchor_hits[a_chosen] += 1

        gen.generation_state.append(next_word)
        if (step + 1) % 5 == 0:
            gen.automorph_state()

        out.append(next_word)
        seed_pair = tuple(out[-2:])

    return " ".join(out)

# ================================================================
# Model builder
# ================================================================
def build_ngram(tokens, n=2):
    m = defaultdict(list)
    for i in range(len(tokens) - n):
        m[tuple(tokens[i : i + n])].append(tokens[i + n])
    return m

# ================================================================
# Gradio app logic
# ================================================================
VARORDER = ["X0", "X1", "X2", "X3", "X4", "X5"]

def ui_load_corpus(infile, lowercase=True, max_tokens=0, rng_seed=0):
    if rng_seed and int(rng_seed) != 0:
        np.random.seed(int(rng_seed))

    path = resolve_gradio_file_to_path(infile)
    if not os.path.exists(path):
        raise gr.Error(f"File not found: {path}")

    text = read_text_file(path)
    if lowercase:
        text = text.lower()

    toks = text.split()
    if max_tokens and int(max_tokens) > 0:
        toks = toks[: int(max_tokens)]

    if len(toks) < 10:
        raise gr.Error("Corpus too small after tokenization (need at least ~10 tokens).")

    model = build_ngram(toks, 2)
    gen = SurjectionGenerator(toks, model)

    stats = (
        f"Loaded file: {os.path.basename(path)}\n"
        f"Tokens: {len(toks):,}\n"
        f"Bigram keys: {len(model):,}\n"
        f"Anchors: {len(gen.anchors)}"
    )
    return gen, None, None, stats

def ui_phase_a_collect(gen: SurjectionGenerator, seed: str, length: int):
    if gen is None:
        raise gr.Error("Load a corpus first.")
    gen.qmc_logs = []
    _ = gen.generate(seed, length=int(length), enable_qmc_logging=True)
    return f"Collected {len(gen.qmc_logs):,} candidate rows."

def ui_phase_b_learn(gen: SurjectionGenerator):
    if gen is None:
        raise gr.Error("Load a corpus first.")
    if not gen.qmc_logs:
        raise gr.Error("No logs yet. Run Phase A first.")

    implicants, expr = learn_minimized_gate_from_logs(gen.qmc_logs, VARORDER)
    gate = make_sop_evaluator(implicants, VARORDER)

    imp_txt = "\n".join(implicants) if implicants else "(none)"
    pretty = f"Implicants:\n{imp_txt}\n\nMinimized SOP:\n{expr}"
    return gate, pretty

def ui_phase_c_generate(gen: SurjectionGenerator, gate, seed: str, length: int):
    if gen is None:
        raise gr.Error("Load a corpus first.")
    if gate is None:
        raise gr.Error("No gate learned. Run Phase B first.")
    return generate_with_implicants(gen, seed, length=int(length), gate=gate, varorder=VARORDER)

def build_app():
    with gr.Blocks(title="Surjection + QMC Gate (Full)", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Automorphic Surjection Generator + QMC Gate (Gradio)\n"
            "Upload a corpus, collect boolean logs, learn a minimized SOP gate, then generate with gating."
        )

        gen_state = gr.State(None)   # holds SurjectionGenerator
        gate_state = gr.State(None)  # holds callable gate(bits)->0/1

        with gr.Row():
            with gr.Column(scale=1):
                infile = gr.File(label="Corpus file (.txt/.md/.py)", file_types=[".txt", ".md", ".py"])
                lowercase = gr.Checkbox(label="Lowercase corpus", value=True)
                max_tokens = gr.Number(label="Max tokens (0 = all)", value=0, precision=0)
                rng_seed = gr.Number(label="Numpy RNG seed (0 = leave)", value=0, precision=0)
                btn_load = gr.Button("Load corpus", variant="primary")

            with gr.Column(scale=2):
                corpus_stats = gr.Textbox(label="Corpus stats", lines=6)
                gate_info = gr.Textbox(label="Gate (learned SOP)", lines=8)

        gr.Markdown("## Phase A: Instructions")
        with gr.Row():
            seed_a = gr.Textbox(label="Seed", value="seed words")
            len_a = gr.Slider(50, 2000, value=400, step=10, label="Length")
            btn_a = gr.Button("Run Phase A", variant="secondary")
        phase_a_out = gr.Textbox(label="Phase A status", lines=2)

        gr.Markdown("## Phase B: Learn minimized gate")
        with gr.Row():
            btn_b = gr.Button("Run Phase B (QMC)", variant="secondary")

        gr.Markdown("## Phase C: Generate with gate")
        with gr.Row():
            seed_c = gr.Textbox(label="Seed", value="def run")
            len_c = gr.Slider(20, 2000, value=300, step=10, label="Length")
            btn_c = gr.Button("Run Phase C", variant="primary")
        out_txt = gr.Textbox(label="Output", lines=16)

        # Wiring
        btn_load.click(
            ui_load_corpus,
            inputs=[infile, lowercase, max_tokens, rng_seed],
            outputs=[gen_state, gate_state, gate_info, corpus_stats],
        )

        btn_a.click(
            ui_phase_a_collect,
            inputs=[gen_state, seed_a, len_a],
            outputs=[phase_a_out],
        )

        btn_b.click(
            ui_phase_b_learn,
            inputs=[gen_state],
            outputs=[gate_state, gate_info],
        )

        btn_c.click(
            ui_phase_c_generate,
            inputs=[gen_state, gate_state, seed_c, len_c],
            outputs=[out_txt],
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
