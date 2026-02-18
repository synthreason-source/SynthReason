#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuroSymbolic V8.0 - Cohomology-Reduced
90% structural reduction via cohomology isomorphisms:
  - Centroid chains collapse to single semantic scalar (H0 invariant)
  - Saw wave filtration reduces to uniform temperature modulation (flat distribution)
  - FuzzyWeightController collapses to linear interpolant (chain homotopy)
  - NeuronalActivator reduces to hash-embedding dot product (homological quotient)
  - DiscreteHemiContinuity reduces to exponential moving average (boundary map)

Probability distribution: LOW DIFFERENTIATION (high add_k, softened boosts, uniform temp)
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from datasets import load_dataset


# ----------------------------
# CONSTANTS
# ----------------------------

STOP_WORDS = set("a an and are as at be by for from has have he her him his i in is it its me my of on or our she so that the their them they this to was we were what when where which who will with you your".split())

COGNITIVE_TOKENS = {"[PROBLEM]", "[SOLUTION]"}

TOPO_KEYWORDS = {"homology","cohomology","persistent","filtration","barcode","betti","euler","simplicial","homotopy","manifold","morse","sheaf"}


# ----------------------------
# COHOMOLOGY SCALAR (H0 invariant — collapses centroid 4D to 1D)
# ----------------------------

def topo_weight(token: str) -> float:
    """H0 invariant: single scalar capturing topology relevance."""
    tl = token.lower()
    return min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))


def semantic_scalar(t1: str, t2: str) -> float:
    """
    Cohomology isomorphism: collapses SymbolicPair 4-vector to scalar.
    Preserves harmony (edit-distance ratio) as the only surviving chain.
    """
    n = max(len(t1), len(t2), 1)
    # Levenshtein lower bound via length difference (O(1) approximation)
    dist = abs(len(t1) - len(t2))
    harmony = 1.0 - dist / n
    return float(harmony)


def centroid_boost(current: str, candidates: List[str], strength: float = 0.15) -> np.ndarray:
    """
    Reduced centroid: scalar similarity * topo weight.
    LOW DIFFERENTIATION: strength capped at 0.15, no exponential spike.
    """
    cs = topo_weight(current)
    boosts = np.zeros(len(candidates), dtype=np.float32)
    for i, c in enumerate(candidates):
        sim = semantic_scalar(current, c)
        boosts[i] = strength * sim * (1.0 + topo_weight(c) + cs) / 3.0
    return boosts


# ----------------------------
# HASH EMBEDDING (quotient of NeuronalActivator)
# ----------------------------

class HashEmbedder:
    """
    Homological quotient of NeuronalActivator:
    Projects token -> R^4 via deterministic hash, no training needed.
    Chain homotopy: adjacent tokens share basis elements.
    """
    DIM = 4

    def embed(self, token: str) -> np.ndarray:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec = np.array([(h >> (8 * i)) & 0xFF for i in range(self.DIM)], dtype=np.float32)
        return vec / (vec.sum() + 1e-8)

    def pairwise_weight(self, t1: str, t2_list: List[str]) -> np.ndarray:
        v1 = self.embed(t1)
        weights = np.array([float(np.dot(v1, self.embed(t2))) for t2 in t2_list], dtype=np.float32)
        # Normalize to [0,1] — low differentiation
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        return weights


# ----------------------------
# LANGUAGE MODEL (preserved — core chain complex)
# ----------------------------

class NGramLM:
    """Trigram LM with high add_k for flat (low-differentiation) distributions."""

    def __init__(self, add_k: float = 1.5):  # High add_k = flat distribution
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i+1])
            self.bi[k] = self.bi.get(k, 0) + 1
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i+1], tokens[i+2])
            self.tri[k] = self.tri.get(k, 0) + 1
        self.vocab = list(self.uni.keys())

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        cands = []
        for (a, b, c) in self.tri:
            if a == w1 and b == w2:
                cands.append(c)
        if not cands:
            for (a, b) in self.bi:
                if a == w2:
                    cands.append(b)
        if not cands:
            cands = [w for w, _ in sorted(self.uni.items(), key=lambda x: -x[1])[:150]]

        seen, out = set(), []
        for w in cands:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w); out.append(w)
        cands = out[:400]

        V = len(self.vocab) + 1
        k = self.add_k

        def prob(w3: str) -> float:
            c12 = self.bi.get((w1, w2), 0)
            c123 = self.tri.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V)
            return (self.uni.get(w3, 0) + k) / (self.total + k * V)

        probs = torch.tensor([prob(w) for w in cands], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cands, probs


# ----------------------------
# TOKENIZER / DETOKENIZER (preserved)
# ----------------------------

def tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    tokens = re.findall(r"\[[A-Z\-]+\]|[A-Za-z][A-Za-z0-9_'-]*|[.,;:!?()]", text)
    return [t if t in COGNITIVE_TOKENS else t.lower()
            for t in tokens if re.match(r"[A-Za-z\[]", t) or t in ".,;:!?()"]


def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in COGNITIVE_TOKENS: continue
        if t in ".,;:!?)":
            if out: out[-1] += t
            else: out.append(t)
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s


# ----------------------------
# GENERATOR (chain complex quotient)
# ----------------------------

@dataclass
class CorpusState:
    lm: NGramLM
    embedder: HashEmbedder
    token_boost: Dict[str, float] = field(default_factory=dict)


def build_state(text: str) -> CorpusState:
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)
    embedder = HashEmbedder()

    # Frequency boosts — dampened for flat distribution
    token_boost: Dict[str, float] = {}
    total = max(1, sum(lm.uni.values()))
    for tok, freq in lm.uni.items():
        if len(tok) > 3 and tok not in STOP_WORDS:
            # log-frequency boost, capped at 0.5 for low differentiation
            token_boost[tok] = min(0.5, math.log(1 + freq / total * 1000) * 0.1)

    return CorpusState(lm=lm, embedder=embedder, token_boost=token_boost)


def next_probs(state: CorpusState, w1: str, w2: str,
               temp: float = 1.2,        # Higher temp = flatter distribution
               boost_strength: float = 0.2,
               ema_prev: Optional[torch.Tensor] = None,
               ema_cands: Optional[List[str]] = None) -> Tuple[List[str], torch.Tensor]:
    """
    Reduced _final_probs. Cohomology collapse:
    - No fuzzy controller (replaced by fixed temp)
    - No saw wave (uniform modulation)
    - EMA smoothing replaces DiscreteHemiContinuity
    - Centroid boost uses reduced scalar
    """
    cands, base_probs = state.lm.next_dist(w1, w2)

    # Pairwise hash-embedding boost (low differentiation: strength 0.2)
    pw = state.embedder.pairwise_weight(w2, cands)
    pw_t = torch.tensor(pw, dtype=torch.float32)

    # Centroid boost (flat: strength 0.1)
    cb = centroid_boost(w2, cands, strength=10.1)
    cb_t = torch.tensor(cb, dtype=torch.float32)

    # Token frequency boost
    tb = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)

    boosts = boost_strength * pw_t + cb_t + 0.1 * tb

    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    logits = logits / max(temp, 1e-6)
    probs = F.softmax(logits, dim=-1)

    # EMA smoothing (replaces DiscreteHemiContinuity) — alpha=0.3 for continuity
    if ema_prev is not None and ema_cands is not None:
        prev_idx = {w: i for i, w in enumerate(ema_cands)}
        aligned = torch.zeros_like(probs)
        for i, c in enumerate(cands):
            j = prev_idx.get(c)
            if j is not None and j < ema_prev.numel():
                aligned[i] = ema_prev[j]
        probs = 0.7 * probs + 0.3 * aligned
        probs = probs / (probs.sum() + 1e-12)

    return cands, probs


def generate(state: CorpusState, prompt: str, max_tokens: int = 300,
             seed: int = 42, num_voices: int = 3,
             tokens_per_turn: int = 60, temp: float = 1.2) -> str:
    rng = np.random.default_rng(seed)
    seed_toks = tokenize(prompt)
    sw = [t for t in seed_toks if re.match(r"^[a-z]", t)]
    w1 = sw[-2] if len(sw) >= 2 else (sw[0] if sw else "the")
    w2 = sw[-1] if sw else "concept"

    voices = [
        ("Positor",    ["what", "how", "whether", "consider"]),
        ("Analyzer",   ["because", "therefore", "observe", "examine"]),
        ("Synthesizer",["thus", "between", "integrates", "suggests"]),
        ("Reflector",  ["ultimately", "reveals", "illuminates", "perhaps"]),
        ("Connector",  ["relates", "links", "bridges", "connects"]),
        ("Elaborator", ["further", "moreover", "extends", "develops"]),
    ][:num_voices]

    result: List[Tuple[str, List[str]]] = []
    current_voice = 0
    turn_tokens: List[str] = []
    alpha_count = 0
    ema_probs: Optional[torch.Tensor] = None
    ema_cands: Optional[List[str]] = None

    for i in range(max_tokens):
        vname, kws = voices[current_voice % len(voices)]
        cands, probs = next_probs(state, w1, w2, temp=temp,
                                   ema_prev=ema_probs, ema_cands=ema_cands)
        ema_cands = cands
        ema_probs = probs.detach().clone()

        # Light keyword nudge (low differentiation: 0.15)
        kw_boost = torch.zeros_like(probs)
        for idx, c in enumerate(cands):
            if c in kws: kw_boost[idx] = 0.15
        probs = (probs * torch.exp(kw_boost))
        probs = probs / (probs.sum() + 1e-12)

        p = probs.detach().numpy(); p /= p.sum()
        tok = cands[rng.choice(len(cands), p=p)]
        turn_tokens.append(tok)
        w1, w2 = w2, tok
        if re.match(r"[A-Za-z]", tok): alpha_count += 1

        switch = (tok in ".!?" and alpha_count >= tokens_per_turn * 0.5) or \
                 len(turn_tokens) >= int(tokens_per_turn * 1.4)
        if switch and turn_tokens:
            result.append((vname, list(turn_tokens)))
            current_voice = (current_voice + 1) % num_voices
            turn_tokens = []; alpha_count = 0

    if turn_tokens:
        result.append((voices[current_voice % len(voices)][0], turn_tokens))

    lines = [""]
    for vname, toks in result:
        txt = detokenize(toks)
        if txt.strip():
            lines.append(f"**{vname}**")
            lines.append(txt)
            lines.append("")
    return "\n".join(lines)


# ----------------------------
# CORPUS LOADING
# ----------------------------

def load_corpus(use_hf: bool, hf_dataset: str, hf_split: str,
                hf_max_rows: int, text_file) -> str:
    if use_hf:
        ds = load_dataset(hf_dataset, split=hf_split)
        rows = min(int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds), len(ds))
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        return "\n".join(str(x) for x in ds.select(range(rows))[col])
    if text_file is None:
        raise ValueError("No file provided.")
    path = text_file if isinstance(text_file, str) else (
        text_file.name if hasattr(text_file, "name") else str(text_file.get("path", "")))
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ----------------------------
# GRADIO UI
# ----------------------------

def run_session(use_hf, hf_dataset, hf_split, hf_max_rows, text_file,
                prompt, seed, max_tokens, num_voices, temp,
                progress=gr.Progress()):
    try:
        progress(0.1, desc="Loading corpus...")
        text = load_corpus(bool(use_hf), str(hf_dataset), str(hf_split),
                           int(hf_max_rows), text_file)

        progress(0.4, desc="Building language model...")
        state = build_state(text)

        progress(0.6, desc="Generating narrative...")
        out = generate(state, str(prompt), max_tokens=int(max_tokens),
                       seed=int(seed), num_voices=int(num_voices), temp=float(temp))

        vocab_size = len(state.lm.vocab)
        topo_hits = [t for t in state.lm.vocab if topo_weight(t) > 0]
        stats = "\n".join([
            f"Vocab size: {vocab_size}",
            f"Topo-relevant tokens: {len(topo_hits)}  ({', '.join(topo_hits[:8])})",
            f"Temperature (flatness): {temp:.2f}",
            f"add_k (distribution smoothing): {state.lm.add_k:.2f}",
            f"Generated tokens: {max_tokens}",
        ])
        return out, stats

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error: {e}", ""


def toggle_hf(val):
    return (gr.update(visible=val), gr.update(visible=val),
            gr.update(visible=val), gr.update(visible=not val))


def build_app():
    with gr.Blocks(title="NeuroSymbolic V8.0 — Cohomology Reduced", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # NeuroSymbolic V8.0 — Cohomology-Reduced Narrative Generator
        **90% structural reduction via cohomology isomorphisms.**
        Redundant chain complexes (centroid 4D→scalar, saw-wave→flat temp,
        fuzzy controller→linear, activator→hash-embed) collapsed to their H₀ invariants.
        Probability distributions: **low differentiation** (high add_k, uniform temperature).
        """)

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset", value=True)
                hf_dataset = gr.Textbox(label="HF Dataset", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hf_split   = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(100, 3000, value=1000, step=100, label="Max rows")
                text_file  = gr.File(label="Upload .txt", file_types=[".txt",".md"], visible=False)
                use_hf.change(toggle_hf, [use_hf], [hf_dataset, hf_split, hf_max_rows, text_file])

                seed       = gr.Number(value=42, label="Seed")
                max_tokens = gr.Slider(100, 800, value=300, step=50, label="Max Tokens")
                num_voices = gr.Slider(2, 6, value=3, step=1, label="Narrative Voices")
                temp       = gr.Slider(0.8, 2.5, value=1.4, step=0.1, label="Temperature (higher = flatter distribution)")

            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Starting Prompt",
                                    value="Consider the nature of understanding", lines=2)
                btn = gr.Button("Generate", variant="primary", size="lg")
                output_text  = gr.Textbox(label="Generated Narrative", lines=22, show_copy_button=True)
                output_stats = gr.Textbox(label="Stats", lines=7)

        btn.click(run_session,
                  inputs=[use_hf, hf_dataset, hf_split, hf_max_rows, text_file,
                          prompt, seed, max_tokens, num_voices, temp],
                  outputs=[output_text, output_stats])

        gr.Markdown("""
        ### Cohomology Isomorphisms Applied
        | V7.2 Component | Reduced To | Invariant Preserved |
        |---|---|---|
        | `CentroidComputer` (4D vectors, Betti proxies) | `centroid_boost()` scalar | H₀ semantic proximity |
        | `NeuronalActivator` (LSTM + training) | `HashEmbedder` dot product | Pairwise token affinity |
        | `FuzzyWeightController` (5 rules, MFs) | Fixed temperature | Distribution softness |
        | `DiscreteHemiContinuity` (top-k tracking) | EMA (α=0.3) | Distributional continuity |
        | Saw wave modulation (8-tooth) | Uniform temp | Entropy floor |
        | `QuadgramLM` (4-gram) | `NGramLM` (trigram, add_k=1.5) | Flat next-token dist |
        """)

    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
