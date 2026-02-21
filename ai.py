#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V8.6 - Length-Dependent Topology Dot Products

Key upgrade (this revision): topology dot products now scale with token length.

Length-Dependent Topology:
  - Embedding dimension DIM scales with word length (longer words → higher-dim space)
  - topo_weight() scales with char-length, rewarding morphologically rich tokens
  - shift_magnitude scales with length (longer words get stronger frame-shift)
  - agreement_bonus scales with length (longer words need stronger cross-frame consensus)
  - A length-weighted topology kernel modulates the final dot-product combination

This means short/simple words (cat, dog) use compact 2-4D embeddings with mild
topology influence, while long/complex words (cohomology, reconstruction) use
up to 12D embeddings with much stronger topological modulation.
"""

from __future__ import annotations

import re
import math
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gradio as gr
import torch
import torch.nn.functional as F
from datasets import load_dataset

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(
    "a an and are as at be by for from has have he her him his i in is it its me my of on or our "
    "she so that the their them they this to was we were what when where which who will with you your"
    .split()
)
COGNITIVE_TOKENS = {"[PROBLEM]", "[SOLUTION]"}
TOPO_KEYWORDS = {
    "homology", "cohomology", "persistent", "filtration", "barcode",
    "betti", "euler", "simplicial", "homotopy", "manifold", "morse", "sheaf"
}

_VOWELS = set("aeiouy")

_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "ou",
    "ed", "nd", "to", "or", "ea", "ti", "es", "st", "ar", "nt",
    "is", "al", "it", "as", "ha", "et", "se", "ng", "le", "of",
}

_LATINATE_PREFIXES = {
    "pre", "post", "anti", "auto", "bio", "geo", "hyper", "hypo",
    "inter", "intra", "micro", "macro", "meta", "mono", "multi",
    "neo", "non", "over", "poly", "pseudo", "semi", "sub", "super",
    "trans", "ultra", "uni", "dis", "mis", "un", "re", "de",
}
_LATINATE_SUFFIXES = {
    "tion", "sion", "ment", "ness", "ity", "ism", "ist", "ize",
    "ise", "ful", "less", "ous", "ious", "eous", "ance", "ence",
    "able", "ible", "ive", "ative", "ology", "ography", "ician",
    "ation", "ization", "isation",
}

_EARLY_WORDS: Dict[str, float] = {
    "cat": 2.5, "dog": 2.5, "mom": 2.2, "dad": 2.2, "baby": 2.8,
    "ball": 2.6, "cup": 2.7, "eye": 2.4, "ear": 2.5, "nose": 2.6,
    "hat": 2.8, "shoe": 2.9, "bed": 2.7, "hot": 3.0, "cold": 3.1,
    "big": 3.0, "small": 3.2, "run": 3.1, "eat": 2.9, "go": 2.5,
    "yes": 2.4, "no": 2.3, "hi": 2.2, "bye": 2.3, "more": 2.8,
    "up": 2.6, "down": 2.8, "in": 2.5, "out": 2.7, "on": 2.6,
    "off": 2.8, "want": 2.7, "help": 3.0, "play": 2.9, "walk": 3.0,
    "look": 2.8, "see": 2.5, "hear": 2.8, "think": 3.5, "know": 3.4,
    "hand": 2.9, "foot": 2.9, "head": 2.7, "face": 2.8, "name": 3.2,
    "home": 3.0, "door": 3.1, "car": 2.8, "tree": 3.0, "book": 3.2,
}

# ────────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT TOPOLOGY PARAMETERS
# ────────────────────────────────────────────────────────────────────────────

# DIM for embedding: scales from DIM_MIN to DIM_MAX based on word length
DIM_MIN = 2          # shortest words (len ≤ 2)
DIM_MAX = 12         # longest words (len ≥ LENGTH_CEIL)
LENGTH_CEIL = 14     # word length at which DIM saturates at DIM_MAX
SHIFT_MAG_MIN = 0.05   # shift magnitude for short words
SHIFT_MAG_MAX = 0.35   # shift magnitude for long words
AGREEMENT_BONUS_MIN = 0.10  # agreement bonus for short words
AGREEMENT_BONUS_MAX = 0.60  # agreement bonus for long words


def length_alpha(word: str, ceil: int = LENGTH_CEIL) -> float:
    """
    Normalised length factor α ∈ [0, 1].
    α = 0 for very short words, 1 for words at/beyond LENGTH_CEIL chars.
    Uses a smooth sigmoid-like curve so medium-length words are partially scaled.
    """
    n = len(word.strip())
    # Soft sigmoid centered at ceil/2
    mid = ceil / 2.0
    return float(1.0 / (1.0 + math.exp(-0.55 * (n - mid))))


def length_dim(word: str) -> int:
    """
    Embedding dimension for a word, scaled by length.
    Short words → DIM_MIN; long words → DIM_MAX.
    Always even (for cleaner hash decomposition).
    """
    α = length_alpha(word)
    raw = DIM_MIN + α * (DIM_MAX - DIM_MIN)
    return max(DIM_MIN, int(round(raw / 2) * 2))  # round to nearest even


def length_shift_mag(word: str) -> float:
    """Shift magnitude scaled by word length."""
    α = length_alpha(word)
    return SHIFT_MAG_MIN + α * (SHIFT_MAG_MAX - SHIFT_MAG_MIN)


def length_agreement_bonus(word: str) -> float:
    """Agreement bonus scaled by word length."""
    α = length_alpha(word)
    return AGREEMENT_BONUS_MIN + α * (AGREEMENT_BONUS_MAX - AGREEMENT_BONUS_MIN)


def length_topo_kernel(word: str) -> float:
    """
    A length-dependent weight for how strongly topology modulates the dot product.
    Short words: topology has little influence.
    Long words: topology strongly modulates the combined score.

    Returns a multiplier in [0.05, 1.0].
    """
    α = length_alpha(word)
    # Topology kernel: exponential ramp
    return float(0.05 + 0.95 * (α ** 1.5))


# ────────────────────────────────────────────────────────────────────────────
# AoA DATASET
# ────────────────────────────────────────────────────────────────────────────
AOA_DATASET_URL = (
    "https://norare.clld.org/contributions/Kuperman-2012-AoA/English-AoA-30K.csv"
)
AOA_COL_WORD = "Word"
AOA_COL_AOA  = "AoA"


def load_aoa_dataset(max_rows: int = 35_000) -> Dict[str, float]:
    try:
        df = pd.read_csv(AOA_DATASET_URL, nrows=max_rows)
        if AOA_COL_WORD not in df.columns or AOA_COL_AOA not in df.columns:
            return {}
        df = df[[AOA_COL_WORD, AOA_COL_AOA]].dropna()
        return {
            str(w).strip().lower(): float(a)
            for w, a in zip(df[AOA_COL_WORD], df[AOA_COL_AOA])
        }
    except Exception:
        return {}


# ────────────────────────────────────────────────────────────────────────────
# WORD-AGE CALCULATOR
# ────────────────────────────────────────────────────────────────────────────
def _count_syllables(word: str) -> int:
    w = word.lower().rstrip("e")
    count = sum(
        1
        for i, c in enumerate(w)
        if c in _VOWELS and (i == 0 or w[i - 1] not in _VOWELS)
    )
    return max(1, count)


def _morpheme_complexity(word: str) -> float:
    w = word.lower()
    score = 0.0
    for p in _LATINATE_PREFIXES:
        if w.startswith(p) and len(w) > len(p) + 2:
            score += 0.25
            break
    for s in _LATINATE_SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            score += 0.25 * (1 + len(s) / 6)
            break
    return min(1.0, score)


def _bigram_familiarity(word: str) -> float:
    w = word.lower()
    if len(w) < 2:
        return 0.5
    bigrams = [w[i:i + 2] for i in range(len(w) - 1)]
    return sum(1 for b in bigrams if b in _COMMON_BIGRAMS) / len(bigrams)


def _ortho_neighborhood_size(word: str, aoa_dict: Dict[str, float]) -> int:
    w = word.lower()
    n = len(w)
    count = 0
    for cand in aoa_dict:
        if len(cand) == n and cand != w:
            diffs = sum(a != b for a, b in zip(w, cand))
            if diffs == 1:
                count += 1
                if count >= 20:
                    break
    return count


def calculate_word_age(
    word: str,
    aoa: Dict[str, float],
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> float:
    w = word.lower().strip()
    if not w or not w[0].isalpha():
        return 10.0
    if w in aoa:
        return aoa[w]
    if w in _EARLY_WORDS:
        return _EARLY_WORDS[w]

    n_chars   = len(w)
    n_syl     = _count_syllables(w)
    morph     = _morpheme_complexity(w)
    bigram_f  = _bigram_familiarity(w)
    neigh     = _ortho_neighborhood_size(w, aoa)

    if corpus_freq and w in corpus_freq:
        rel_freq = corpus_freq[w] / max(corpus_total, 1)
        log_freq = math.log(1 + rel_freq * 1_000_000)
    else:
        log_freq = 0.0

    intercept = 8.5
    β_len     = 0.30
    β_syl     = 0.55
    β_morph   = 2.80
    β_big     = 1.60
    β_freq    = 0.18
    β_neigh   = 0.40

    estimated = (
        intercept
        + β_len   * (n_chars - 5)
        + β_syl   * (n_syl  - 2)
        + β_morph * morph
        - β_big   * bigram_f
        - β_freq  * log_freq
        - β_neigh * math.log(1 + neigh)
    )

    return float(max(2.0, min(20.0, estimated)))


def word_age(
    aoa: Dict[str, float],
    token: str,
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> float:
    return calculate_word_age(token, aoa, corpus_freq, corpus_total)


def age_continuity_boost(age1: float, age2: float, strength: float = 0.12) -> float:
    d = abs(age1 - age2)
    early = min(age1, age2, 8.0) / 8.0
    return float(strength * math.exp(-d / 3.0) * early)


# ────────────────────────────────────────────────────────────────────────────
# COHOMOLOGY SCALARS — now length-dependent
# ────────────────────────────────────────────────────────────────────────────
def topo_weight(token: str) -> float:
    """
    Topology weight, now length-dependent.

    Base keyword score is amplified by the token's length-topology kernel:
    longer tokens are more likely to carry topological meaning (e.g. "cohomology"
    vs "co"), so we scale the raw keyword hit by length_topo_kernel().
    """
    tl = token.lower()
    base = min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))
    # Even without a keyword hit, longer words get a mild topology presence
    length_presence = 0.05 * length_alpha(token)
    raw = base + length_presence
    return float(min(1.0, raw * length_topo_kernel(token)))


def semantic_scalar(t1: str, t2: str) -> float:
    n = max(len(t1), len(t2), 1)
    dist = abs(len(t1) - len(t2))
    return float(1.0 - dist / n)


def centroid_boost(
    aoa: Dict[str, float],
    current: str,
    candidates: List[str],
    strength: float = 0.10,
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> np.ndarray:
    cs_topo = topo_weight(current)
    cs_age  = word_age(aoa, current, corpus_freq, corpus_total)
    boosts  = np.zeros(len(candidates), dtype=np.float32)
    for i, c in enumerate(candidates):
        sim = semantic_scalar(current, c)
        tw  = (topo_weight(c) + cs_topo) * 0.5
        ab  = age_continuity_boost(cs_age, word_age(aoa, c, corpus_freq, corpus_total))
        boosts[i] = strength * sim * (1.0 + tw + ab) / 3.0
    return boosts


# ────────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT DOUBLE ENTENDRE EMBEDDER
# ────────────────────────────────────────────────────────────────────────────
class LengthDependentEmbedder:
    """
    Length-dependent double-entendre dot product.

    For each (w1, w2, candidate) triple:
      - DIM is determined by the CANDIDATE's length (the thing being scored)
      - shift_mag and agreement_bonus scale with the ANCHOR word (w2) length
      - A length-topology kernel modulates the final combined score

    Two passes:
      pass1 = dot(embed(w2, dim), embed(c, dim))
      pass2 = dot(embed(w2, dim) + shift(w1, dim, mag), embed(c, dim))

    combined = topo_kernel(c) * [0.5*(norm01(p1)+norm01(p2)) + bonus*min(p1,p2)]
             + (1 - topo_kernel(c)) * 0.5*(norm01(p1)+norm01(p2))

    This means topology modulation only kicks in for longer/more complex candidates.
    """

    def embed(self, token: str, dim: Optional[int] = None) -> np.ndarray:
        """Hash-based embedding in `dim`-dimensional space (length-dependent if dim=None)."""
        d = dim if dim is not None else length_dim(token)
        # Use MD5 for the first 16 bytes, SHA256 for more if needed
        raw_bytes = hashlib.sha256(token.encode("utf-8")).digest()  # 32 bytes
        # Repeat to fill d bytes
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def shift_vector(self, token: str, dim: int, magnitude: float) -> np.ndarray:
        """Length-aware shift: magnitude already pre-scaled by caller."""
        raw_bytes = hashlib.md5(token.encode("utf-8")).digest()  # 16 bytes
        repeated = (raw_bytes * ((dim // 16) + 2))[:dim]
        vec = np.array(list(repeated), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-8)) * magnitude

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-12)

    def length_dependent_weights(
        self,
        w1: str,
        w2: str,
        candidates: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute length-dependent double-entendre weights for each candidate.

        Returns (pass1_norm, pass2_norm, combined) all in [0,1].
        """
        N = len(candidates)
        pass1_raw = np.zeros(N, dtype=np.float32)
        pass2_raw = np.zeros(N, dtype=np.float32)
        topo_kernels = np.zeros(N, dtype=np.float32)

        # Anchor parameters depend on w2's length
        anchor_shift_mag  = length_shift_mag(w2)
        anchor_agree_bonus = length_agreement_bonus(w2)

        for i, c in enumerate(candidates):
            # Each candidate uses its own length-dependent DIM
            dim = length_dim(c)

            # Embed w2 and candidate in the candidate's dimensional space
            e_w2 = self.embed(w2, dim=dim)
            e_c  = self.embed(c,  dim=dim)

            # Shift uses w1 in the same dim
            shift = self.shift_vector(w1, dim=dim, magnitude=anchor_shift_mag)
            e_w2_shifted = e_w2 + shift
            norm_s = float(e_w2_shifted.sum())
            e_w2_shifted = e_w2_shifted / (abs(norm_s) + 1e-8)

            pass1_raw[i] = float(np.dot(e_w2, e_c))
            pass2_raw[i] = float(np.dot(e_w2_shifted, e_c))
            topo_kernels[i] = length_topo_kernel(c)

        p1 = self._norm01(pass1_raw)
        p2 = self._norm01(pass2_raw)

        de_score = np.minimum(p1, p2)

        # Base combination (same for all candidates)
        base_combined = 0.5 * (p1 + p2)

        # Agreement bonus scales with w2 length (anchor-level parameter)
        agreement_part = float(anchor_agree_bonus) * de_score

        # Topology kernel gates how much the agreement bonus applies
        # Short candidates: topology kernel ≈ 0 → agreement bonus suppressed
        # Long candidates: topology kernel ≈ 1 → full agreement bonus
        combined = base_combined + topo_kernels * agreement_part
        combined = self._norm01(combined)

        return p1, p2, combined


# Keep the old name as an alias for backwards compatibility
DoubleEntendreEmbedder = LengthDependentEmbedder


# ────────────────────────────────────────────────────────────────────────────
# LANGUAGE MODEL
# ────────────────────────────────────────────────────────────────────────────
class NGramLM:
    def __init__(self, add_k: float = 1.5):
        self.add_k  = float(add_k)
        self.uni:   Dict[str, int]                    = {}
        self.bi:    Dict[Tuple[str, str], int]        = {}
        self.tri:   Dict[Tuple[str, str, str], int]   = {}
        self.vocab: List[str]                         = []
        self.total  = 0

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            self.bi[k] = self.bi.get(k, 0) + 1
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.tri[k] = self.tri.get(k, 0) + 1
        self.vocab = list(self.uni.keys())

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        cands: List[str] = []
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
                seen.add(w)
                out.append(w)
        cands = out[:400]
        V = len(self.vocab) + 1
        k = self.add_k

        def prob(w3: str) -> float:
            c12  = self.bi.get((w1, w2), 0)
            c123 = self.tri.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V)
            return (self.uni.get(w3, 0) + k) / (self.total + k * V)

        probs = torch.tensor([prob(w) for w in cands], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cands, probs


# ────────────────────────────────────────────────────────────────────────────
# TOKENIZER / DETOKENIZER
# ────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"\[[A-Z\-]+\]|[A-Za-z][A-Za-z0-9_'-]*|[.,;:!?()]")


def tokenize(text: str) -> List[str]:
    text = text.replace("\\n", " ")
    tokens = _TOKEN_RE.findall(text)
    out: List[str] = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            out.append(t)
        elif re.match(r"[A-Za-z]", t):
            out.append(t.lower())
        elif t in ".,;:!?()":
            out.append(t)
    return out


def detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            continue
        if t in ".,;:!?)":
            if out:
                out[-1] += t
            else:
                out.append(t)
        elif t == "(":
            out.append(t)
        else:
            if out and out[-1].endswith("("):
                out[-1] += t
            else:
                out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s


# ────────────────────────────────────────────────────────────────────────────
# CORPUS STATE
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusState:
    lm:          NGramLM
    embedder:    LengthDependentEmbedder
    aoa:         Dict[str, float]
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int]   = field(default_factory=dict)
    corpus_total: int             = 1


def build_state(text: str, aoa: Dict[str, float]) -> CorpusState:
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)
    embedder = LengthDependentEmbedder()

    total = max(1, sum(lm.uni.values()))
    token_boost: Dict[str, float] = {}
    for tok, freq in lm.uni.items():
        if len(tok) > 3 and tok not in STOP_WORDS and re.match(r"^[a-z]", tok):
            token_boost[tok] = min(0.5, math.log(1 + (freq / total) * 1000.0) * 0.1)

    return CorpusState(
        lm=lm,
        embedder=embedder,
        aoa=aoa,
        token_boost=token_boost,
        corpus_freq=lm.uni,
        corpus_total=total,
    )


# ────────────────────────────────────────────────────────────────────────────
# GENERATOR
# ────────────────────────────────────────────────────────────────────────────
def next_probs(
    state: CorpusState,
    w1: str,
    w2: str,
    temp: float = 1.2,
    de_strength: float = 0.18,
    ema_prev: Optional[torch.Tensor] = None,
    ema_cands: Optional[List[str]] = None,
) -> Tuple[List[str], torch.Tensor]:
    cands, base_probs = state.lm.next_dist(w1, w2)

    # Length-dependent double-entendre dot-product weights
    _, _, de_combined = state.embedder.length_dependent_weights(
        w1=w1, w2=w2, candidates=cands,
    )
    de_t = torch.tensor(de_combined, dtype=torch.float32)

    cb = centroid_boost(
        state.aoa, w2, cands,
        strength=0.10,
        corpus_freq=state.corpus_freq,
        corpus_total=state.corpus_total,
    )
    cb_t  = torch.tensor(cb, dtype=torch.float32)
    tb    = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)

    w2_age  = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array(
        [age_continuity_boost(
            w2_age,
            word_age(state.aoa, c, state.corpus_freq, state.corpus_total),
        ) for c in cands],
        dtype=np.float32,
    )
    age_t = torch.tensor(age_arr, dtype=torch.float32)

    # Length-dependent topology also modulates the centroid boost
    topo_kernels = torch.tensor(
        [length_topo_kernel(c) for c in cands], dtype=torch.float32
    )
    topo_cb = cb_t * (0.5 + 0.5 * topo_kernels)  # short words: 0.5x; long: 1x boost

    boosts = float(de_strength) * de_t + topo_cb + 0.10 * tb + 0.15 * age_t
    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    logits = logits / max(float(temp), 1e-6)
    probs  = F.softmax(logits, dim=-1)

    if ema_prev is not None and ema_cands is not None:
        prev_idx = {w: i for i, w in enumerate(ema_cands)}
        aligned  = torch.zeros_like(probs)
        for i, c in enumerate(cands):
            j = prev_idx.get(c)
            if j is not None and j < int(ema_prev.numel()):
                aligned[i] = ema_prev[j]
        probs = 0.7 * probs + 0.3 * aligned
        probs = probs / (probs.sum() + 1e-12)

    return cands, probs


def generate(
    state:           CorpusState,
    prompt:          str,
    max_tokens:      int  = 300,
    seed:            int  = 42,
    num_voices:      int  = 3,
    tokens_per_turn: int  = 60,
    temp:            float = 1.2,
) -> str:
    rng = np.random.default_rng(int(seed))
    seed_toks = tokenize(prompt)
    sw = [t for t in seed_toks if re.match(r"^[a-z]", t)]
    w1 = sw[-2] if len(sw) >= 2 else (sw[0] if sw else "the")
    w2 = sw[-1] if sw else "concept"

    voices = [
    ("Positor", [
        "what", "how", "when", "why", "where", "whether", "imagine", "suppose", "consider", "define",
        "state", "pose", "query", "assert", "envision", "propose", "determine", "specify", "outline", "identify",
        "explore", "focus", "express", "declare", "suggest"
    ]),
    ("Analyzer", [
        "because", "therefore", "thus", "hence", "examine", "observe", "inspect", "compare", "contrast", "deduce",
        "infer", "evaluate", "scrutinize", "measure", "determine", "diagnose", "trace", "test", "quantify", "assess",
        "prove", "analyze", "dissect", "uncover", "establish"
    ]),
    ("Synthesizer", [
        "thus", "between", "integrates", "suggests", "combines", "merges", "connects", "unifies", "fuses", "blends",
        "resolves", "harmonizes", "links", "joins", "bridges", "reconciles", "aligns", "connects", "coalesces", "balances",
        "melds", "incorporates", "relates", "summarizes", "converges"
    ]),
    ("Reflector", [
        "ultimately", "reveals", "illuminates", "perhaps", "maybe", "indicates", "implies", "evokes", "signifies", "suggests",
        "contemplates", "meditates", "distills", "uncovers", "concludes", "infers", "recognizes", "appreciates", "ponders", "rethinks",
        "interprets", "acknowledges", "realizes", "wonders", "discerns"
    ]),
    ("Connector", [
        "relates", "links", "bridges", "connects", "associates", "correlates", "binds", "ties", "concatenates", "couples",
        "unites", "joins", "interweaves", "crosses", "maps", "compares", "contextualizes", "interrelates", "interlaces", "binds",
        "matches", "aggregates", "corresponds", "equates", "aligns"
    ]),
    ("Elaborator", [
        "further", "moreover", "extends", "develops", "expands", "deepens", "broadens", "amplifies", "details", "illustrates",
        "enhances", "supports", "enriches", "reiterates", "strengthens", "continues", "adds", "accentuates", "clarifies", "builds",
        "reinforces", "emphasizes", "substantiates", "heightens", "extends"
    ]),
][: max(1, int(num_voices))]

    result: List[Tuple[str, List[str]]] = []
    current_voice = 0
    turn_tokens:  List[str] = []
    alpha_count   = 0
    ema_probs:  Optional[torch.Tensor] = None
    ema_cands:  Optional[List[str]]    = None

    for _ in range(int(max_tokens)):
        vname, kws = voices[current_voice % len(voices)]
        cands, probs = next_probs(
            state, w1, w2,
            temp=float(temp),
            ema_prev=ema_probs,
            ema_cands=ema_cands,
        )
        ema_cands = cands
        ema_probs = probs.detach().clone()

        kw_boost = torch.zeros_like(probs)
        for idx, c in enumerate(cands):
            if c in kws:
                kw_boost[idx] = 0.15
        probs = probs * torch.exp(kw_boost)
        probs = probs / (probs.sum() + 1e-12)

        p   = probs.detach().cpu().numpy()
        p   = p / (p.sum() + 1e-12)
        tok = cands[int(rng.choice(len(cands), p=p))]
        turn_tokens.append(tok)
        w1, w2 = w2, tok

        if re.match(r"[A-Za-z]", tok):
            alpha_count += 1

        switch = (
            (tok in ".!?" and alpha_count >= tokens_per_turn * 0.5)
            or (len(turn_tokens) >= int(tokens_per_turn * 1.4))
        )
        if switch and turn_tokens:
            result.append((vname, list(turn_tokens)))
            current_voice = (current_voice + 1) % len(voices)
            turn_tokens   = []
            alpha_count   = 0

    if turn_tokens:
        vname, _ = voices[current_voice % len(voices)]
        result.append((vname, turn_tokens))

    lines: List[str] = []
    for vname, toks in result:
        txt = detokenize(toks).strip()
        if txt:
            lines.append(f"### {vname}")
            lines.append(txt)
            lines.append("")
    return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ────────────────────────────────────────────────────────────────────────────
def load_corpus(
    use_hf: bool,
    hf_dataset: str,
    hf_split: str,
    hf_max_rows: int,
    text_file,
) -> str:
    if use_hf:
        ds   = load_dataset(hf_dataset, split=hf_split)
        rows = min(int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds), len(ds))
        col  = "text" if "text" in ds.column_names else ds.column_names[0]
        return "\n".join(str(x) for x in ds.select(range(rows))[col])
    if text_file is None:
        raise ValueError("No file provided.")
    path = text_file if isinstance(text_file, str) else (
        text_file.name if hasattr(text_file, "name")
        else str(text_file.get("path", ""))
    )
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ────────────────────────────────────────────────────────────────────────────
# AGE + LENGTH ANALYSIS HELPER
# ────────────────────────────────────────────────────────────────────────────
def age_and_length_analysis(
    state: CorpusState,
    top_n: int = 10,
) -> str:
    alpha_vocab = [t for t in state.lm.vocab if t.isalpha() and t not in STOP_WORDS]
    if not alpha_vocab:
        return "No alpha vocabulary found."

    ages = {
        t: word_age(state.aoa, t, state.corpus_freq, state.corpus_total)
        for t in alpha_vocab
    }
    sorted_ages = sorted(ages.items(), key=lambda x: x[1])
    youngest = sorted_ages[:top_n]
    oldest   = sorted_ages[-top_n:][::-1]

    normed   = sum(1 for t in alpha_vocab if t in state.aoa)
    computed = len(alpha_vocab) - normed
    mean_age = sum(ages.values()) / max(1, len(ages))
    sd_age   = math.sqrt(
        sum((v - mean_age) ** 2 for v in ages.values()) / max(1, len(ages))
    )

    # Length-dependent topology analysis
    topo_by_len: Dict[int, List[Tuple[str, float]]] = {}
    for t in alpha_vocab:
        d   = length_dim(t)
        tw  = topo_weight(t)
        α   = length_alpha(t)
        kern = length_topo_kernel(t)
        if d not in topo_by_len:
            topo_by_len[d] = []
        topo_by_len[d].append((t, tw * kern))

    dim_summary_lines = []
    for d in sorted(topo_by_len.keys()):
        entries = topo_by_len[d]
        avg_tw  = sum(v for _, v in entries) / max(1, len(entries))
        top_ex  = sorted(entries, key=lambda x: -x[1])[:3]
        ex_str  = ", ".join(f"{w}({v:.2f})" for w, v in top_ex)
        dim_summary_lines.append(
            f"  DIM={d:2d} | {len(entries):4d} words | mean topo×kernel={avg_tw:.3f} | top: {ex_str}"
        )

    lines = [
        f"Alpha vocab: {len(alpha_vocab)} words",
        f"  Normed (Kuperman): {normed}",
        f"  Calculated (estimated): {computed}",
        f"  Mean AoA: {mean_age:.2f} yr  SD: {sd_age:.2f} yr",
        "",
        f"Youngest {top_n} (earliest acquired):",
        "  " + ", ".join(f"{w}({a:.1f})" for w, a in youngest),
        "",
        f"Oldest {top_n} (latest acquired):",
        "  " + ", ".join(f"{w}({a:.1f})" for w, a in oldest),
        "",
        "── Length-Dependent Topology Dot-Product Summary ──",
        f"  DIM range: {DIM_MIN}–{DIM_MAX}  |  length ceil: {LENGTH_CEIL}",
        f"  shift_mag range: {SHIFT_MAG_MIN:.2f}–{SHIFT_MAG_MAX:.2f}",
        f"  agreement_bonus range: {AGREEMENT_BONUS_MIN:.2f}–{AGREEMENT_BONUS_MAX:.2f}",
        "",
    ] + dim_summary_lines

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ────────────────────────────────────────────────────────────────────────────
def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows,
    text_file, prompt, seed, max_tokens, num_voices, temp, tokens_per_turn,
    progress=gr.Progress(),
):
    try:
        progress(0.05, desc="Loading AoA dataset (Kuperman 2012)…")
        aoa = load_aoa_dataset()

        progress(0.15, desc="Loading corpus…")
        text = load_corpus(bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows), text_file)

        progress(0.40, desc="Building language model…")
        state = build_state(text, aoa)

        progress(0.60, desc="Analysing word ages + length topology…")
        age_stats = age_and_length_analysis(state)

        progress(0.70, desc="Generating narrative…")
        out_md = generate(
            state, str(prompt),
            max_tokens=int(max_tokens),
            seed=int(seed),
            num_voices=int(num_voices),
            temp=float(temp),
            tokens_per_turn=int(tokens_per_turn),
        )

        vocab_size  = len(state.lm.vocab)
        topo_hits   = [t for t in state.lm.vocab if topo_weight(t) > 0.05]
        normed      = sum(1 for t in state.lm.vocab if t.isalpha() and t in aoa)
        alpha_total = sum(1 for t in state.lm.vocab if t.isalpha())

        # Sample length-dim distribution
        alpha_vocab = [t for t in state.lm.vocab if t.isalpha()]
        dim_counts: Dict[int, int] = {}
        for t in alpha_vocab:
            d = length_dim(t)
            dim_counts[d] = dim_counts.get(d, 0) + 1
        dim_dist = "  " + "  ".join(f"DIM{d}:{n}" for d, n in sorted(dim_counts.items()))

        stats = "\n".join([
            f"Vocab size: {vocab_size}",
            f"AoA normed (Kuperman exact):    {normed}/{alpha_total}",
            f"AoA calculated (feature model): {alpha_total - normed}/{alpha_total}",
            f"Topo tokens (length-weighted):  {len(topo_hits)}",
            f"Temperature: {float(temp):.2f}  |  add_k: {state.lm.add_k:.2f}",
            f"Generated tokens: {int(max_tokens)}",
            "",
            "── Length→DIM distribution ──",
            dim_dist,
            "",
            "── Word-Age + Length-Topology Analysis ──",
            age_stats,
        ])
        return out_md, stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"### Error\n{e}", ""


def toggle_hf(val):
    return (
        gr.update(visible=val),
        gr.update(visible=val),
        gr.update(visible=val),
        gr.update(visible=not val),
    )


def build_app():
    with gr.Blocks(
        title="NeuroSymbolic V8.6 — Length-Dependent Topology Dot Products",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# NeuroSymbolic V8.6 — Length-Dependent Topology Dot Products\n"
            "The topology dot-product now **scales with word/token length**.\n\n"
            "| Parameter | Short words | Long words |\n"
            "|-----------|------------|------------|\n"
            "| Embedding DIM | 2–4 | 8–12 |\n"
            "| Shift magnitude | 0.05 | 0.35 |\n"
            "| Agreement bonus | 0.10 | 0.60 |\n"
            "| Topo kernel gate | ~0.05 | ~1.0 |\n\n"
            "**Effect:** Short words (cat, big) have compact, lightly modulated dot products. "
            "Long words (cohomology, reconstruction) use high-dimensional embeddings with strong "
            "topological agreement gating and large frame-shift vectors."
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf      = gr.Checkbox(label="Use Hugging Face Dataset", value=True)
                hf_dataset  = gr.Textbox(label="HF Dataset", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hf_split    = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(0, 3000, value=1000, step=100, label="Max rows")
                text_file   = gr.File(label="Upload .txt/.md", file_types=[".txt", ".md"], visible=False)
                use_hf.change(toggle_hf, [use_hf], [hf_dataset, hf_split, hf_max_rows, text_file])

                seed        = gr.Number(value=42,  label="Seed")
                max_tokens  = gr.Slider(100, 800, value=300, step=50, label="Max Tokens")
                num_voices  = gr.Slider(2, 6,    value=3,   step=1,  label="Narrative Voices")
                temp        = gr.Slider(0.8, 2.5, value=1.4, step=0.1, label="Temperature")
                tokens_per_turn = gr.Slider(20, 200, value=170, step=10, label="Tokens per Role")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Starting Prompt",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Generate", variant="primary", size="lg")
                gr.Markdown("## Generated Narrative (roles)")
                output_md    = gr.Markdown(value="")
                output_stats = gr.Textbox(label="Stats + Length-Topology Analysis", lines=25)

        btn.click(
            run_session,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows,
                    text_file, prompt, seed, max_tokens, num_voices, temp, tokens_per_turn],
            outputs=[output_md, output_stats],
        )

        gr.Markdown(
            "### Design Notes\n"
            "- `length_alpha(word)` → smooth sigmoid in [0,1] centered at half of `LENGTH_CEIL`\n"
            "- `length_dim(word)` → embedding dimension 2–12 (always even, rounded)\n"
            "- `length_topo_kernel(word)` → gates agreement bonus: short=0.05, long≈1.0\n"
            "- `topo_weight(word)` → keyword hit × length_topo_kernel (length-amplified)\n"
            "- `centroid_boost` modulated by topo_kernel: short words get 0.5× boost\n"
            "- Install: `pip install gradio datasets torch pandas numpy`"
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
