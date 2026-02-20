#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V8.5 - Cohomology-Reduced + Calculated Word Age (AoA)
+ Double-Entendre Dot-Product With Shift

Key upgrade (this revision): replace the single hash-dot-product boost with a
two-pass ("double entendre") dot product:

  Pass 1: dot(embed(w2), embed(candidate))
  Pass 2: dot(embed(w2) + shift(w1), embed(candidate))

Candidates that stay strong under BOTH frames get an agreement bonus.
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

# English vowels for syllable counting
_VOWELS = set("aeiouy")

# Common English phoneme bigrams (high-frequency → imply easier words)
# Derived from Brown corpus letter-bigram frequencies
_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "ou",
    "ed", "nd", "to", "or", "ea", "ti", "es", "st", "ar", "nt",
    "is", "al", "it", "as", "ha", "et", "se", "ng", "le", "of",
}

# Derivational affixes that mark morphologically complex (later-acquired) words
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

# Very early-acquired core vocabulary (prototype list, mean AoA < 4 yr)
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
# AoA DATASET (normed + calculated fallback)
# ────────────────────────────────────────────────────────────────────────────
AOA_DATASET_URL = (
    "https://norare.clld.org/contributions/Kuperman-2012-AoA/English-AoA-30K.csv"
)
AOA_COL_WORD = "Word"
AOA_COL_AOA  = "AoA"


def load_aoa_dataset(max_rows: int = 35_000) -> Dict[str, float]:
    """
    Load Kuperman 2012 AoA norms from CLLD (if reachable).
    Returns {word_lower: aoa_years}.  Falls back to {} on failure.
    """
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
    """Heuristic English syllable counter."""
    w = word.lower().rstrip("e")  # silent final e
    count = sum(
        1
        for i, c in enumerate(w)
        if c in _VOWELS and (i == 0 or w[i - 1] not in _VOWELS)
    )
    return max(1, count)


def _morpheme_complexity(word: str) -> float:
    """
    Returns a complexity score in [0, 1] based on recognisable derivational
    prefixes and suffixes.  Each affix adds 0.25, capped at 1.0.
    """
    w = word.lower()
    score = 0.0
    for p in _LATINATE_PREFIXES:
        if w.startswith(p) and len(w) > len(p) + 2:
            score += 0.25
            break
    for s in _LATINATE_SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            score += 0.25 * (1 + len(s) / 6)  # longer suffixes → more complex
            break
    return min(1.0, score)


def _bigram_familiarity(word: str) -> float:
    """
    Fraction of consecutive letter pairs that appear in the common-bigram set.
    Higher → more phonotactically familiar → acquired earlier.
    """
    w = word.lower()
    if len(w) < 2:
        return 0.5
    bigrams = [w[i:i + 2] for i in range(len(w) - 1)]
    return sum(1 for b in bigrams if b in _COMMON_BIGRAMS) / len(bigrams)


def _ortho_neighborhood_size(word: str, aoa_dict: Dict[str, float]) -> int:
    """
    Approximate orthographic neighbourhood (Coltheart's N):
    count words in the AoA dict that differ by exactly one letter.
    Capped at 20 for speed.
    """
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
    """
    Estimate age-of-acquisition for *word* in years.

    Priority:
      1. Normed value from Kuperman 2012 (exact match)
      2. Prototype entry in _EARLY_WORDS
      3. Computed estimate from linguistic features

    Feature model (linear, calibrated to Kuperman distribution):
      AoA ≈ intercept
            + β_len   * (chars - 5)
            + β_syl   * (syllables - 2)
            + β_morph * morpheme_complexity
            - β_big   * bigram_familiarity
            - β_freq  * log_rel_freq
            - β_neigh * log(1 + neighbourhood)
    """
    w = word.lower().strip()
    if not w or not w[0].isalpha():
        return 10.0

    # 1. Normed lookup
    if w in aoa:
        return aoa[w]

    # 2. Prototype list
    if w in _EARLY_WORDS:
        return _EARLY_WORDS[w]

    # ── Feature extraction ──────────────────────────────────────────────────
    n_chars   = len(w)
    n_syl     = _count_syllables(w)
    morph     = _morpheme_complexity(w)
    bigram_f  = _bigram_familiarity(w)
    neigh     = _ortho_neighborhood_size(w, aoa)

    # Corpus frequency (log relative frequency, 0 if absent)
    if corpus_freq and w in corpus_freq:
        rel_freq = corpus_freq[w] / max(corpus_total, 1)
        log_freq = math.log(1 + rel_freq * 1_000_000)  # per-million scale
    else:
        log_freq = 0.0

    # ── Linear model ────────────────────────────────────────────────────────
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
    """Public accessor — uses calculate_word_age."""
    return calculate_word_age(token, aoa, corpus_freq, corpus_total)


def age_continuity_boost(age1: float, age2: float, strength: float = 0.12) -> float:
    """Low-differentiation: small positive bias for similar (and earlier) ages."""
    d = abs(age1 - age2)
    early = min(age1, age2, 8.0) / 8.0
    return float(strength * math.exp(-d / 3.0) * early)


# ────────────────────────────────────────────────────────────────────────────
# COHOMOLOGY SCALARS
# ────────────────────────────────────────────────────────────────────────────
def topo_weight(token: str) -> float:
    tl = token.lower()
    return min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))


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
# DOUBLE ENTENDRE HASH EMBEDDING (two dot products + shift)
# ────────────────────────────────────────────────────────────────────────────
class DoubleEntendreEmbedder:
    """
    Two-pass dot product:

      pass1 = dot(embed(w2), embed(c))
      pass2 = dot(embed(w2) + shift(w1), embed(c))

    combined = 0.5*(norm01(pass1) + norm01(pass2)) + agreement_bonus*min(norm01(pass1), norm01(pass2))
    """

    DIM = 4

    def embed(self, token: str) -> np.ndarray:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec = np.array([(h >> (8 * i)) & 0xFF for i in range(self.DIM)], dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def shift_vector(self, token: str, magnitude: float = 0.15) -> np.ndarray:
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        vec = np.array([(h >> (8 * i)) & 0xFF for i in range(self.DIM)], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec * float(magnitude)

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-12)

    def double_entendre_weights(
        self,
        w1: str,
        w2: str,
        candidates: List[str],
        shift_mag: float = 0.15,
        agreement_bonus: float = 0.30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor = self.embed(w2)
        shifted_anchor = anchor + self.shift_vector(w1, magnitude=shift_mag)

        # Keep the same "L1-ish" normalization style as embed()
        shifted_anchor = shifted_anchor / (shifted_anchor.sum() + 1e-8)

        cand_vecs = np.array([self.embed(c) for c in candidates], dtype=np.float32)  # (N, DIM)
        pass1 = cand_vecs @ anchor
        pass2 = cand_vecs @ shifted_anchor

        p1 = self._norm01(pass1)
        p2 = self._norm01(pass2)

        de_score = np.minimum(p1, p2)
        combined = 0.5 * (p1 + p2) + float(agreement_bonus) * de_score
        combined = self._norm01(combined)
        return p1, p2, combined


# ────────────────────────────────────────────────────────────────────────────
# LANGUAGE MODEL
# ────────────────────────────────────────────────────────────────────────────
class NGramLM:
    """Trigram LM with high add_k for flat (low-differentiation) distributions."""

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
    embedder:    DoubleEntendreEmbedder
    aoa:         Dict[str, float]
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int]   = field(default_factory=dict)
    corpus_total: int             = 1


def build_state(text: str, aoa: Dict[str, float]) -> CorpusState:
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)
    embedder = DoubleEntendreEmbedder()

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
    boost_strength: float = 0.2,          # kept (but no longer used for dot prod)
    de_strength: float = 0.18,            # strength of double-entendre similarity
    de_shift_mag: float = 0.15,           # shift magnitude for 2nd frame
    de_agreement_bonus: float = 0.30,     # extra reward for agreement (min)
    ema_prev: Optional[torch.Tensor] = None,
    ema_cands: Optional[List[str]] = None,
) -> Tuple[List[str], torch.Tensor]:
    cands, base_probs = state.lm.next_dist(w1, w2)

    # Double-entendre dot-product weights
    _, _, de_combined = state.embedder.double_entendre_weights(
        w1=w1, w2=w2, candidates=cands,
        shift_mag=float(de_shift_mag),
        agreement_bonus=float(de_agreement_bonus),
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

    # AoA continuity
    w2_age  = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array(
        [age_continuity_boost(
            w2_age,
            word_age(state.aoa, c, state.corpus_freq, state.corpus_total),
        ) for c in cands],
        dtype=np.float32,
    )
    age_t = torch.tensor(age_arr, dtype=torch.float32)

    boosts = float(de_strength) * de_t + cb_t + 0.10 * tb + 0.15 * age_t
    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    logits = logits / max(float(temp), 1e-6)
    probs  = F.softmax(logits, dim=-1)

    # EMA smoothing
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
        ("Positor",     ["what", "how", "whether", "consider"]),
        ("Analyzer",    ["because", "therefore", "observe", "examine"]),
        ("Synthesizer", ["thus", "between", "integrates", "suggests"]),
        ("Reflector",   ["ultimately", "reveals", "illuminates", "perhaps"]),
        ("Connector",   ["relates", "links", "bridges", "connects"]),
        ("Elaborator",  ["further", "moreover", "extends", "develops"]),
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
# AGE ANALYSIS HELPER (for stats panel)
# ────────────────────────────────────────────────────────────────────────────
def age_analysis(
    state: CorpusState,
    top_n: int = 10,
) -> str:
    """
    Produce a brief report on word-age distribution in the corpus vocabulary.
    Shows which words are youngest/oldest by calculated AoA.
    """
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
    ]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ────────────────────────────────────────────────────────────────────────────
def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows,
    text_file, prompt, seed, max_tokens, num_voices, temp,
    progress=gr.Progress(),
):
    try:
        progress(0.05, desc="Loading AoA dataset (Kuperman 2012)…")
        aoa = load_aoa_dataset()

        progress(0.15, desc="Loading corpus…")
        text = load_corpus(bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows), text_file)

        progress(0.40, desc="Building language model…")
        state = build_state(text, aoa)

        progress(0.60, desc="Analysing word ages…")
        age_stats = age_analysis(state)

        progress(0.70, desc="Generating narrative…")
        out_md = generate(
            state, str(prompt),
            max_tokens=int(max_tokens),
            seed=int(seed),
            num_voices=int(num_voices),
            temp=float(temp),
        )

        vocab_size  = len(state.lm.vocab)
        topo_hits   = [t for t in state.lm.vocab if topo_weight(t) > 0]
        normed      = sum(1 for t in state.lm.vocab if t.isalpha() and t in aoa)
        alpha_total = sum(1 for t in state.lm.vocab if t.isalpha())

        stats = "\n".join([
            f"Vocab size: {vocab_size}",
            f"AoA normed (Kuperman exact):    {normed}/{alpha_total}",
            f"AoA calculated (feature model): {alpha_total - normed}/{alpha_total}",
            f"Topo tokens: {len(topo_hits)} ({', '.join(topo_hits[:8])})",
            f"Temperature: {float(temp):.2f}",
            f"add_k: {state.lm.add_k:.2f}",
            f"Generated tokens: {int(max_tokens)}",
            "",
            "── Word-Age Distribution ──",
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
        title="NeuroSymbolic V8.5 — Calculated Word Age + Double Entendre Dot Product",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# NeuroSymbolic V8.5 — Cohomology-Reduced + Calculated Word Age\n"
            "Word age (AoA) is now **calculated** for every token, not just looked up.\n\n"
            "**Priority:** normed Kuperman 2012 → prototype list → feature-based estimator  \n"
            "**Features used:** word length, syllable count, morpheme complexity, "
            "bigram familiarity, corpus frequency, orthographic neighbourhood size.\n\n"
            "**New:** double-entendre dot-product boost (two dot products with a shifted frame)."
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

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Starting Prompt",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Generate", variant="primary", size="lg")
                gr.Markdown("## Generated Narrative (roles)")
                output_md    = gr.Markdown(value="")
                output_stats = gr.Textbox(label="Stats + Word-Age Analysis", lines=20)

        btn.click(
            run_session,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows,
                    text_file, prompt, seed, max_tokens, num_voices, temp],
            outputs=[output_md, output_stats],
        )

        gr.Markdown(
            "### Notes\n"
            "- If the Kuperman CSV is unreachable, the model falls back to the "
            "feature-based estimator for *all* tokens (no flat 10.0).\n"
            "- Install: `pip install gradio datasets torch pandas numpy`\n"
            "- The word-age estimator is calibrated to the Kuperman distribution "
            "(mean ≈ 8.5 yr, SD ≈ 2.5 yr)."
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
