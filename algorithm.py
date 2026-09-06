#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGram Engine + Image-Modulated Generation (with USB camera capture)
===============================================================================
This replaces the previous "V18-RP-ANISO-RIPPLE-SPAGHETTI-CARDAN" engine
(hyperbolic Bolyai/Thebault token geometry + Cardan Grille Isomorphism vocab
partitions + Mirrored Instruction Distribution + the 23-strand "spaghetti"
router) with a much simpler, transparent core: an n-gram backoff language
model (`NGramModel`) plus a TF-IDF/SVD corpus-evidence layer (`CorpusSearch`).

Everything under SECTION 1-3 is the n-gram core (previously a separate
script); it is now the main generation engine end to end.

The one piece of the old engine that is preserved is the *image recognition
/ image-modulated generation* feature: a picture can be uploaded in the
Gradio UI, optionally scaled by a live Arduino sensor reading, and it biases
which tokens get chosen during generation. In the old script this lived in
`NanowireCanvas` / `NanowireStream` and operated on torch tensors of
(rho, theta, sigma) token coordinates. Since the n-gram model has no such
geometry, it's re-implemented here as `ImagePixelModulator`: each candidate
token is deterministically mapped to a column of the image, and that
column's average color feeds three lightweight "brush" trends (contrast,
chromatic phase, glow) that combine into a per-token bias, exactly mirroring
the spirit (and the three-brush structure) of the original.

NEW: USB camera capture via OpenCV with a single button click.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import gradio as gr

try:
    import serial
except Exception:  # pyserial may not be installed / no Arduino attached
    serial = None

try:
    import cv2
except Exception:
    cv2 = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — AUTONOMIC ARDUINO CARRIER  (preserved from the old GUI)
# ════════════════════════════════════════════════════════════════════════════
# Same role as before: an optional live sensor value (0..1) that scales how
# strongly the uploaded image modulates generation. Defaults to 1.0 so
# everything works fine with no Arduino attached.

LATEST_AUTONOMIC_VAL = 1.0
AUTONOMIC_SAVE_FILE = "autonomic_state.json"


def autonomic_serial_worker(port: str = "COM4", baud: int = 9600) -> None:
    global LATEST_AUTONOMIC_VAL
    if serial is None:
        return
    try:
        ser = serial.Serial(port, baud)
        print(f"Listening to Arduino on {port}...")
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.isdigit():
                    LATEST_AUTONOMIC_VAL = int(line) / 1023.0
            except Exception:
                pass
    except Exception as e:
        print(f"Serial stream error: {e}")


threading.Thread(target=autonomic_serial_worker, daemon=True).start()


def save_autonomic_ui() -> str:
    global LATEST_AUTONOMIC_VAL
    try:
        with open(AUTONOMIC_SAVE_FILE, "w") as f:
            json.dump({"autonomic_value": LATEST_AUTONOMIC_VAL}, f, indent=4)
        return f"Saved value: {LATEST_AUTONOMIC_VAL:.4f}"
    except Exception as e:
        return f"Error saving: {e}"


def load_autonomic_ui() -> Tuple[str, float]:
    global LATEST_AUTONOMIC_VAL
    if os.path.exists(AUTONOMIC_SAVE_FILE):
        try:
            with open(AUTONOMIC_SAVE_FILE, "r") as f:
                data = json.load(f)
            val = float(data.get("autonomic_value", 1.0))
            LATEST_AUTONOMIC_VAL = val
            return f"Loaded value: {val:.4f}", val
        except Exception as e:
            return f"Error loading: {e}", float(LATEST_AUTONOMIC_VAL)
    return "No saved state found.", float(LATEST_AUTONOMIC_VAL)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — N-GRAM CORE  (this is now "the main AI")
# ════════════════════════════════════════════════════════════════════════════
#
# Pipeline per user turn:
#   1. tokenize prompt, find closest corpus sentences (evidence display)
#   2. run several independent "scratch" generations from the model
#   3. combine those runs into a single candidate modifier: tokens that
#      show up consistently across runs reinforce each other, tokens that
#      only appear in a minority of runs cancel out
#   4. sample the final continuation, biased by that consensus modifier,
#      the corpus evidence modifier, AND (new) the image modulator below.

MODEL_PATH = "model.json"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 20

MIN_COUNT = 1
INFLUENCE_TAU = 0.5

CURVE_K = 8.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55

EVIDENCE_WEIGHT = 0.4

NUM_GENERATIONS = 5
CANCEL_STRENGTH = 1.0
MODIFIER_WEIGHT = 0.6
CONSENSUS_BASELINE_SPLIT = 0.5

IMAGE_WEIGHT = 0.5  # default strength of the preserved image modulation

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")
IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a) - IGNORED_TOKENS
    sb = set(b) - IGNORED_TOKENS
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


# ---------------- Evidence ----------------

class Evidence(Enum):
    TRUE = "⊤"
    FALSE = "⊥"
    UNKNOWN = "?"
    CONFLICT = "!"


NEGATORS = ("not ", "never ", "did not ", "didn't ", "no ", "without ")


@dataclass
class EvidenceRecord:
    state: Evidence = Evidence.UNKNOWN
    true_count: int = 0
    false_count: int = 0
    similarities: List[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.true_count + self.false_count

    @property
    def consistency(self) -> float:
        if self.total == 0:
            return 0.0
        return max(self.true_count, self.false_count) / self.total

    def add(self, result: Evidence, similarity: float) -> None:
        if result is Evidence.UNKNOWN:
            return
        self.similarities.append(similarity)
        if result is Evidence.TRUE:
            self.true_count += 1
        elif result is Evidence.FALSE:
            self.false_count += 1
        if self.true_count and self.false_count:
            self.state = Evidence.CONFLICT
        elif self.true_count:
            self.state = Evidence.TRUE
        elif self.false_count:
            self.state = Evidence.FALSE

    @property
    def verdict(self) -> str:
        return {
            Evidence.TRUE: "SUPPORTED",
            Evidence.FALSE: "REFUTED",
            Evidence.UNKNOWN: "UNRESOLVED",
            Evidence.CONFLICT: "CONFLICTED",
        }[self.state]


# ---------------- Geometric text space (TF-IDF -> SVD) ----------------

class GeometricDataset:
    def __init__(self, texts: List[str], dimensions: int = 16) -> None:
        self.texts = texts
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)

        max_dim = min(dimensions, max(1, X.shape[0] - 1), max(1, X.shape[1] - 1))

        if X.shape[1] <= 1:
            self.X = X.toarray().astype(float)
            self.svd = None
        else:
            self.svd = TruncatedSVD(n_components=max_dim, random_state=42)
            self.X = self.svd.fit_transform(X)

        self.X = normalize(self.X)

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        if self.svd is not None:
            X = self.svd.transform(X)
        return normalize(X)

    def vector(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


@dataclass
class CorpusReference:
    sentence: str
    tokens: List[str]
    vector: np.ndarray
    frequency: int = 1
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)


@dataclass
class Candidate:
    sentence: str
    symbolic_overlap: float
    vector_similarity: float
    frequency: int
    score: float
    state: Evidence
    verdict: str
    true_count: int
    false_count: int
    consistency: float
    samples: int
    rank: int = 0


class CorpusSearch:
    """
    Finds corpus sentences closest to a prompt, and classifies each one as
    supported, refuted, unresolved, or conflicted evidence given everything
    said so far this session.
    """

    def __init__(self, lexical_weight: float = LEXICAL_WEIGHT, vector_weight: float = VECTOR_WEIGHT) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []
        self.dataset: Optional[GeometricDataset] = None

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(s.lower() for s in sentences)

        kept = [(s, tokenize(s)) for s in sentences]
        kept = [(s, t) for s, t in kept if t]
        if not kept:
            self.references = []
            self.dataset = None
            return

        self.dataset = GeometricDataset([s for s, _ in kept], dimensions=16)

        self.references = [
            CorpusReference(
                sentence=sentence,
                tokens=tokens,
                vector=self.dataset.X[i],
                frequency=counts[sentence.lower()],
            )
            for i, (sentence, tokens) in enumerate(kept)
        ]

    def analyze(self, prompt: str, limit: int = 5) -> Tuple[List[Candidate], Dict[str, float]]:
        if not self.references or self.dataset is None:
            return [], {}

        prompt_tokens = tokenize(prompt)
        prompt_vector = self.dataset.vector(prompt)

        similarities = [float(np.dot(prompt_vector, ref.vector)) for ref in self.references]

        ceiling = max(similarities) if similarities else 0.0
        support_threshold = 0.55 * ceiling
        contradiction_threshold = 0.45 * ceiling

        text = " " + prompt.lower() + " "
        explicitly_negative = any(neg in text for neg in NEGATORS)

        candidates = []
        raw_modifier: Dict[str, float] = defaultdict(float)

        for ref, vector_sim in zip(self.references, similarities):
            if ceiling <= 1e-9:
                result = Evidence.UNKNOWN
            elif explicitly_negative and vector_sim >= contradiction_threshold:
                result = Evidence.FALSE
            elif vector_sim >= support_threshold:
                result = Evidence.TRUE
            else:
                result = Evidence.UNKNOWN

            ref.evidence.add(result, vector_sim)

            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            score = self.lexical_weight * symbolic + self.vector_weight * vector_sim

            candidates.append(
                Candidate(
                    sentence=ref.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_sim,
                    frequency=ref.frequency,
                    score=score,
                    state=ref.evidence.state,
                    verdict=ref.evidence.verdict,
                    true_count=ref.evidence.true_count,
                    false_count=ref.evidence.false_count,
                    consistency=ref.evidence.consistency,
                    samples=ref.evidence.total,
                )
            )

            state = ref.evidence.state
            if state is Evidence.TRUE:
                sign = 1.0
            elif state is Evidence.FALSE:
                sign = -1.0
            elif state is Evidence.CONFLICT:
                total = ref.evidence.total
                sign = (ref.evidence.true_count - ref.evidence.false_count) / total if total else 0.0
            else:
                sign = 0.0

            if sign != 0.0:
                weight = sign * score
                for token in ref.tokens:
                    if token not in IGNORED_TOKENS:
                        raw_modifier[token] += weight

        evidence_modifier: Dict[str, float] = {}
        if raw_modifier:
            largest = max(abs(v) for v in raw_modifier.values())
            if largest > 0:
                evidence_modifier = {t: v / largest for t, v in raw_modifier.items()}

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates, evidence_modifier


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IMAGE MODULATION  (preserved "image recognition" feature)
# ════════════════════════════════════════════════════════════════════════════
#
# Old version: NanowireCanvas/NanowireStream operated on torch tensors of
# per-token (rho, theta, sigma) hyperbolic coordinates and looked up image
# columns by normalizing those coordinates into pixel positions, then ran
# three "brush" trends (contrast-sharpen, chromatic-phase, bloom-glow) over
# the resulting intensities.
#
# The n-gram model has no such geometry, so each token is instead mapped to
# a stable pseudo-random column position via a hash of the token string
# itself. Everything else — the three-brush structure, and scaling by the
# live Arduino "autonomic" value — is preserved.

class ImagePixelModulator:
    """Biases token selection using an uploaded image, optionally scaled by
    a live sensor reading. This is the preserved image-modulation feature."""

    def __init__(self) -> None:
        self.image: Optional[np.ndarray] = None  # (H, W, 3) float32 in [0, 1]

    def update_image(self, numpy_img, autonomic_value: float = 1.0) -> None:
        if numpy_img is None:
            self.image = None
            return
        if isinstance(numpy_img, dict):
            for key in ("composite", "image", "background"):
                if numpy_img.get(key) is not None:
                    numpy_img = numpy_img[key]
                    break
            else:
                self.image = None
                return
        if not isinstance(numpy_img, np.ndarray) or numpy_img.ndim != 3 or numpy_img.shape[2] < 3:
            self.image = None
            return
        img = numpy_img[:, :, :3].astype(np.float32) / 255.0
        img = np.clip(img * max(0.0, float(autonomic_value)), 0.0, 1.0)
        self.image = img

    @staticmethod
    def _token_position(token: str) -> float:
        h = 0
        for ch in token:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return (h % 10007) / 10007.0

    def token_bias(self, token: str) -> float:
        """Returns a bias in roughly [-0.5, 0.5], 0.0 if no image is set."""
        if self.image is None:
            return 0.0
        H, W, _ = self.image.shape
        x = int(self._token_position(token) * (W - 1))
        column = self.image[:, x, :]
        r_mean, g_mean, b_mean = (float(v) for v in column.mean(axis=0))
        contrast = 1.0 / (1.0 + math.exp(-8.0 * (r_mean - 0.5)))   # "ContrastSharpen" brush
        phase = 1.0 + 0.5 * math.sin(g_mean * 3.0 * math.pi)        # "ChromaticPhase" brush
        glow = math.tanh(3.0 * b_mean)                               # "BloomGlow" brush
        return (contrast * phase * glow) - 0.5


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — N-GRAM LANGUAGE MODEL  (extended with an image_modulator hook)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    influence_tau: float = INFLUENCE_TAU
    curve_k: float = CURVE_K
    curve_midpoint: float = CURVE_MIDPOINT

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    # ---------------- ingestion ----------------

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue
            sequence = ["<bos>", "<bos>"] + words + [self.eos_token]
            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    # ---------------- training ----------------

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in self.vocabulary:
            source_vec = self.lexical_vectors.get(source, {})
            scores = {}
            for target in self.vocabulary:
                if source == target:
                    continue
                sim = cosine_similarity(source_vec, self.lexical_vectors.get(target, {}))
                if sim >= self.influence_tau:
                    scores[target] = sim
            self.influence_vectors[source] = scores

        self.finalized = True

    # ---------------- generation ----------------

    def _backoff_distribution(self, prev: str, prev_prev: Optional[str]) -> Dict[str, float]:
        if prev_prev is not None:
            counts = self.trigram.get(f"{prev_prev}\t{prev}")
            if counts:
                return self._normalize(counts)
        counts = self.bigram.get(prev)
        if counts:
            return self._normalize(counts)
        return self._normalize(self.unigram)

    @staticmethod
    def _normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def _curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))
        z = self.curve_k * (p_eos - self.curve_midpoint)
        return 1.0 / (1.0 + math.exp(z))

    def _resolve_context(self, prompt: str) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)
        if not tokens:
            return "<bos>", None
        prev = tokens[-1]
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        return prev, prev_prev

    def _score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
        image_modulator: Optional[ImagePixelModulator] = None,
        image_weight: float = 0.0,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        prev, prev_prev = self._resolve_context(prompt)
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}

        candidates = [t for t in sorted(base, key=base.get, reverse=True) if t not in IGNORED_TOKENS][:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        curve = self._curve_weight(base.get(self.eos_token, 0.0))

        scores = {}
        for token in candidates:
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            score = (
                safe_log(base[token])
                + curve * 0.35 * similarity
                + curve * 0.65 * influence
            )
            if candidate_modifier and modifier_weight:
                consensus_weight = candidate_modifier.get(token, 0.0)
                baseline_prob = base.get(token, 0.0)
                blended_bias = (
                    CONSENSUS_BASELINE_SPLIT * consensus_weight
                    + (1.0 - CONSENSUS_BASELINE_SPLIT) * baseline_prob
                )
                score += modifier_weight * blended_bias
            if evidence_modifier and evidence_weight:
                score += evidence_weight * evidence_modifier.get(token, 0.0)
            if image_modulator is not None and image_weight:
                score += image_weight * image_modulator.token_bias(token)
            scores[token] = score
        return scores

    def _probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
        image_modulator: Optional[ImagePixelModulator] = None,
        image_weight: float = 0.0,
    ) -> Dict[str, float]:
        scores = self._score_next_token(
            prompt, candidate_limit, candidate_modifier, modifier_weight,
            evidence_modifier, evidence_weight, image_modulator, image_weight,
        )
        if not scores:
            return {}
        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
        image_modulator: Optional[ImagePixelModulator] = None,
        image_weight: float = 0.0,
    ) -> str:
        probs = self._probabilities(
            prompt, temperature, max(top_k, 1), candidate_modifier, modifier_weight,
            evidence_modifier, evidence_weight, image_modulator, image_weight,
        )
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
        image_modulator: Optional[ImagePixelModulator] = None,
        image_weight: float = 0.0,
    ) -> str:
        generated = tokenize(prompt)
        for _ in range(max_new_tokens):
            token = self.sample_next(
                " ".join(generated), temperature, top_k, candidate_modifier, modifier_weight,
                evidence_modifier, evidence_weight, image_modulator, image_weight,
            )
            generated.append(token)
        return self.detokenize(generated)

    def generate_with_trace(
        self, prompt: str, max_new_tokens: int, temperature: float, top_k: int
    ) -> Tuple[str, List[str]]:
        generated = tokenize(prompt)
        start = len(generated)
        for _ in range(max_new_tokens):
            token = self.sample_next(" ".join(generated), temperature, top_k)
            generated.append(token)
        return self.detokenize(generated), generated[start:]

    def multi_generate(
        self,
        prompt: str,
        num_generations: int = NUM_GENERATIONS,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> List[List[str]]:
        """Run several independent scratch generations from the same prompt.
        Left unbiased by consensus/evidence/image so the runs stay independent."""
        runs: List[List[str]] = []
        for _ in range(num_generations):
            _, new_tokens = self.generate_with_trace(prompt, max_new_tokens, temperature, top_k)
            runs.append(new_tokens)
        return runs

    @staticmethod
    def build_candidate_modifier(
        runs: List[List[str]], cancel_strength: float = CANCEL_STRENGTH
    ) -> Dict[str, float]:
        """
        modifier[token] = mean_count(token) - cancel_strength * std(token)

        Tokens that appear consistently across runs (low variance) survive;
        tokens that only show up in a minority of runs cancel out to <= 0
        and are dropped. Survivors are normalized to 0..1.
        """
        if not runs:
            return {}

        counters = [Counter(run) for run in runs]
        vocab = set()
        for c in counters:
            vocab.update(c.keys())

        n = len(counters)
        modifier: Dict[str, float] = {}
        for token in vocab:
            values = [c.get(token, 0) for c in counters]
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance)
            net = mean - cancel_strength * std
            if net > 0:
                modifier[token] = net

        if not modifier:
            return {}
        max_val = max(modifier.values())
        return {t: v / max_val for t, v in modifier.items()}

    def generate_consensus(
        self,
        prompt: str,
        num_generations: int = NUM_GENERATIONS,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        cancel_strength: float = CANCEL_STRENGTH,
        modifier_weight: float = MODIFIER_WEIGHT,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
        image_modulator: Optional[ImagePixelModulator] = None,
        image_weight: float = 0.0,
    ) -> Tuple[str, List[List[str]], Dict[str, float]]:
        """
        Full pipeline: run several scratch generations, cancel them out into
        a consensus modifier, then do one more final generation biased by
        that consensus modifier, the corpus evidence_modifier, and (new)
        the image_modulator.
        """
        runs = self.multi_generate(prompt, num_generations, max_new_tokens, temperature, top_k)
        modifier = self.build_candidate_modifier(runs, cancel_strength)
        final_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            candidate_modifier=modifier,
            modifier_weight=modifier_weight,
            evidence_modifier=evidence_modifier,
            evidence_weight=evidence_weight,
            image_modulator=image_modulator,
            image_weight=image_weight,
        )
        return final_text, runs, modifier

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        return text

    # ---------------- persistence ----------------

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "influence_tau": self.influence_tau,
            "curve_k": self.curve_k,
            "curve_midpoint": self.curve_midpoint,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data["eos_token"],
            unk_token=data["unk_token"],
            min_count=data["min_count"],
            influence_tau=data["influence_tau"],
            curve_k=data["curve_k"],
            curve_midpoint=data["curve_midpoint"],
        )
        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data["bigram"].items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data["trigram"].items()})
        model.lexical_vectors = data["lexical_vectors"]
        model.influence_vectors = data["influence_vectors"]
        model.vocabulary = data["vocabulary"]
        model.finalized = data["finalized"]
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENGINE WRAPPER  (plays the role the old V18RPEngine played)
# ════════════════════════════════════════════════════════════════════════════

class NGramEngine:
    def __init__(self) -> None:
        self.model: Optional[NGramModel] = None
        self.search: Optional[CorpusSearch] = None
        self.modulator = ImagePixelModulator()
        self._initialised = False

    def train(self, corpus_text: str, min_count: int = MIN_COUNT, influence_tau: float = INFLUENCE_TAU) -> None:
        self.model = NGramModel(min_count=min_count, influence_tau=influence_tau)
        self.model.ingest_text(corpus_text)
        self.model.finalize()
        self.search = CorpusSearch(LEXICAL_WEIGHT, VECTOR_WEIGHT)
        self.search.build_index(corpus_text)
        self._initialised = True

    def generate(
        self,
        prompt: str,
        num_generations: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        cancel_strength: float,
        modifier_weight: float,
        evidence_weight: float,
        image_weight: float,
        autonomic_value: float,
        art_image,
    ) -> Tuple[str, List[Candidate], List[List[str]], Dict[str, float]]:
        assert self._initialised and self.model is not None and self.search is not None
        self.modulator.update_image(art_image, autonomic_value)
        candidates, evidence_modifier = self.search.analyze(prompt, limit=CANDIDATE_LIMIT)
        final_text, runs, modifier = self.model.generate_consensus(
            prompt,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            cancel_strength=cancel_strength,
            modifier_weight=modifier_weight,
            evidence_modifier=evidence_modifier,
            evidence_weight=evidence_weight,
            image_modulator=self.modulator,
            image_weight=image_weight,
        )
        return final_text, candidates, runs, modifier


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _format_candidates(candidates: List[Candidate]) -> str:
    if not candidates:
        return "No candidates found."
    lines = []
    for c in candidates:
        lines.append(f"[{c.rank}] score={c.score:.3f}  state={c.state.value} ({c.verdict})  {c.sentence}")
        lines.append(
            f"      sim={c.vector_similarity:.3f}  symbolic={c.symbolic_overlap:.3f}  "
            f"true={c.true_count}  false={c.false_count}  "
            f"consistency={c.consistency:.2f}  samples={c.samples}"
        )
    return "\n".join(lines)


def _format_ensemble(runs: List[List[str]], modifier: Dict[str, float]) -> str:
    lines = [f"{len(runs)} scratch generations (cancelled against each other):"]
    for i, run in enumerate(runs, start=1):
        preview = NGramModel.detokenize(run[:20])
        lines.append(f"[run {i}] {preview}{' ...' if len(run) > 20 else ''}")
    top_survivors = sorted(modifier.items(), key=lambda kv: kv[1], reverse=True)[:15]
    lines.append("")
    lines.append("Surviving tokens after cancel-out (top 15):")
    if not top_survivors:
        lines.append("  (none survived — runs disagreed on everything)")
    for token, weight in top_survivors:
        lines.append(f"  {token!r:<15} weight={weight:.3f}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GRADIO GUI
# ════════════════════════════════════════════════════════════════════════════

_engine: Optional[NGramEngine] = None


def _gui_init(file_in, min_count, influence_tau):
    global _engine
    try:
        if file_in is None:
            return "❌ No file uploaded."
        text = Path(file_in.name).read_text(encoding="utf-8", errors="replace")
        _engine = NGramEngine()
        _engine.train(text, min_count=int(min_count), influence_tau=float(influence_tau))
        return (
            "✅ N-gram engine trained.\n"
            f"Vocabulary: {len(_engine.model.vocabulary):,}\n"
            f"Unigrams: {len(_engine.model.unigram):,}\n"
            f"Bigram contexts: {len(_engine.model.bigram):,}\n"
            f"Trigram contexts: {len(_engine.model.trigram):,}\n"
            f"Corpus sentences indexed: {len(_engine.search.references):,}"
        )
    except Exception:
        import traceback
        return f"❌ Error:\n{traceback.format_exc()}"


def _gui_generate(prompt, n_gens, max_tokens, temp, top_k, cancel_strength,
                   mod_weight, evid_weight, img_weight, art_image):
    global _engine, LATEST_AUTONOMIC_VAL
    if _engine is None or not _engine._initialised:
        return "❌ Initialise engine first.", "", ""
    try:
        final_text, candidates, runs, modifier = _engine.generate(
            prompt,
            int(n_gens), int(max_tokens), float(temp), int(top_k),
            float(cancel_strength), float(mod_weight), float(evid_weight), float(img_weight),
            LATEST_AUTONOMIC_VAL, art_image,
        )
        return final_text, _format_candidates(candidates), _format_ensemble(runs, modifier)
    except Exception:
        import traceback
        return f"❌ Error:\n{traceback.format_exc()}", "", ""


def capture_usb_camera(camera_index=0, width=1280, height=720):
    """
    Capture one frame from a USB camera.

    Returns:
        RGB NumPy image suitable for Gradio ImageEditor/Image.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Install it with: pip install opencv-python"
        )

    camera_index = int(camera_index)

    # CAP_DSHOW is useful on Windows; fall back to the default backend if needed.
    if os.name == "nt":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try camera index 1 or 2."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    # Discard a few initial frames so exposure and focus can settle.
    frame = None
    for _ in range(5):
        ok, frame = camera.read()

    camera.release()

    if not ok or frame is None:
        raise RuntimeError("The camera opened, but no image frame was received.")

    # OpenCV returns BGR; Gradio expects RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="NGram Engine + Image-Modulated Generation") as demo:
        gr.Markdown("# NGram Engine")
        gr.Markdown(
            "Core generation now runs on a trigram/bigram backoff model "
            "(`NGramModel`) plus a TF-IDF/SVD corpus-evidence layer "
            "(`CorpusSearch`), instead of the previous hyperbolic-geometry / "
            "Cardan-grille / spaghetti-router engine.\n\n"
            "**Image modulation is preserved**: an uploaded picture — "
            "optionally scaled by a live Arduino sensor reading — still "
            "biases which tokens get chosen during generation, via "
            "`ImagePixelModulator`.\n\n"
            "**USB camera capture added**: click 'Capture USB Camera' to grab a frame."
        )

        with gr.Tab("Init / Train"):
            file_in = gr.File(label="Upload corpus .txt")
            with gr.Row():
                min_count = gr.Slider(1, 10, value=MIN_COUNT, step=1, label="Min token count")
                influence_tau = gr.Slider(0.0, 1.0, value=INFLUENCE_TAU, step=0.05,
                                           label="Influence similarity threshold")
            init_btn = gr.Button("Train Engine")
            init_out = gr.Textbox(lines=10, label="Init output")
            init_btn.click(_gui_init, inputs=[file_in, min_count, influence_tau], outputs=init_out)

        with gr.Tab("Generate"):
            prompt_txt = gr.Textbox(label="Prompt", lines=2)
            with gr.Row():
                n_gens = gr.Slider(1, 12, value=NUM_GENERATIONS, step=1, label="Scratch generations")
                max_tokens = gr.Slider(10, 500, value=MAX_NEW_TOKENS, step=10, label="Max new tokens")
                temp = gr.Slider(0.1, 2.0, value=TEMPERATURE, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 100, value=TOP_K, step=1, label="Top-k")
            with gr.Row():
                cancel_strength = gr.Slider(0.0, 3.0, value=CANCEL_STRENGTH, step=0.1, label="Cancel strength")
                mod_weight = gr.Slider(0.0, 2.0, value=MODIFIER_WEIGHT, step=0.05, label="Consensus modifier weight")
                evid_weight = gr.Slider(0.0, 2.0, value=EVIDENCE_WEIGHT, step=0.05, label="Evidence weight")
                img_weight = gr.Slider(0.0, 2.0, value=IMAGE_WEIGHT, step=0.05, label="Image modulation weight")

            gr.Markdown("### Image source")

            with gr.Row():
                try:
                    art_img = gr.ImageEditor(
                        type="numpy",
                        label="Modulation image",
                        image_mode="RGB",
                    )
                except AttributeError:
                    art_img = gr.Image(
                        type="numpy",
                        label="Modulation image",
                    )

                with gr.Column():
                    camera_index = gr.Number(
                        value=0,
                        precision=0,
                        label="USB camera index",
                    )

                    capture_camera_btn = gr.Button(
                        "Capture USB Camera",
                        variant="secondary",
                    )

                    capture_status = gr.Textbox(
                        label="Camera status",
                        interactive=False,
                    )

                    capture_camera_btn.click(
                        fn=capture_usb_camera,
                        inputs=[camera_index],
                        outputs=[art_img],
                    ).then(
                        fn=lambda: "USB camera frame captured.",
                        inputs=[],
                        outputs=[capture_status],
                    )

            gen_btn = gr.Button("Generate")
            gen_out = gr.Textbox(lines=10, label="Generated text")
            cand_out = gr.Textbox(lines=14, label="Corpus evidence / candidates")
            ens_out = gr.Textbox(lines=14, label="Ensemble (scratch runs + surviving tokens)")

            gen_btn.click(
                _gui_generate,
                inputs=[prompt_txt, n_gens, max_tokens, temp, top_k, cancel_strength,
                        mod_weight, evid_weight, img_weight, art_img],
                outputs=[gen_out, cand_out, ens_out],
            )
    return demo


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_gradio_app().launch(share=False)
