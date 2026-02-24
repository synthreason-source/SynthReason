#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V8.6+ — 100 Sentences, One Form Per Sentence

Architecture:
- Extract N words from first sentence (e.g., "consider", "nature", "understanding")
- Generate 100 unique syntactic forms for these words
- Generate 100 sentences: sentence[i] uses form[i] as its primary feature
- Each sentence yields exactly one form, spreading its activation through that sentence
- Forms accumulate value across different sentence contexts

Key: Each sentence is a distinct syntactic-semantic environment where one form dominates.
"""

from __future__ import annotations
import re
import math
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import gradio as gr
import torch
import torch.nn.functional as F
from datasets import load_dataset

# ──────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(
    "a an and are as at be by for from has have he her him his i in is it its me my of on or our "
    "she so that the their them they this to was we were what when where which who will with you your"
    .split()
)

COGNITIVE_TOKENS = {"[PROBLEM]", "[SOLUTION]"}

TOPO_KEYWORDS = {
    "homology", "cohomology", "persistent", "filtration", "barcode", "betti",
    "euler", "simplicial", "homotopy", "manifold", "morse", "sheaf"
}

_VOWELS = set("aeiouy")
_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "ou", "ed", "nd",
    "to", "or", "ea", "ti", "es", "st", "ar", "nt", "is", "al", "it", "as",
    "ha", "et", "se", "ng", "le", "of",
}

_LATINATE_PREFIXES = {
    "pre", "post", "anti", "auto", "bio", "geo", "hyper", "hypo", "inter",
    "intra", "micro", "macro", "meta", "mono", "multi", "neo", "non", "over",
    "poly", "pseudo", "semi", "sub", "super", "trans", "ultra", "uni", "dis",
    "mis", "un", "re", "de",
}

_LATINATE_SUFFIXES = {
    "tion", "sion", "ment", "ness", "ity", "ism", "ist", "ize", "ise", "ful",
    "less", "ous", "ious", "eous", "ance", "ence", "able", "ible", "ive",
    "ative", "ology", "ography", "ician", "ation", "ization", "isation",
}

_EARLY_WORDS: Dict[str, float] = {
    "cat": 2.5, "dog": 2.5, "mom": 2.2, "dad": 2.2, "baby": 2.8, "ball": 2.6,
    "cup": 2.7, "eye": 2.4, "ear": 2.5, "nose": 2.6, "hat": 2.8, "shoe": 2.9,
    "bed": 2.7, "hot": 3.0, "cold": 3.1, "big": 3.0, "small": 3.2, "run": 3.1,
    "eat": 2.9, "go": 2.5, "yes": 2.4, "no": 2.3, "hi": 2.2, "bye": 2.3,
    "more": 2.8, "up": 2.6, "down": 2.8, "in": 2.5, "out": 2.7, "on": 2.6,
    "off": 2.8, "want": 2.7, "help": 3.0, "play": 2.9, "walk": 3.0, "look": 2.8,
    "see": 2.5, "hear": 2.8, "think": 3.5, "know": 3.4, "hand": 2.9, "foot": 2.9,
    "head": 2.7, "face": 2.8, "name": 3.2, "home": 3.0, "door": 3.1, "car": 2.8,
    "tree": 3.0, "book": 3.2,
}

# ──────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT TOPOLOGY PARAMETERS
# ──────────────────────────────────────────────────────────────────────────

DIM_MIN = 2
DIM_MAX = 12
LENGTH_CEIL = 14
SHIFT_MAG_MIN = 0.05
SHIFT_MAG_MAX = 0.35
AGREEMENT_BONUS_MIN = 0.10
AGREEMENT_BONUS_MAX = 0.60

# ──────────────────────────────────────────────────────────────────────────
# SYNTACTIC FORM: 100 Forms for One-Per-Sentence System
# ──────────────────────────────────────────────────────────────────────────

class SyntacticForm:
    """A single syntactic/morphological form variant."""
    
    # 100 unique forms, one per sentence
    FORMS = [
        # Base & Root (0-4)
        "base", "root", "stem", "lemma", "canonical",
        
        # Tense Forms (5-14)
        "present_tense", "past_tense", "future_tense", "present_progressive",
        "past_progressive", "future_progressive", "present_perfect", "past_perfect",
        "future_perfect", "simple_present",
        
        # Person & Number (15-24)
        "singular", "plural", "first_person_singular", "first_person_plural",
        "second_person_singular", "second_person_plural", "third_person_singular",
        "third_person_plural", "person_neutral", "number_neutral",
        
        # Case & Role (25-34)
        "nominative", "accusative", "genitive", "dative", "locative",
        "ablative", "allative", "inessive", "elative", "illative",
        
        # Part of Speech (35-44)
        "noun_form", "verb_form", "adjective_form", "adverb_form", "preposition_form",
        "conjunction_form", "article_form", "pronoun_form", "determiner_form", "numeral_form",
        
        # Voice (45-54)
        "active_voice", "passive_voice", "middle_voice", "reflexive_voice", "reciprocal_voice",
        "causative_voice", "inchoative_voice", "iterative_voice", "habitual_voice", "frequentative_voice",
        
        # Mood (55-64)
        "indicative_mood", "subjunctive_mood", "conditional_mood", "imperative_mood", "optative_mood",
        "necessitative_mood", "potential_mood", "desiderative_mood", "dubitative_mood", "permissive_mood",
        
        # Aspect (65-74)
        "perfective_aspect", "imperfective_aspect", "habitual_aspect", "iterative_aspect",
        "inceptive_aspect", "terminative_aspect", "continuative_aspect", "stative_aspect",
        "dynamic_aspect", "aorist_aspect",
        
        # Degree (75-84)
        "positive_degree", "comparative_degree", "superlative_degree", "diminutive_form",
        "augmentative_form", "pejorative_form", "ameliorative_form", "intensive_form",
        "attenuative_form", "disparagingly_form",
        
        # Derivation (85-94)
        "agentive_noun", "instrumental_noun", "locative_noun", "abstract_noun",
        "action_noun", "quality_noun", "state_noun", "relational_adjective",
        "qualitative_adjective", "derivational_adjective",
        
        # Transitivity & Valence (95-99)
        "transitive_form", "intransitive_form", "ditransitive_form", "bitransitive_form", "ambitransitive_form",
    ]
    
    def __init__(self, word: str, form_name: str, sentence_index: int):
        self.word = word.lower()
        self.form_name = form_name if form_name in self.FORMS else "base"
        self.sentence_index = sentence_index
        
        # Activation tracking per sentence
        self.activation_per_sentence: Dict[int, float] = {}
        self.total_activation: float = 0.0
        self.spreading_context: List[str] = []  # Words influenced in this form's sentence
        self.value_accumulated: float = 0.0
    
    def __repr__(self) -> str:
        return f"{self.word}[{self.form_name}@sent{self.sentence_index}]"
    
    def to_string(self) -> str:
        return f"{self.word}_{self.form_name}_{self.sentence_index}"
    
    def activate_in_sentence(self, sentence_index: int, strength: float = 1.0):
        """Record activation in a specific sentence."""
        self.activation_per_sentence[sentence_index] = \
            self.activation_per_sentence.get(sentence_index, 0.0) + strength
        self.total_activation += strength
    
    def spread_to_word(self, word: str, strength: float = 0.5):
        """Record spreading influence to another word."""
        if word not in self.spreading_context:
            self.spreading_context.append(word)
        self.value_accumulated += strength
    
    def get_total_value(self) -> float:
        """Total value = base activations + accumulated spread."""
        return self.total_activation + self.value_accumulated


@dataclass
class SentenceFormPlan:
    """Plan for 100 sentences, one form per sentence."""
    
    extracted_words: List[str] = field(default_factory=list)  # Words from first sentence
    forms_list: List[SyntacticForm] = field(default_factory=list)  # 100 forms
    form_by_sentence: Dict[int, SyntacticForm] = field(default_factory=dict)  # sent_idx -> form
    sentence_outputs: Dict[int, str] = field(default_factory=dict)  # sent_idx -> generated text
    form_report: str = ""
    
    def build_forms(self, words: List[str]):
        """Build 100 forms from extracted words, cycling through if needed."""
        self.extracted_words = words
        form_index = 0
        word_index = 0
        
        # Create 100 forms by cycling through words and forms
        for sent_idx in range(100):
            # Cycle through extracted words
            if word_index >= len(words):
                word_index = 0
            word = words[word_index]
            
            # Cycle through available forms
            if form_index >= len(SyntacticForm.FORMS):
                form_index = 0
            form_name = SyntacticForm.FORMS[form_index]
            
            # Create the form
            form = SyntacticForm(word, form_name, sent_idx)
            self.forms_list.append(form)
            self.form_by_sentence[sent_idx] = form
            
            # Advance pointers
            word_index += 1
            form_index += 1
    
    def record_sentence_generation(self, sentence_index: int, text: str, 
                                   form_activation: float = 1.0,
                                   influenced_words: Optional[List[str]] = None):
        """Record generated sentence and form activation."""
        self.sentence_outputs[sentence_index] = text
        
        form = self.form_by_sentence.get(sentence_index)
        if form:
            form.activate_in_sentence(sentence_index, form_activation)
            if influenced_words:
                for w in influenced_words:
                    form.spread_to_word(w, 0.5)
    
    def generate_report(self) -> str:
        """Generate detailed form-by-form report."""
        lines = [
            f"{'='*70}",
            f"  100-Sentence Syntactic Form Plan — One Form Per Sentence",
            f"{'='*70}",
            f"Extracted words: {', '.join(self.extracted_words)}",
            f"Total forms generated: {len(self.forms_list)}",
            f"Sentences generated: {len(self.sentence_outputs)}",
            f"",
        ]
        
        # Sort forms by total value
        sorted_forms = sorted(self.forms_list, key=lambda f: f.get_total_value(), reverse=True)
        
        total_value = sum(f.get_total_value() for f in sorted_forms)
        
        lines.append(f"Total cumulative activation: {total_value:.4f}")
        lines.append("")
        lines.append("Form Rankings (Top 30):")
        lines.append(f"{'Rank':<5} {'Sentence':<8} {'Word':<15} {'Form':<25} {'Total Value':<12} {'% of Total':<10} {'Influenced':<10}")
        lines.append(f"{'-'*90}")
        
        for rank, form in enumerate(sorted_forms[:30], 1):
            pct = 100 * form.get_total_value() / max(total_value, 1e-8)
            num_influenced = len(form.spreading_context)
            lines.append(
                f"{rank:<5} {form.sentence_index:<8} {form.word:<15} {form.form_name:<25} "
                f"{form.get_total_value():<12.4f} {pct:<10.2f} {num_influenced:<10}"
            )
        
        lines.append("")
        lines.append("Form-to-Word Influence Map (Top 10 Forms):")
        lines.append("")
        
        for rank, form in enumerate(sorted_forms[:10], 1):
            if form.spreading_context:
                influenced_str = ", ".join(form.spreading_context[:8])
                if len(form.spreading_context) > 8:
                    influenced_str += f", ... (+{len(form.spreading_context)-8} more)"
                lines.append(
                    f"{rank:2d}. {form.word}[{form.form_name}@sent{form.sentence_index}]"
                    f"\n    → Influenced: {influenced_str}"
                )
        
        lines.append("")
        lines.append("Sentence-by-Sentence Form Assignments:")
        lines.append("")
        
        for sent_idx in range(min(30, len(self.sentence_outputs))):  # Show first 30
            form = self.form_by_sentence.get(sent_idx)
            output = self.sentence_outputs.get(sent_idx, "(not generated)")
            if form:
                preview = output[:60] + "..." if len(output) > 60 else output
                lines.append(
                    f"Sent[{sent_idx:2d}] Form: {form.word}[{form.form_name:<25s}] "
                    f"Value: {form.get_total_value():.3f}  Preview: {preview}"
                )
        
        if len(self.sentence_outputs) > 30:
            lines.append(f"... ({len(self.sentence_outputs) - 30} more sentences)")
        
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY
# ──────────────────────────────────────────────────────────────────────────

def semantic_similarity(word1: str, word2: str) -> float:
    """Compute similarity between two words."""
    w1, w2 = word1.lower(), word2.lower()
    
    if w1 == w2:
        return 1.0
    
    lev_dist = edit_distance(w1, w2)
    max_len = max(len(w1), len(w2))
    lev_sim = 1.0 - (lev_dist / max(max_len, 1))
    
    len_dist = abs(len(w1) - len(w2))
    len_sim = 1.0 - (len_dist / max_len)
    
    bigrams1 = set(w1[i:i+2] for i in range(len(w1)-1))
    bigrams2 = set(w2[i:i+2] for i in range(len(w2)-1))
    if bigrams1 and bigrams2:
        bigram_sim = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
    else:
        bigram_sim = 0.0
    
    combined = 0.4 * lev_sim + 0.3 * len_sim + 0.3 * bigram_sim
    return float(combined)


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


# ──────────────────────────────────────────────────────────────────────────
# LENGTH FACTOR FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────

def length_alpha(word: str, ceil: int = LENGTH_CEIL) -> float:
    n = len(word.strip())
    mid = ceil / 2.0
    return float(1.0 / (1.0 + math.exp(-0.55 * (n - mid))))


def length_dim(word: str) -> int:
    α = length_alpha(word)
    raw = DIM_MIN + α * (DIM_MAX - DIM_MIN)
    return max(DIM_MIN, int(round(raw / 2) * 2))


def length_shift_mag(word: str) -> float:
    α = length_alpha(word)
    return SHIFT_MAG_MIN + α * (SHIFT_MAG_MAX - SHIFT_MAG_MIN)


def length_agreement_bonus(word: str) -> float:
    α = length_alpha(word)
    return AGREEMENT_BONUS_MIN + α * (AGREEMENT_BONUS_MAX - AGREEMENT_BONUS_MIN)


def length_topo_kernel(word: str) -> float:
    α = length_alpha(word)
    return float(0.05 + 0.95 * (α ** 1.5))


# ──────────────────────────────────────────────────────────────────────────
# AoA DATASET
# ──────────────────────────────────────────────────────────────────────────

AOA_DATASET_URL = (
    "https://norare.clld.org/contributions/Kuperman-2012-AoA/English-AoA-30K.csv"
)
AOA_COL_WORD = "Word"
AOA_COL_AOA = "AoA"


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


# ──────────────────────────────────────────────────────────────────────────
# WORD-AGE CALCULATOR
# ──────────────────────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    w = word.lower().rstrip("e")
    count = sum(
        1 for i, c in enumerate(w)
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
    
    n_chars = len(w)
    n_syl = _count_syllables(w)
    morph = _morpheme_complexity(w)
    bigram_f = _bigram_familiarity(w)
    neigh = _ortho_neighborhood_size(w, aoa)
    
    if corpus_freq and w in corpus_freq:
        rel_freq = corpus_freq[w] / max(corpus_total, 1)
        log_freq = math.log(1 + rel_freq * 1_000_000)
    else:
        log_freq = 0.0
    
    intercept = 8.5
    β_len = 0.30
    β_syl = 0.55
    β_morph = 2.80
    β_big = 1.60
    β_freq = 0.18
    β_neigh = 0.40
    
    estimated = (
        intercept + β_len * (n_chars - 5) + β_syl * (n_syl - 2)
        + β_morph * morph - β_big * bigram_f - β_freq * log_freq
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


# ──────────────────────────────────────────────────────────────────────────
# COHOMOLOGY SCALARS
# ──────────────────────────────────────────────────────────────────────────

def topo_weight(token: str) -> float:
    tl = token.lower()
    base = min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))
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
    cs_age = word_age(aoa, current, corpus_freq, corpus_total)
    boosts = np.zeros(len(candidates), dtype=np.float32)
    for i, c in enumerate(candidates):
        sim = semantic_scalar(current, c)
        tw = (topo_weight(c) + cs_topo) * 0.5
        ab = age_continuity_boost(
            cs_age, word_age(aoa, c, corpus_freq, corpus_total)
        )
        boosts[i] = strength * sim * (1.0 + tw + ab) / 3.0
    return boosts


# ──────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT EMBEDDER
# ──────────────────────────────────────────────────────────────────────────

class LengthDependentEmbedder:
    """Embedder supporting both words and syntactic forms."""
    
    def embed(self, token: str, dim: Optional[int] = None) -> np.ndarray:
        d = dim if dim is not None else length_dim(token)
        raw_bytes = hashlib.sha256(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)
    
    def embed_form(self, form: SyntacticForm, dim: Optional[int] = None) -> np.ndarray:
        form_str = form.to_string()
        d = dim if dim is not None else length_dim(form.word)
        raw_bytes = hashlib.sha256(form_str.encode("utf-8")).digest()
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)
    
    def shift_vector(self, token: str, dim: int, magnitude: float) -> np.ndarray:
        raw_bytes = hashlib.md5(token.encode("utf-8")).digest()
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
        self, w1: str, w2: str, candidates: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(candidates)
        pass1_raw = np.zeros(N, dtype=np.float32)
        pass2_raw = np.zeros(N, dtype=np.float32)
        topo_kernels = np.zeros(N, dtype=np.float32)
        
        anchor_shift_mag = length_shift_mag(w2)
        anchor_agree_bonus = length_agreement_bonus(w2)
        
        for i, c in enumerate(candidates):
            dim = length_dim(c)
            e_w2 = self.embed(w2, dim=dim)
            e_c = self.embed(c, dim=dim)
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
        base_combined = 0.5 * (p1 + p2)
        agreement_part = float(anchor_agree_bonus) * de_score
        combined = base_combined + topo_kernels * agreement_part
        combined = self._norm01(combined)
        
        return p1, p2, combined


DoubleEntendreEmbedder = LengthDependentEmbedder

# ──────────────────────────────────────────────────────────────────────────
# LANGUAGE MODEL
# ──────────────────────────────────────────────────────────────────────────

class NGramLM:
    def __init__(self, add_k: float = 1.5):
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
            c12 = self.bi.get((w1, w2), 0)
            c123 = self.tri.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V)
            return (self.uni.get(w3, 0) + k) / (self.total + k * V)
        
        probs = torch.tensor([prob(w) for w in cands], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cands, probs


# ──────────────────────────────────────────────────────────────────────────
# TOKENIZER / DETOKENIZER
# ──────────────────────────────────────────────────────────────────────────

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
        if t in ".,;:?)":
            if out:
                out[-1] += t
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


# ──────────────────────────────────────────────────────────────────────────
# CORPUS STATE
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusState:
    lm: NGramLM
    embedder: LengthDependentEmbedder
    aoa: Dict[str, float]
    sentence_form_plan: SentenceFormPlan = field(default_factory=SentenceFormPlan)
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int] = field(default_factory=dict)
    corpus_total: int = 1


def build_state(text: str, aoa: Dict[str, float], prompt: str = "") -> CorpusState:
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)
    embedder = LengthDependentEmbedder()
    
    total = max(1, sum(lm.uni.values()))
    token_boost: Dict[str, float] = {}
    for tok, freq in lm.uni.items():
        if len(tok) > 3 and tok not in STOP_WORDS and re.match(r"^[a-z]", tok):
            token_boost[tok] = min(0.5, math.log(1 + (freq / total) * 1000.0) * 0.1)
    
    # Extract words from prompt
    prompt_tokens = tokenize(prompt)
    alpha_tokens = [t for t in prompt_tokens if re.match(r"^[a-z]", t) and t not in STOP_WORDS]
    
    # Build form plan: 100 forms, one per sentence
    form_plan = SentenceFormPlan()
    form_plan.build_forms(alpha_tokens if alpha_tokens else ["word"])
    
    return CorpusState(
        lm=lm,
        embedder=embedder,
        aoa=aoa,
        sentence_form_plan=form_plan,
        token_boost=token_boost,
        corpus_freq=lm.uni,
        corpus_total=total,
    )


# ──────────────────────────────────────────────────────────────────────────
# GENERATOR: 100 SENTENCES, ONE FORM EACH
# ──────────────────────────────────────────────────────────────────────────

def next_probs(
    state: CorpusState,
    w1: str,
    w2: str,
    sentence_index: int,
    temp: float = 1.2,
    de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    """Generate next word probabilities for a sentence."""
    cands, base_probs = state.lm.next_dist(w1, w2)
    
    _, _, de_combined = state.embedder.length_dependent_weights(
        w1=w1, w2=w2, candidates=cands,
    )
    de_t = torch.tensor(de_combined, dtype=torch.float32)
    
    # Boost for current sentence's form
    current_form = state.sentence_form_plan.form_by_sentence.get(sentence_index)
    form_boost = torch.zeros_like(de_t)
    
    if current_form:
        for idx, c in enumerate(cands):
            # Boost words semantically similar to the form's word
            sim = semantic_similarity(current_form.word, c)
            form_boost[idx] = 0.25 * sim  # 25% strength boost
    
    cb = centroid_boost(
        state.aoa, w2, cands, strength=0.10,
        corpus_freq=state.corpus_freq, corpus_total=state.corpus_total,
    )
    cb_t = torch.tensor(cb, dtype=torch.float32)
    
    tb = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)
    
    w2_age = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array(
        [age_continuity_boost(
            w2_age, word_age(state.aoa, c, state.corpus_freq, state.corpus_total),
        ) for c in cands],
        dtype=np.float32,
    )
    age_t = torch.tensor(age_arr, dtype=torch.float32)
    
    topo_kernels = torch.tensor(
        [length_topo_kernel(c) for c in cands], dtype=torch.float32
    )
    topo_cb = cb_t * (0.5 + 0.5 * topo_kernels)
    
    # Combine: base + double-entendre + centroid + form_boost + token + age
    boosts = float(de_strength) * de_t + topo_cb + 0.10 * tb + 0.15 * age_t + form_boost
    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    logits = logits / max(float(temp), 1e-6)
    probs = F.softmax(logits, dim=-1)
    
    return cands, probs


def generate_100_sentences(
    state: CorpusState,
    prompt: str,
    seed: int = 42,
    tokens_per_sentence: int = 15,
    temp: float = 1.2,
) -> str:
    """Generate exactly 100 sentences, one form per sentence."""
    rng = np.random.default_rng(int(seed))
    seed_toks = tokenize(prompt)
    sw = [t for t in seed_toks if re.match(r"^[a-z]", t)]
    w1 = sw[-2] if len(sw) >= 2 else (sw[0] if sw else "the")
    w2 = sw[-1] if sw else "concept"
    
    result_sentences: List[str] = []
    
    for sent_idx in range(100):
        sentence_tokens: List[str] = []
        alpha_count = 0
        influenced_words: Set[str] = set()
        
        # Generate tokens for this sentence
        for _ in range(int(tokens_per_sentence)):
            cands, probs = next_probs(
                state, w1, w2, sentence_index=sent_idx, temp=float(temp)
            )
            
            p = probs.detach().cpu().numpy()
            p = p / (p.sum() + 1e-12)
            tok = cands[int(rng.choice(len(cands), p=p))]
            sentence_tokens.append(tok)
            
            # Track influenced words
            current_form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
            if current_form and semantic_similarity(current_form.word, tok) > 0.4:
                influenced_words.add(tok)
            
            w1, w2 = w2, tok
            
            if re.match(r"[A-Za-z]", tok):
                alpha_count += 1
            
            # End sentence on period
            if tok in ".!?" or len(sentence_tokens) >= tokens_per_sentence + 2:
                break
        
        # Detokenize and record
        sentence_text = detokenize(sentence_tokens).strip()
        result_sentences.append(sentence_text)
        
        # Record in form plan
        form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if form:
            state.sentence_form_plan.record_sentence_generation(
                sent_idx, sentence_text, form_activation=1.0,
                influenced_words=list(influenced_words)
            )
    
    return "\n".join(result_sentences)


# ──────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ──────────────────────────────────────────────────────────────────────────

def load_corpus(
    use_hf: bool, hf_dataset: str, hf_split: str, hf_max_rows: int, text_file,
) -> str:
    if use_hf:
        ds = load_dataset(hf_dataset, split=hf_split)
        rows = min(int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds), len(ds))
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        return "\n".join(str(x) for x in ds.select(range(rows))[col])
    if text_file is None:
        raise ValueError("No file provided.")
    path = text_file if isinstance(text_file, str) else (
        text_file.name if hasattr(text_file, "name") else str(text_file.get("path", ""))
    )
    return Path(path).read_text(encoding="utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────
# MAIN SESSION
# ──────────────────────────────────────────────────────────────────────────

def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows, text_file, prompt,
    seed, num_sentences, tokens_per_sentence, temp,  # 10 inputs total
    progress=gr.Progress(),
):
    try:
        progress(0.05, desc="Loading AoA dataset (Kuperman 2012)…")
        aoa = load_aoa_dataset()
        
        progress(0.15, desc="Loading corpus…")
        text = load_corpus(bool(use_hf), str(hf_dataset), str(hf_split),
                          int(hf_max_rows), text_file)
        
        progress(0.30, desc="Building language model and form plan…")
        state = build_state(text, aoa, prompt=str(prompt))
        
        progress(0.50, desc="Generating sentences (one form per sentence)…")
        sentences = generate_100_sentences(
            state, str(prompt), seed=int(seed),
            tokens_per_sentence=int(tokens_per_sentence),
            temp=float(temp)
        )
        
        progress(0.80, desc="Analyzing form activation…")
        form_report = state.sentence_form_plan.generate_report()
        
        return sentences, form_report
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", ""


def toggle_hf(val):
    return (
        gr.update(visible=val),
        gr.update(visible=val),
        gr.update(visible=val),
        gr.update(visible=not val),
    )

def build_app():
    with gr.Blocks(
        title="NeuroSymbolic V8.6+ — N Sentences, One Form Each",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# NeuroSymbolic V8.6+ — N Sentences"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset", value=True)
                hf_dataset = gr.Textbox(
                    label="HF Dataset",
                    value="AiresPucrs/stanford-encyclopedia-philosophy"
                )
                hf_split = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(0, 2000, value=300, step=100, label="Max rows")
                text_file = gr.File(
                    label="Upload .txt/.md", file_types=[".txt", ".md"], visible=False
                )
                use_hf.change(toggle_hf, [use_hf],
                             [hf_dataset, hf_split, hf_max_rows, text_file])
                
                seed = gr.Number(value=42, label="Seed")
                num_sentences = gr.Slider(
                    1, 500, value=100, step=10, 
                    label="Number of Sentences"
                )
                tokens_per_sentence = gr.Slider(8, 180, value=15, step=2, 
                                               label="Tokens per Sentence")
                temp = gr.Slider(0.8, 2.5, value=1.2, step=0.1, label="Temperature")
            
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt (extracts words for N forms)",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Generate Sentences", variant="primary", size="lg")
        
        gr.Markdown("## Generated Sentences (One Form Per Sentence)")
        output_sentences = gr.Textbox(label="Sentences", lines=40)
        
        gr.Markdown("## Form Activation Analysis")
        output_report = gr.Textbox(label="Form Report", lines=40)
        
        btn.click(
            run_session,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows, text_file, prompt,
                   seed, num_sentences, tokens_per_sentence, temp],
            outputs=[output_sentences, output_report],
        )
        
        gr.Markdown(
            "### Key Features\\n"
            "- **N Sentences:** Adjustable 1-500 via slider\\n"
            "- **Form Count:** Matches N (cycles through 100+ form types)\\n"
            "- **Form Boost:** Semantic similarity to form's word\\n"
            "- **Activation Tracking:** Cumulative value + influence map\\n"
            "- **Length-Dependent Topology:** Preserved from V8.6"
        )
    
    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)

