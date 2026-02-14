#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuroSymbolic V6.1 - Decision Theory Analyzer
Multi-Entity Problem Solving with Comparative Decision Analysis

Generates two problem-solving sessions from different prompts, then:
1. Shows both sessions side-by-side
2. Analyzes key differences in reasoning patterns
3. Reveals what the system "prefers" for each prompt based on differences
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict, Counter

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


# ----------------------------
# CORE SYMBOLIC ENGINE (Targets)
# ----------------------------

@dataclass
class SymbolicPair:
    token1: str
    token2: str
    harmony: float
    density: float
    momentum: float
    resonance: float

    @classmethod
    def from_tokens(cls, t1: str, t2: str) -> "SymbolicPair":
        harmony = 1.0 - (cls._edit_distance(t1, t2) / max(len(t1), len(t2), 1))
        density = math.tanh((len(t1) + len(t2)) / 20.0)
        if len(t1) > 0:
            momentum = (len(t2) - len(t1)) / (len(t1) + len(t2))
            momentum = (momentum + 1.0) / 2.0
        else:
            momentum = 0.5
        pair_str = f"{t1}|{t2}"
        hash_val = int(hashlib.md5(pair_str.encode()).hexdigest(), 16)
        resonance = (hash_val % 10000) / 10000.0
        return cls(t1, t2, harmony, density, momentum, resonance)

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return SymbolicPair._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def aesthetic_vector(self) -> np.ndarray:
        return np.array([self.harmony, self.density, self.momentum, self.resonance], dtype=np.float32)


# ----------------------------
# DIFFERENTIABLE POSITION + ACTIVATOR
# ----------------------------

class NeuronalActivator(nn.Module):
    def __init__(self, vocab_size: int = 50000, emb_dim: int = 64, hidden: int = 96,
                 pos_fourier: int = 16):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.pos_fourier = int(pos_fourier)

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)

        in_dim = 2 * self.emb_dim + 2 * self.pos_fourier
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

        self.register_buffer("global_mean4", torch.full((4,), 0.5, dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _hash_token_to_id(tok: str, mod: int) -> int:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        return int(h % mod)

    def token_ids(self, toks: List[str], device: torch.device) -> torch.LongTensor:
        ids = [self._hash_token_to_id(t, self.vocab_size) for t in toks]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _pos_fourier(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.view(1)
        freqs = torch.arange(1, self.pos_fourier + 1, device=x.device, dtype=x.dtype) * (2.0 * math.pi)
        ang = x.view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def predict_vec4(self, t1_ids: torch.LongTensor, t2_ids: torch.LongTensor, x_pos: torch.Tensor) -> torch.Tensor:
        e1 = self.emb(t1_ids)
        e2 = self.emb(t2_ids)
        pf = self._pos_fourier(x_pos.to(dtype=e1.dtype))
        if pf.shape[0] != e1.shape[0]:
            pf = pf.expand(e1.shape[0], -1)
        z = torch.cat([e1, e2, pf], dim=-1)
        vec4 = torch.sigmoid(self.mlp(z))
        return vec4

    def weight_from_vec4(self, vec4: torch.Tensor) -> torch.Tensor:
        d = torch.linalg.norm(vec4 - self.global_mean4.view(1, 4), dim=-1)
        return torch.exp(-d)

    @torch.no_grad()
    def update_global_mean(self, vec4_all: torch.Tensor):
        self.global_mean4.copy_(vec4_all.mean(dim=0).clamp(0.0, 1.0))

    def forward_weight(self, t1: str, t2_list: List[str], x_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.emb.weight.device
        t1_ids = self.token_ids([t1] * len(t2_list), device=device)
        t2_ids = self.token_ids(t2_list, device=device)
        vec4 = self.predict_vec4(t1_ids, t2_ids, x_pos=x_pos)
        w = self.weight_from_vec4(vec4)
        return w, vec4

    def bootstrap_on_tokens(self, tokens: List[str], epochs: int = 25, lr: float = 3e-3,
                            max_pairs: int = 4000, progress=None) -> Dict[str, float]:
        if len(tokens) < 2:
            return {"pairs": 0, "loss": 0.0}

        N = len(tokens)
        pairs = []
        for i in range(N - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if t1 in COGNITIVE_TOKENS or t2 in COGNITIVE_TOKENS:
                continue
            x = float(i / max(1, (N - 2)))
            pairs.append((t1, t2, x))
            if len(pairs) >= max_pairs:
                break

        if not pairs:
            return {"pairs": 0, "loss": 0.0}

        y = torch.tensor([SymbolicPair.from_tokens(a, b).aesthetic_vector() for a, b, _ in pairs], dtype=torch.float32)
        self.update_global_mean(y)

        device = self.emb.weight.device
        t1_ids = self.token_ids([a for a, _, _ in pairs], device=device)
        t2_ids = self.token_ids([b for _, b, _ in pairs], device=device)
        x_pos = torch.tensor([x for _, _, x in pairs], dtype=torch.float32, device=device)

        y = y.to(device=device)

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        self.train()
        bs = 128
        losses = []

        for ep in range(int(epochs)):
            perm = torch.randperm(len(pairs), device=device)
            ep_loss = 0.0
            for k in range(0, len(pairs), bs):
                idx = perm[k:k+bs]
                pred = self.predict_vec4(t1_ids[idx], t2_ids[idx], x_pos[idx])
                loss = loss_fn(pred, y[idx])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())

            losses.append(ep_loss / max(1, math.ceil(len(pairs) / bs)))

        self.eval()
        return {"pairs": len(pairs), "loss": float(losses[-1]) if losses else 0.0}


# ----------------------------
# CONSTANTS
# ----------------------------

COGNITIVE_TOKENS = {
    "[PROBLEM]": 2.5,
    "[SOLUTION]": 3.0,
    "[PAIR-BEGIN]": 1.5,
    "[PAIR-END]": 1.5,
}

STOP_WORDS = set(
    """
    a an and are as at be by for from has have he her hers him his i in is it its me my
    of on or our ours she so that the their them they this to was we were what when where
    which who will with you your yours
    """.split()
)

PROBLEM_PATTERNS = [
    r"(?:problem|issue|challenge|difficulty|question|dilemma|paradox|conundrum|puzzle|obstacle)[s]?\s*(?:of|in|with|for|to)?\s*(?:the\s+)?(?:\w+\s*){0,10}\?",
    r"(?:what|how|why|when|where)\s+\w+\s+(?:problem|issue|challenge|question)[s]?\??",
]

SOLUTION_PATTERNS = [
    r"(?:solution|answer|resolution|fix|remedy|approach|method|strategy|technique)[s]?\s*(?:is|are|to|for|of)?\s*(?:the\s+)?(?:\w+\s*){0,8}(?:\.|:)",
    r"(?:solved|resolved|fixed|addressed|overcome|tackled)\s+(?:by|using|through|with)\s*(?:the\s+)?(?:\w+\s*){0,10}",
]


# ----------------------------
# TEXT PROCESSING
# ----------------------------

def inject_cognitive_tokens(text: str) -> str:
    lines = text.split("\n")
    marked = []
    for i, line in enumerate(lines):
        orig_line = line
        modified = False
        for pat in SOLUTION_PATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                line = f"[SOLUTION] {line}"
                modified = True
                break
        if not modified:
            for pat in PROBLEM_PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    line = f"[PROBLEM] {line}"
                    break
        if i > 0 and marked and ("?" in marked[-1]) and ("[SOLUTION]" not in line):
            if any(w in orig_line.lower() for w in ["answer", "solution", "because", "since", "therefore"]):
                line = f"[SOLUTION] {line}"
        marked.append(line)
    return "\n".join(marked)

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def basic_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    tokens = re.findall(r"\[[A-Z\-]+\]|[A-Za-z][A-Za-z0-9_'-]*|[.,;:!?()]", text)
    out = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            out.append(t)
        elif re.match(r"[A-Za-z]", t):
            out.append(t.lower())
        else:
            out.append(t)
    return out

def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            continue
        if t in [".", ",", ";", ":", "!", "?", ")", "("]:
            if t in ["(", ")"]:
                out.append(t)
            else:
                if out:
                    out[-1] += t
                else:
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

def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in [".txt", ".md"]:
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError("Unsupported file extension")


# ----------------------------
# PROBLEM FLOW FIELD
# ----------------------------

def compute_problem_flow_by_token(tokens: List[str]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    in_seg = False
    seg_start = None
    for i, tok in enumerate(tokens):
        if tok == "[PROBLEM]":
            in_seg = True
            seg_start = i
        elif tok == "[SOLUTION]" and in_seg and seg_start is not None:
            seg_end = i
            L = seg_end - seg_start
            if L > 1:
                for j in range(seg_start + 1, seg_end):
                    t = tokens[j]
                    if t in COGNITIVE_TOKENS:
                        continue
                    pos = (j - seg_start) / max(1, L)
                    acc.setdefault(t, []).append(float(pos))
            in_seg = False
            seg_start = None
    return {t: float(sum(v) / len(v)) for t, v in acc.items()}


# ----------------------------
# TF-IDF + SVD
# ----------------------------

def pure_tfidf(docs: List[str], max_features: int = 8000) -> Tuple[np.ndarray, List[str]]:
    all_words = set()
    for doc in docs:
        words = re.findall(r"\b\w+\b", doc.lower())
        all_words.update(words)
    vocab = list(all_words)[:max_features]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        word_counts = {}
        for word in re.findall(r"\b\w+\b", doc.lower()):
            word_counts[word] = word_counts.get(word, 0) + 1
        if not word_counts:
            continue
        uniq = len(word_counts)
        for word, count in word_counts.items():
            if word in word_to_idx:
                j = word_to_idx[word]
                tf = count / uniq
                df = sum(1 for d in docs if word in d.lower())
                idf = math.log(len(docs) / (1 + df))
                X[i, j] = tf * idf
    return X, vocab

def pure_truncated_svd(X: np.ndarray, n_components: int, random_state: int = 42) -> Any:
    np.random.seed(random_state)
    m, n = X.shape
    k = min(n_components, min(m, n))
    if k < 1:
        return type("SVD", (), {"components_": np.zeros((0, n), dtype=np.float32)})()

    Q = np.random.randn(n, k).astype(np.float32)
    Q, _ = np.linalg.qr(Q)

    for _ in range(10):
        B = X.T @ X @ Q
        Q, _ = np.linalg.qr(B)

    B = X @ Q
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return type("SVD", (), {"components_": Vt[:k].astype(np.float32)})()


# ----------------------------
# GRAPH
# ----------------------------

@dataclass
class SimpleGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, Dict[str, Any]]]
    cognitive_map: Dict[str, List[int]]
    pairwise_aesthetics: Dict[Tuple[int, int], torch.Tensor]

    @classmethod
    def from_token_sequence(cls, tokens: List[str], activator: NeuronalActivator,
                            max_nodes: int = 220, x_pos_default: float = 0.5):
        toks = tokens[:max_nodes]
        nodes = [{"id": i, "token": t} for i, t in enumerate(toks)]
        edges = []
        cog_map = {"[PROBLEM]": [], "[SOLUTION]": [], "[PAIR-BEGIN]": [], "[PAIR-END]": []}
        aest = {}

        if len(toks) >= 2:
            device = activator.emb.weight.device
            x = torch.tensor(float(x_pos_default), dtype=torch.float32, device=device)
            for i in range(len(toks) - 1):
                edges.append((i, i + 1, {"rel": "adj"}))
                t1, t2 = toks[i], toks[i + 1]
                w, vec4 = activator.forward_weight(t1, [t2], x_pos=x)
                aest[(i, i + 1)] = vec4[0].detach().cpu()

        for i in range(len(toks) - 2):
            edges.append((i, i + 2, {"rel": "skip"}))

        for i, t in enumerate(toks):
            if t in COGNITIVE_TOKENS:
                cog_map[t].append(i)

        return cls(nodes, edges, cog_map, aest)

    def get_aesthetic_flow(self) -> float:
        if not self.pairwise_aesthetics:
            return 0.5
        vecs = list(self.pairwise_aesthetics.values())
        if not vecs:
            return 0.5
        V = torch.stack(vecs, dim=0)
        return float(torch.mean(torch.linalg.norm(V, dim=1)).item())


# ----------------------------
# FUZZY CONTROLLER
# ----------------------------

def mf_tri(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    x = x.clamp(min=min(a, c), max=max(a, c))
    left = (x - a) / max(1e-9, (b - a))
    right = (c - x) / max(1e-9, (c - b))
    return torch.clamp(torch.minimum(left, right), 0.0, 1.0)

def mf_trap(x: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    x = x.clamp(min=min(a, d), max=max(a, d))
    up = (x - a) / max(1e-9, (b - a))
    down = (d - x) / max(1e-9, (d - c))
    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.clamp(torch.minimum(torch.minimum(up, one), down), 0.0, 1.0)

def tnorm_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def snorm_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)

class FuzzyWeightController(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_low = 0.20
        self.z_mid = 0.55
        self.z_high = 0.95

    def forward(self, entropy01: torch.Tensor, peak01: torch.Tensor,
                boost01: torch.Tensor, aesthetic_flow01: torch.Tensor,
                osculator_strength: float = 0.0) -> torch.Tensor:
        e = entropy01.clamp(0, 1)
        p = peak01.clamp(0, 1)
        b = boost01.clamp(0, 1)
        a = aesthetic_flow01.clamp(0, 1)

        e_low  = mf_trap(e, 0.0, 0.0, 0.25, 0.45)
        e_mid  = mf_tri(e, 0.25, 0.50, 0.75)
        e_high = mf_trap(e, 0.55, 0.75, 1.0, 1.0)

        p_low  = mf_trap(p, 0.0, 0.0, 0.20, 0.40)
        p_mid  = mf_tri(p, 0.25, 0.50, 0.75)
        p_high = mf_trap(p, 0.60, 0.80, 1.0, 1.0)

        b_low  = mf_trap(b, 0.0, 0.0, 0.20, 0.45)
        b_mid  = mf_tri(b, 0.25, 0.50, 0.75)
        b_high = mf_trap(b, 0.55, 0.80, 1.0, 1.0)

        a_high = mf_trap(a, 0.5, 0.7, 1.0, 1.0)

        w1 = tnorm_prod(e_high, p_low)
        w2 = tnorm_prod(e_mid, b_mid)
        w3 = snorm_max(p_high, b_high)
        w4 = tnorm_prod(e_low, p_mid)
        w5 = tnorm_prod(a_high, e_mid)

        Z = torch.tensor([self.z_high, self.z_mid, self.z_low, self.z_low, self.z_high],
                         device=e.device, dtype=torch.float32)
        W = torch.stack([w1, w2, w3, w4, w5]).to(dtype=torch.float32).clamp_min(0.0)

        g = (W * Z).sum() / (W.sum() + 1e-12)

        s = float(osculator_strength)
        if math.isfinite(s) and s > 0.0:
            s = max(0.0, min(1.0, s))
            osc = 1.0 - ((g - 0.5) / 0.5) ** 2
            osc = osc.clamp(0.0, 1.0)
            g = (1.0 - s) * g + s * osc

        return g.clamp(0.0, 0.5)


class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.15, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = int(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)


# ----------------------------
# LM
# ----------------------------

class QuadgramLM:
    def __init__(self, add_k: float = 0.25):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        self.uni.clear(); self.bi.clear(); self.tri.clear(); self.quad.clear()
        self.total = 0
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            self.bi[k] = self.bi.get(k, 0) + 1
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.tri[k] = self.tri.get(k, 0) + 1
        for i in range(len(tokens) - 3):
            k = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
            self.quad[k] = self.quad.get(k, 0) + 1
        self.vocab = list(self.uni.keys())

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        cont = []
        for (a, b, c, d), _count in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c), _count in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b), _count in self.bi.items():
                if a == w3:
                    cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]

        seen = set()
        cand = []
        for w in cont:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w)
                cand.append(w)
        cand = cand[:500]

        V = len(self.vocab) + 1
        add_k = self.add_k

        def get_prob(w4: str) -> float:
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0:
                return (c1234 + add_k) / (c123 + add_k * V)
            c12 = self.bi.get((w2, w3), 0)
            c123_tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0:
                return (c123_tri + add_k) / (c12 + add_k * V)
            c1 = self.uni.get(w3, 0)
            c12_bi = self.bi.get((w3, w4), 0)
            if c1 > 0:
                return (c12_bi + add_k) / (c1 + add_k * V)
            return (self.uni.get(w4, 0) + add_k) / (self.total + add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        if probs.numel() > 0:
            probs = probs / (probs.sum() + 1e-12)
        else:
            cand = ["the"]
            probs = torch.ones(1, dtype=torch.float32)
        return cand, probs


# ----------------------------
# HEMICONTINUITY
# ----------------------------

class DiscreteHemiContinuity(nn.Module):
    def __init__(self, top_k: int = 64, mass_eps: float = 1e-4, penalty_strength: float = 0.15, smooth_alpha: float = 0.05):
        super().__init__()
        self.top_k = int(top_k)
        self.mass_eps = float(mass_eps)
        self.penalty_strength = float(penalty_strength)
        self.smooth_alpha = float(smooth_alpha)

        self._prev_cand: Optional[List[str]] = None
        self._prev_probs: Optional[torch.Tensor] = None
        self._prev_index: Optional[Dict[str, int]] = None

    def reset(self):
        self._prev_cand = None
        self._prev_probs = None
        self._prev_index = None

    def _topk_mask(self, probs: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, min(int(k), int(probs.numel())))
        idx = torch.topk(probs, k=k, largest=True, sorted=False).indices
        m = torch.zeros_like(probs, dtype=torch.bool)
        m[idx] = True
        return m

    def apply(self, cand: List[str], probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if probs.dim() != 1:
            raise ValueError("DiscreteHemiContinuity expects 1D probs")

        hemi_pen = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        if self._prev_cand is None or self._prev_probs is None or self._prev_index is None:
            self._prev_cand = list(cand)
            self._prev_probs = probs.detach().clone()
            self._prev_index = {w: i for i, w in enumerate(self._prev_cand)}
            return probs, hemi_pen

        prev_index = self._prev_index
        prev_probs = self._prev_probs.to(device=probs.device, dtype=probs.dtype)

        prev_on_curr = torch.zeros_like(probs)
        for i, w in enumerate(cand):
            j = prev_index.get(w, None)
            if j is not None and j < prev_probs.numel():
                prev_on_curr[i] = prev_probs[j]

        prev_mask = self._topk_mask(prev_on_curr, self.top_k)
        curr_mask = self._topk_mask(probs, self.top_k)

        new_outside = curr_mask & (~prev_mask)
        upper_violation_mass = probs[new_outside].sum()
        hemi_pen = hemi_pen + upper_violation_mass

        lost = prev_mask & (~curr_mask)
        lower_violation_mass = prev_on_curr[lost].sum()
        hemi_pen = hemi_pen + lower_violation_mass

        if self.smooth_alpha > 0.0:
            alpha = float(self.smooth_alpha)
            probs_adj = probs * (1.0 - alpha) + prev_on_curr * alpha
            probs_adj = probs_adj / (probs_adj.sum() + 1e-12)
        else:
            probs_adj = probs

        self._prev_cand = list(cand)
        self._prev_probs = probs_adj.detach().clone()
        self._prev_index = {w: i for i, w in enumerate(self._prev_cand)}

        return probs_adj, (self.penalty_strength * hemi_pen).detach()


# ----------------------------
# STATE
# ----------------------------

@dataclass
class Nodelet:
    idx: int
    top_terms: List[Tuple[str, float]]
    energy: float
    narrative: str

@dataclass
class ModelState:
    nodelets: List[Nodelet]
    vocab100: List[str]
    binding_W: torch.Tensor
    bar_probs: torch.Tensor
    token_boost: Dict[str, float]
    semantic_graph: SimpleGraph
    lm: QuadgramLM
    activator: NeuronalActivator
    problem_flow_by_token: Dict[str, float]

@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    state: ModelState


class RadixLRUCache:
    def __init__(self, max_items: int = 25000):
        self.max_items = int(max(256, max_items))
        self._od = OrderedDict()

    def get(self, key):
        v = self._od.get(key, None)
        if v is None:
            return None
        self._od.move_to_end(key)
        return v

    def put(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.max_items:
            self._od.popitem(last=False)

    def clear(self):
        self._od.clear()


# ----------------------------
# GENERATOR + DECISION ANALYZER
# ----------------------------

@dataclass
class GenerationMetrics:
    """Tracks metrics for decision analysis"""
    role_token_counts: Dict[str, int]
    role_avg_flow: Dict[str, float]
    keyword_usage: Counter
    sentence_lengths: List[int]
    question_count: int
    assertion_count: int
    avg_aesthetic_flow: float
    total_tokens: int


class NeuroSymbolicGraphGenerator:
    def __init__(self,
                 nodelets_n: int = 10,
                 bars_n: int = 100,
                 svd_random_state: int = 7,
                 softmax_temp: float = 0.85,
                 steer_strength: float = 1.35,
                 lm_add_k: float = 0.25,
                 focus_strength: float = 0.5,
                 pairwise_strength: float = 0.4,
                 osculator_strength: float = 0.1,
                 activator_boot_epochs: int = 25,
                 hemi_enable: bool = True,
                 hemi_top_k: int = 64,
                 hemi_strength: float = 0.15,
                 hemi_smooth_alpha: float = 0.05):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp = float(softmax_temp)
        self.lm_add_k = float(lm_add_k)
        self.base_steer = float(steer_strength)
        self.base_temp = float(softmax_temp)
        self.pairwise_strength = float(pairwise_strength)
        self.osculator_strength = float(osculator_strength)
        self.activator_boot_epochs = int(activator_boot_epochs)

        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.fuzzy_ctl = FuzzyWeightController()

        self.cache = RadixLRUCache(max_items=20000)
        self.cache_version = 0

        self.hemi_enable = bool(hemi_enable)
        self.hemi = DiscreteHemiContinuity(
            top_k=int(hemi_top_k),
            penalty_strength=float(hemi_strength),
            smooth_alpha=float(hemi_smooth_alpha),
        )

    def _pick_initial_context(self, lm: QuadgramLM, seed_words: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_'-]*$", t) and t not in COGNITIVE_TOKENS]
        if len(sw) >= 3:
            return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2:
            return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1:
            return (sw[-1], sw[-1], sw[-1])
        seed_tok = lm.vocab[0] if lm.vocab else "the"
        return (seed_tok, seed_tok, seed_tok)

    def build_state(self, text: str, progress=None) -> ModelState:
        tokens = basic_tokenize(text)
        flow_by_token = compute_problem_flow_by_token(tokens)

        lm = QuadgramLM(add_k=self.lm_add_k)
        lm.ingest(tokens)

        token_boost: Dict[str, float] = {}
        for tok, boost_val in COGNITIVE_TOKENS.items():
            token_boost[tok] = boost_val

        activator = NeuronalActivator()
        if torch.cuda.is_available():
            activator = activator.cuda()
        activator.bootstrap_on_tokens(tokens, epochs=self.activator_boot_epochs, progress=progress)

        G = SimpleGraph.from_token_sequence(tokens, activator, max_nodes=220, x_pos_default=0.5)

        clean_text = text.replace("[PROBLEM]", "").replace("[SOLUTION]", "").replace("[PAIR-BEGIN]", "").replace("[PAIR-END]", "")
        docs = re.split(r"\n\s*\n", clean_text)[:500]
        X, vocab = pure_tfidf(docs, max_features=8000)

        nodelets: List[Nodelet] = []
        vocab100: List[str] = []
        W = torch.zeros(0, 0)
        probs = torch.ones(1)

        if X.size != 0 and len(vocab) != 0:
            top_idx = np.argsort(-X.sum(axis=0))[: self.bars_n]
            vocab100 = [vocab[i] for i in top_idx]
            X_svd = X[:, top_idx]
            n_rows, n_cols = X_svd.shape
            max_rank = min(n_rows, n_cols)
            k = 1 if max_rank <= 1 else min(self.nodelets_n, max_rank, 10)

            svd = pure_truncated_svd(X_svd, n_components=k, random_state=self.svd_random_state)
            for i, comp in enumerate(svd.components_):
                terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))],
                               key=lambda x: -abs(x[1]))[:10]
                eng = float(np.linalg.norm(comp))
                nodelets.append(Nodelet(i, terms, eng, f"Nodelet {i}"))

            W = torch.tensor(svd.components_, dtype=torch.float32)
            W = F.relu(W)
            if W.numel() > 0:
                W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)
                energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32)
                energies = energies / (energies.max() + 1e-12)
                logits = (energies.view(-1, 1) * W).sum(dim=0)
                probs = F.softmax(logits / max(self.softmax_temp, 1e-6), dim=-1)
                probs = self.focus_layer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)

            for w, p in zip(vocab100, probs.detach().cpu().tolist()):
                for subw in w.split():
                    if len(subw) > 2 and subw not in STOP_WORDS:
                        token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            token_boost=token_boost,
            semantic_graph=G,
            lm=lm,
            activator=activator,
            problem_flow_by_token=flow_by_token,
        )

    def prepare_corpus(self, raw_text: str, progress=None) -> PreparedCorpus:
        text = inject_cognitive_tokens(raw_text)
        text = normalize(text)
        state = self.build_state(text, progress=progress)
        tokens = basic_tokenize(text)

        if self.hemi_enable:
            self.hemi.reset()

        return PreparedCorpus(text=text, tokens=tokens, state=state)

    def _final_probs(self, prep: PreparedCorpus, w1: str, w2: str, w3: str,
                     x_pos: torch.Tensor, allow_cache: bool = True) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        cache_ok = allow_cache and (not x_pos.requires_grad) and (x_pos.numel() == 1)

        key = None
        if cache_ok:
            key = (self.cache_version, w1, w2, w3, float(x_pos.item()))
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        cand, base_probs = prep.state.lm.next_distribution(w1, w2, w3)
        base_p = base_probs.to(dtype=torch.float32)
        base_p = base_p / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        val = base_p.clamp_min(1e-12)
        H = -torch.sum(base_p * torch.log(val))
        V = float(base_p.numel())
        entropy01 = (H / max(1e-9, math.log(max(2.0, V)))).clamp(0.0, 1.0)
        peak01 = base_p.max().clamp(0.0, 1.0)

        device = prep.state.activator.emb.weight.device
        x_pos = x_pos.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand],
                              dtype=torch.float32, device=device)

        w_pair, _vec4 = prep.state.activator.forward_weight(w3, cand, x_pos=x_pos)
        w_pair = w_pair / (w_pair.mean() + 1e-12)
        boosts = boosts + self.pairwise_strength * w_pair

        context_str = f"{w1} {w2} {w3}"
        gravity = 0.0
        if "[PROBLEM]" in context_str:
            gravity = 0.2
        if "[SOLUTION]" in context_str:
            gravity = 0.3
        flow_w3 = float(prep.state.problem_flow_by_token.get(w3, 0.0))
        flow_w3_t = torch.tensor(flow_w3, dtype=torch.float32, device=device)

        boost01 = torch.tanh((boosts.abs().mean() + gravity + flow_w3_t) / 3.0).clamp(0.0, 1.0)

        base_flow = float(prep.state.semantic_graph.get_aesthetic_flow())
        base_flow_t = torch.tensor(base_flow, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        aesthetic_flow01 = (base_flow_t * (0.5 + 0.5 * x_pos)).clamp(0.0, 1.0)

        g = self.fuzzy_ctl(entropy01, peak01, boost01, aesthetic_flow01, osculator_strength=self.osculator_strength)

        effective_steer = self.base_steer * g
        effective_temp = self.base_temp * (1.2 - 0.7 * g)

        potentials = torch.log(base_p.to(device=device).clamp_min(1e-12)) + effective_steer * boosts
        potentials = potentials / torch.clamp(effective_temp, min=1e-6)
        final_probs = F.softmax(potentials, dim=-1)

        hemi_pen = torch.tensor(0.0, device=device, dtype=final_probs.dtype)
        if self.hemi_enable:
            final_probs, hemi_pen = self.hemi.apply(cand, final_probs)

        flow_vec = torch.tensor([prep.state.problem_flow_by_token.get(w, 0.0) for w in cand],
                                dtype=torch.float32, device=device).clamp(0.0, 1.0)

        out = (cand, final_probs, flow_vec)
        if cache_ok and key is not None:
            self.cache.put(key, (cand, final_probs.detach(), flow_vec.detach()))
        return out

    @torch.no_grad()
    def generate(self, prep: PreparedCorpus, prompt: str, start_x: float,
                 max_tokens: int = 220, seed: int = 42, num_speakers: int = 2,
                 tokens_per_turn: int = 50, problem_solving_mode: bool = True) -> Tuple[str, GenerationMetrics]:
        rng = np.random.default_rng(int(seed))
        seed_toks = basic_tokenize(prompt)
        w1, w2, w3 = self._pick_initial_context(prep.state.lm, seed_toks)

        device = prep.state.activator.emb.weight.device
        total_steps = int(max_tokens)
        
        if problem_solving_mode:
            role_definitions = [
                ("Problem Poser", 0.0, 0.25, "questioning", ["?", "what", "how", "why"]),
                ("Analyzer", 0.2, 0.45, "analyzing", ["because", "consider", "examine", "observe"]),
                ("Solution Proposer", 0.4, 0.7, "proposing", ["solution", "approach", "method", "could"]),
                ("Critic", 0.5, 0.75, "critiquing", ["however", "but", "issue", "problem"]),
                ("Synthesizer", 0.7, 1.0, "synthesizing", ["therefore", "thus", "overall", "combining"]),
            ]
            
            speaker_roles = []
            for i in range(int(num_speakers)):
                role_idx = i % len(role_definitions)
                speaker_roles.append(role_definitions[role_idx])
        else:
            speaker_roles = [(f"Speaker {chr(65 + i)}", 0.0, 1.0, "speaking", []) 
                            for i in range(int(num_speakers))]
        
        conversation: List[Tuple[str, str, List[str], float]] = []
        current_speaker_idx = 0
        current_turn_tokens: List[str] = []
        alpha_count_turn = 0
        
        # Metrics tracking
        role_token_counts = {role[0]: 0 for role in speaker_roles}
        role_flow_sums = {role[0]: 0.0 for role in speaker_roles}
        role_flow_counts = {role[0]: 0 for role in speaker_roles}
        keyword_usage = Counter()
        sentence_lengths = []
        question_count = 0
        assertion_count = 0
        aesthetic_flows = []
        
        current_sentence_length = 0

        for i in range(total_steps):
            role_name, x_min, x_max, mode, keywords = speaker_roles[current_speaker_idx]
            
            progress = i / max(1, total_steps)
            global_x = start_x + (1.0 - start_x) * progress
            role_x_bias = (x_min + x_max) / 2.0
            
            curr_x_val = 0.7 * global_x + 0.3 * role_x_bias
            curr_x_val = max(x_min, min(x_max, curr_x_val))
            
            x = torch.tensor(float(curr_x_val), dtype=torch.float32, device=device)

            cand, probs, flow_vec = self._final_probs(prep, w1, w2, w3, x_pos=x, allow_cache=True)
            
            if keywords and problem_solving_mode:
                keyword_boost = torch.zeros_like(probs)
                for idx, c in enumerate(cand):
                    if c.lower() in keywords:
                        keyword_boost[idx] = 0.3
                    flow_match = 1.0 - abs(flow_vec[idx].item() - role_x_bias)
                    keyword_boost[idx] += 0.2 * flow_match
                
                probs = probs * torch.exp(keyword_boost)
                probs = probs / (probs.sum() + 1e-12)
            
            p = probs.detach().cpu().numpy()
            p = p / (p.sum() + 1e-12)
            idx = rng.choice(len(cand), p=p)
            tok = cand[idx]
            current_turn_tokens.append(tok)
            
            # Track metrics
            role_token_counts[role_name] += 1
            role_flow_sums[role_name] += flow_vec[idx].item()
            role_flow_counts[role_name] += 1
            aesthetic_flows.append(curr_x_val)
            
            if tok.lower() in keywords:
                keyword_usage[tok.lower()] += 1
            
            if tok == "?":
                question_count += 1
                sentence_lengths.append(current_sentence_length)
                current_sentence_length = 0
            elif tok in [".", "!"]:
                assertion_count += 1
                sentence_lengths.append(current_sentence_length)
                current_sentence_length = 0
            else:
                current_sentence_length += 1

            w1, w2, w3 = w2, w3, tok
            if re.match(r"[A-Za-z]", tok):
                alpha_count_turn += 1
            
            should_switch = False
            
            if tok in {".", "!", "?"} and alpha_count_turn >= min(tokens_per_turn * 0.6, 20):
                should_switch = True
            elif len(current_turn_tokens) >= tokens_per_turn * 1.5:
                should_switch = True
            elif len(current_turn_tokens) >= tokens_per_turn and alpha_count_turn >= 15:
                if tok in {",", ";", ":"} or (i > 0 and rng.random() < 0.3):
                    should_switch = True
            
            if should_switch and current_turn_tokens:
                conversation.append((role_name, mode, list(current_turn_tokens), role_x_bias))
                current_speaker_idx = (current_speaker_idx + 1) % num_speakers
                current_turn_tokens = []
                alpha_count_turn = 0
        
        if current_turn_tokens:
            role_name, x_min, x_max, mode, keywords = speaker_roles[current_speaker_idx]
            role_x_bias = (x_min + x_max) / 2.0
            conversation.append((role_name, mode, current_turn_tokens, role_x_bias))
        
        # Calculate average flows
        role_avg_flow = {
            role: (role_flow_sums[role] / role_flow_counts[role] if role_flow_counts[role] > 0 else 0.0)
            for role in role_token_counts.keys()
        }
        
        metrics = GenerationMetrics(
            role_token_counts=role_token_counts,
            role_avg_flow=role_avg_flow,
            keyword_usage=keyword_usage,
            sentence_lengths=[s for s in sentence_lengths if s > 0],
            question_count=question_count,
            assertion_count=assertion_count,
            avg_aesthetic_flow=sum(aesthetic_flows) / len(aesthetic_flows) if aesthetic_flows else 0.0,
            total_tokens=total_steps
        )
        
        text = self._format_problem_solving_conversation(conversation, problem_solving_mode)
        return text, metrics
    
    def _format_problem_solving_conversation(self, conversation: List[Tuple[str, str, List[str], float]], 
                                              problem_solving_mode: bool) -> str:
        lines = []
        
        if problem_solving_mode:
            lines.append("=" * 60)
            lines.append("MULTI-ENTITY PROBLEM SOLVING SESSION")
            lines.append("=" * 60)
            lines.append("")
        
        for role, mode, tokens, x_bias in conversation:
            text = detokenize(tokens)
            if text.strip():
                if problem_solving_mode:
                    lines.append(f"[{role.upper()}] ({mode}, flow: {x_bias:.2f})")
                    lines.append(f"{text}")
                    lines.append("")
                else:
                    lines.append(f"{role}: {text}")
                    lines.append("")
        
        if problem_solving_mode:
            lines.append("=" * 60)
            lines.append("END OF SESSION")
            lines.append("=" * 60)
        
        return "\n".join(lines)


def extract_text_differences(text1: str, text2: str) -> Dict[str, Any]:
    """Extract and analyze textual differences between two generations"""
    
    # Extract actual content (skip headers/footers)
    def clean_text(text: str) -> str:
        lines = text.split('\n')
        content_lines = []
        skip_markers = {'=' * 60, 'MULTI-ENTITY', 'END OF SESSION'}
        for line in lines:
            if not any(marker in line for marker in skip_markers):
                if not line.startswith('[') or ']' not in line[:30]:
                    content_lines.append(line)
                elif ']' in line[:30]:
                    # Extract just the speech part after role marker
                    parts = line.split(']', 1)
                    if len(parts) > 1:
                        content_lines.append(parts[1].strip())
        return ' '.join(content_lines)
    
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    # Tokenize into words
    words1 = set(re.findall(r'\b[a-z]{3,}\b', clean1.lower()))
    words2 = set(re.findall(r'\b[a-z]{3,}\b', clean2.lower()))
    
    # Remove common stop words
    stop = {'the', 'and', 'that', 'this', 'with', 'for', 'are', 'was', 'were', 
            'been', 'have', 'has', 'had', 'not', 'but', 'from', 'they', 'which'}
    words1 = words1 - stop
    words2 = words2 - stop
    
    unique_to_1 = words1 - words2
    unique_to_2 = words2 - words1
    common = words1 & words2
    
    # Get word frequencies for unique words
    def get_word_freq(text: str, word_set: set) -> List[Tuple[str, int]]:
        text_lower = text.lower()
        freq = Counter()
        for word in word_set:
            freq[word] = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        return sorted(freq.items(), key=lambda x: -x[1])
    
    freq1 = get_word_freq(clean1, unique_to_1)
    freq2 = get_word_freq(clean2, unique_to_2)
    
    # Calculate similarity metrics
    total_unique = len(words1 | words2)
    similarity = len(common) / total_unique if total_unique > 0 else 0
    
    # Extract distinctive phrases (3-grams that appear in one but not the other)
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}
    
    phrases1 = get_ngrams(clean1)
    phrases2 = get_ngrams(clean2)
    
    unique_phrases_1 = list(phrases1 - phrases2)[:10]
    unique_phrases_2 = list(phrases2 - phrases1)[:10]
    
    return {
        'unique_to_1': freq1[:20],
        'unique_to_2': freq2[:20],
        'common_count': len(common),
        'similarity': similarity,
        'unique_phrases_1': unique_phrases_1,
        'unique_phrases_2': unique_phrases_2,
        'total_words_1': len(words1),
        'total_words_2': len(words2),
    }


def analyze_decision_preferences(metrics1: GenerationMetrics, metrics2: GenerationMetrics,
                                  prompt1: str, prompt2: str, text1: str, text2: str) -> str:
    """Analyze differences between two generations and reveal decision preferences"""
    
    # Extract text differences
    text_diff = extract_text_differences(text1, text2)
    
    analysis = []
    analysis.append("=" * 80)
    analysis.append("DECISION THEORY ANALYSIS")
    analysis.append("Comparative Analysis of Two Problem-Solving Sessions")
    analysis.append("=" * 80)
    analysis.append("")
    
    # 0. TEXT DIFFERENCES SECTION (NEW)
    analysis.append("### 0. VOCABULARY DIFFERENCES")
    analysis.append("")
    analysis.append(f"**Lexical Similarity**: {text_diff['similarity']:.1%}")
    analysis.append(f"**Shared vocabulary**: {text_diff['common_count']} words")
    analysis.append(f"**Unique to Session 1**: {len(text_diff['unique_to_1'])} words")
    analysis.append(f"**Unique to Session 2**: {len(text_diff['unique_to_2'])} words")
    analysis.append("")
    
    analysis.append(f"**Top words ONLY in Session 1** (for prompt: {prompt1[:50]}...):")
    for word, count in text_diff['unique_to_1'][:15]:
        analysis.append(f"  • {word} ({count}x)")
    analysis.append("")
    
    analysis.append(f"**Top words ONLY in Session 2** (for prompt: {prompt2[:50]}...):")
    for word, count in text_diff['unique_to_2'][:15]:
        analysis.append(f"  • {word} ({count}x)")
    analysis.append("")
    
    if text_diff['unique_phrases_1']:
        analysis.append("**Distinctive 3-word phrases in Session 1**:")
        for phrase in text_diff['unique_phrases_1'][:5]:
            analysis.append(f"  • '{phrase}'")
        analysis.append("")
    
    if text_diff['unique_phrases_2']:
        analysis.append("**Distinctive 3-word phrases in Session 2**:")
        for phrase in text_diff['unique_phrases_2'][:5]:
            analysis.append(f"  • '{phrase}'")
        analysis.append("")
    
    analysis.append("=" * 80)
    analysis.append("")
    
    # 1. Role Distribution Comparison
    analysis.append("### 1. ROLE DISTRIBUTION COMPARISON")
    analysis.append("")
    analysis.append(f"**Prompt 1**: {prompt1[:80]}...")
    for role, count in metrics1.role_token_counts.items():
        pct = (count / metrics1.total_tokens) * 100
        analysis.append(f"  - {role}: {count} tokens ({pct:.1f}%)")
    analysis.append("")
    
    analysis.append(f"**Prompt 2**: {prompt2[:80]}...")
    for role, count in metrics2.role_token_counts.items():
        pct = (count / metrics2.total_tokens) * 100
        analysis.append(f"  - {role}: {count} tokens ({pct:.1f}%)")
    analysis.append("")
    
    # 2. Flow Position Analysis
    analysis.append("### 2. FLOW POSITION ANALYSIS")
    analysis.append("(0.0 = problem space, 1.0 = solution space)")
    analysis.append("")
    
    analysis.append("**Prompt 1** - Average flow per role:")
    for role, flow in metrics1.role_avg_flow.items():
        analysis.append(f"  - {role}: {flow:.3f}")
    analysis.append(f"  Overall aesthetic flow: {metrics1.avg_aesthetic_flow:.3f}")
    analysis.append("")
    
    analysis.append("**Prompt 2** - Average flow per role:")
    for role, flow in metrics2.role_avg_flow.items():
        analysis.append(f"  - {role}: {flow:.3f}")
    analysis.append(f"  Overall aesthetic flow: {metrics2.avg_aesthetic_flow:.3f}")
    analysis.append("")
    
    # 3. Communication Pattern Analysis
    analysis.append("### 3. COMMUNICATION PATTERNS")
    analysis.append("")
    
    avg_sent_1 = sum(metrics1.sentence_lengths) / len(metrics1.sentence_lengths) if metrics1.sentence_lengths else 0
    avg_sent_2 = sum(metrics2.sentence_lengths) / len(metrics2.sentence_lengths) if metrics2.sentence_lengths else 0
    
    q_ratio_1 = metrics1.question_count / (metrics1.question_count + metrics1.assertion_count) if (metrics1.question_count + metrics1.assertion_count) > 0 else 0
    q_ratio_2 = metrics2.question_count / (metrics2.question_count + metrics2.assertion_count) if (metrics2.question_count + metrics2.assertion_count) > 0 else 0
    
    analysis.append(f"**Prompt 1**:")
    analysis.append(f"  - Average sentence length: {avg_sent_1:.1f} tokens")
    analysis.append(f"  - Questions: {metrics1.question_count}, Assertions: {metrics1.assertion_count}")
    analysis.append(f"  - Question ratio: {q_ratio_1:.2%}")
    analysis.append("")
    
    analysis.append(f"**Prompt 2**:")
    analysis.append(f"  - Average sentence length: {avg_sent_2:.1f} tokens")
    analysis.append(f"  - Questions: {metrics2.question_count}, Assertions: {metrics2.assertion_count}")
    analysis.append(f"  - Question ratio: {q_ratio_2:.2%}")
    analysis.append("")
    
    # 4. Key Differences
    analysis.append("### 4. KEY DIFFERENCES")
    analysis.append("")
    
    # Compare dominant roles
    max_role_1 = max(metrics1.role_token_counts.items(), key=lambda x: x[1])[0]
    max_role_2 = max(metrics2.role_token_counts.items(), key=lambda x: x[1])[0]
    
    analysis.append(f"**Dominant Role:**")
    analysis.append(f"  - Prompt 1: {max_role_1}")
    analysis.append(f"  - Prompt 2: {max_role_2}")
    analysis.append("")
    
    # Flow space preference
    flow_diff = metrics2.avg_aesthetic_flow - metrics1.avg_aesthetic_flow
    if abs(flow_diff) > 0.1:
        if flow_diff > 0:
            analysis.append(f"**Flow Space**: Prompt 2 operates more in solution space (+{flow_diff:.3f})")
        else:
            analysis.append(f"**Flow Space**: Prompt 1 operates more in solution space (+{abs(flow_diff):.3f})")
    else:
        analysis.append(f"**Flow Space**: Both operate in similar regions (Δ={flow_diff:.3f})")
    analysis.append("")
    
    # Communication style
    if abs(q_ratio_1 - q_ratio_2) > 0.1:
        if q_ratio_1 > q_ratio_2:
            analysis.append(f"**Communication Style**: Prompt 1 is more exploratory/questioning ({q_ratio_1:.1%} vs {q_ratio_2:.1%})")
        else:
            analysis.append(f"**Communication Style**: Prompt 2 is more exploratory/questioning ({q_ratio_2:.1%} vs {q_ratio_1:.1%})")
    else:
        analysis.append(f"**Communication Style**: Similar balance of questions and assertions")
    analysis.append("")
    
    # 5. Decision Preferences
    analysis.append("=" * 80)
    analysis.append("### 5. REVEALED DECISION PREFERENCES")
    analysis.append("Based on the differences, what the system 'prefers' for each prompt:")
    analysis.append("=" * 80)
    analysis.append("")
    
    analysis.append(f"**For Prompt 1** ({prompt1[:60]}...):")
    analysis.append("")
    
    # Preference 1: Role emphasis
    if metrics1.role_token_counts.get("Problem Poser", 0) > metrics2.role_token_counts.get("Problem Poser", 0):
        analysis.append(f"  ✓ PREFERS: Deeper problem exploration (Problem Poser more active)")
    elif metrics1.role_token_counts.get("Synthesizer", 0) > metrics2.role_token_counts.get("Synthesizer", 0):
        analysis.append(f"  ✓ PREFERS: Integration and synthesis (Synthesizer more active)")
    
    # Preference 2: Flow position
    if metrics1.avg_aesthetic_flow < 0.4:
        analysis.append(f"  ✓ PREFERS: Staying in problem space (flow: {metrics1.avg_aesthetic_flow:.2f})")
    elif metrics1.avg_aesthetic_flow > 0.6:
        analysis.append(f"  ✓ PREFERS: Moving toward solutions (flow: {metrics1.avg_aesthetic_flow:.2f})")
    else:
        analysis.append(f"  ✓ PREFERS: Balanced exploration (flow: {metrics1.avg_aesthetic_flow:.2f})")
    
    # Preference 3: Communication style
    if q_ratio_1 > 0.3:
        analysis.append(f"  ✓ PREFERS: Interrogative approach ({metrics1.question_count} questions)")
    else:
        analysis.append(f"  ✓ PREFERS: Declarative approach ({metrics1.assertion_count} assertions)")
    
    # Preference 4: Complexity
    if avg_sent_1 > avg_sent_2 + 2:
        analysis.append(f"  ✓ PREFERS: Complex, detailed sentences (avg: {avg_sent_1:.1f} tokens)")
    elif avg_sent_1 < avg_sent_2 - 2:
        analysis.append(f"  ✓ PREFERS: Concise, focused sentences (avg: {avg_sent_1:.1f} tokens)")
    
    analysis.append("")
    analysis.append(f"**For Prompt 2** ({prompt2[:60]}...):")
    analysis.append("")
    
    # Preference 1: Role emphasis
    if metrics2.role_token_counts.get("Problem Poser", 0) > metrics1.role_token_counts.get("Problem Poser", 0):
        analysis.append(f"  ✓ PREFERS: Deeper problem exploration (Problem Poser more active)")
    elif metrics2.role_token_counts.get("Synthesizer", 0) > metrics1.role_token_counts.get("Synthesizer", 0):
        analysis.append(f"  ✓ PREFERS: Integration and synthesis (Synthesizer more active)")
    
    # Preference 2: Flow position
    if metrics2.avg_aesthetic_flow < 0.4:
        analysis.append(f"  ✓ PREFERS: Staying in problem space (flow: {metrics2.avg_aesthetic_flow:.2f})")
    elif metrics2.avg_aesthetic_flow > 0.6:
        analysis.append(f"  ✓ PREFERS: Moving toward solutions (flow: {metrics2.avg_aesthetic_flow:.2f})")
    else:
        analysis.append(f"  ✓ PREFERS: Balanced exploration (flow: {metrics2.avg_aesthetic_flow:.2f})")
    
    # Preference 3: Communication style
    if q_ratio_2 > 0.3:
        analysis.append(f"  ✓ PREFERS: Interrogative approach ({metrics2.question_count} questions)")
    else:
        analysis.append(f"  ✓ PREFERS: Declarative approach ({metrics2.assertion_count} assertions)")
    
    # Preference 4: Complexity
    if avg_sent_2 > avg_sent_1 + 2:
        analysis.append(f"  ✓ PREFERS: Complex, detailed sentences (avg: {avg_sent_2:.1f} tokens)")
    elif avg_sent_2 < avg_sent_1 - 2:
        analysis.append(f"  ✓ PREFERS: Concise, focused sentences (avg: {avg_sent_2:.1f} tokens)")
    
    analysis.append("")
    analysis.append("=" * 80)
    
    # 6. Conclusion
    analysis.append("")
    analysis.append("### DECISION THEORY INSIGHT")
    analysis.append("")
    analysis.append("The system's 'decisions' emerge from the interaction between:")
    analysis.append("  1. Prompt semantics (what the question asks)")
    analysis.append("  2. Flow space dynamics (problem→solution gradient)")
    analysis.append("  3. Role-specific biases (each role's preferred flow position)")
    analysis.append("  4. Corpus knowledge (available linguistic patterns)")
    analysis.append("")
    analysis.append("Different prompts activate different regions of this decision space,")
    analysis.append("revealing what the neurosymbolic system 'wants to do' for each context.")
    analysis.append("=" * 80)
    
    return "\n".join(analysis)


# ----------------------------
# GRADIO UI
# ----------------------------

def _resolve_gradio_file_to_path(infile) -> str:
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


def _load_corpus(use_hf: bool, hf_dataset: str, hf_split: str, hf_max_rows: int, text_file) -> str:
    if use_hf:
        ds = load_dataset(hf_dataset, split=hf_split)
        rows = int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds)
        rows = min(rows, len(ds))
        if "text" in ds.column_names:
            return "\n".join(str(x) for x in ds.select(range(rows))["text"])
        return "\n".join(str(ds[i]) for i in range(rows))
    else:
        # Load from text file
        if text_file is None:
            raise ValueError("No text file provided. Please upload a .txt or .md file.")
        path = _resolve_gradio_file_to_path(text_file)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() not in [".txt", ".md"]:
            raise ValueError(f"Unsupported file type: {p.suffix}. Please use .txt or .md files.")
        return p.read_text(encoding="utf-8", errors="replace")


def create_visual_diff_html(text1: str, text2: str, prompt1: str, prompt2: str) -> str:
    """Create HTML visualization of text differences"""
    
    text_diff = extract_text_differences(text1, text2)
    
    # Create unique word sets for highlighting
    unique_words_1 = {word for word, _ in text_diff['unique_to_1']}
    unique_words_2 = {word for word, _ in text_diff['unique_to_2']}
    
    def highlight_text(text: str, unique_words: set, color: str) -> str:
        """Highlight unique words in text"""
        # Extract content only
        lines = text.split('\n')
        content_parts = []
        
        for line in lines:
            if '=' * 30 in line or 'MULTI-ENTITY' in line or 'END OF SESSION' in line:
                continue
            
            # Handle role markers
            if line.startswith('[') and ']' in line[:30]:
                parts = line.split(']', 1)
                if len(parts) > 1:
                    role_part = parts[0] + ']'
                    text_part = parts[1]
                    content_parts.append(f'<div style="margin: 10px 0;"><strong>{role_part}</strong>{text_part}</div>')
            elif line.strip():
                content_parts.append(f'<div style="margin: 5px 0;">{line}</div>')
        
        full_html = ''.join(content_parts)
        
        # Highlight unique words
        for word in unique_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{word}</span>'
            full_html = re.sub(pattern, replacement, full_html, flags=re.IGNORECASE)
        
        return full_html
    
    # Create side-by-side comparison
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px;">
        <h2 style="text-align: center; color: #2c3e50;">📊 Text Difference Visualization</h2>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h3 style="margin-top: 0;">Summary Statistics</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div>
                    <strong>Lexical Similarity:</strong> {text_diff['similarity']:.1%}<br>
                    <strong>Shared Words:</strong> {text_diff['common_count']}
                </div>
                <div>
                    <strong>Unique to Session 1:</strong> {len(text_diff['unique_to_1'])} words<br>
                    <strong>Unique to Session 2:</strong> {len(text_diff['unique_to_2'])} words
                </div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="border: 2px solid #ffcc80; border-radius: 8px; padding: 15px; background: #fff3e0;">
                <h3 style="color: #e65100; margin-top: 0;">🔴 Top Unique Words - Session 1</h3>
                <p style="font-size: 12px; color: #666; margin-bottom: 10px;">
                    Prompt: <em>{prompt1[:70]}...</em>
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {"".join(f'<span style="background: #ffcc80; padding: 4px 10px; border-radius: 15px; font-size: 13px;">{word} ({count})</span>' for word, count in text_diff['unique_to_1'][:20])}
                </div>
            </div>
            
            <div style="border: 2px solid #90caf9; border-radius: 8px; padding: 15px; background: #e3f2fd;">
                <h3 style="color: #1565c0; margin-top: 0;">🔵 Top Unique Words - Session 2</h3>
                <p style="font-size: 12px; color: #666; margin-bottom: 10px;">
                    Prompt: <em>{prompt2[:70]}...</em>
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {"".join(f'<span style="background: #90caf9; padding: 4px 10px; border-radius: 15px; font-size: 13px;">{word} ({count})</span>' for word, count in text_diff['unique_to_2'][:20])}
                </div>
            </div>
        </div>
        
        <div style="margin: 30px 0;">
            <h3>📝 Distinctive Phrases</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: #e65100;">Session 1 Only:</h4>
                    <ul style="list-style: none; padding: 0;">
                        {"".join(f'<li style="padding: 5px; margin: 5px 0; background: #fff3e0; border-left: 3px solid #ffcc80;">"{phrase}"</li>' for phrase in text_diff['unique_phrases_1'][:8])}
                    </ul>
                </div>
                <div>
                    <h4 style="color: #1565c0;">Session 2 Only:</h4>
                    <ul style="list-style: none; padding: 0;">
                        {"".join(f'<li style="padding: 5px; margin: 5px 0; background: #e3f2fd; border-left: 3px solid #90caf9;">"{phrase}"</li>' for phrase in text_diff['unique_phrases_2'][:8])}
                    </ul>
                </div>
            </div>
        </div>
        
        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 20px 0; text-align: center;">
            <strong>Legend:</strong> 
            <span style="background: #ffcc80; padding: 2px 8px; margin: 0 10px; border-radius: 3px;">Session 1 Unique</span>
            <span style="background: #90caf9; padding: 2px 8px; margin: 0 10px; border-radius: 3px;">Session 2 Unique</span>
        </div>
    </div>
    """
    
    return html


def run_decision_analysis(use_hf, hf_dataset, hf_split, hf_max_rows, text_file,
                          prompt1, prompt2, seed, max_tokens, num_speakers,
                          steer, focus, pairwise, progress=gr.Progress()):
    """Generate two sessions and analyze decision preferences"""
    
    try:
        progress(0.0, desc="Loading corpus...")
        corpus_text = _load_corpus(bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows), text_file)
    except Exception as e:
        return f"Error: {e}", "", "", ""
    
    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        pairwise_strength=float(pairwise),
        activator_boot_epochs=15,
        hemi_enable=True,
    )
    
    progress(0.1, desc="Preparing corpus...")
    prep = gen.prepare_corpus(corpus_text, progress=progress)
    
    progress(0.4, desc="Generating session 1...")
    text1, metrics1 = gen.generate(
        prep, 
        prompt=str(prompt1), 
        start_x=0.0, 
        max_tokens=int(max_tokens),
        seed=int(seed), 
        num_speakers=int(num_speakers), 
        tokens_per_turn=60,
        problem_solving_mode=True
    )
    
    progress(0.7, desc="Generating session 2...")
    # Reset hemicontinuity for second generation
    gen.hemi.reset()
    
    text2, metrics2 = gen.generate(
        prep, 
        prompt=str(prompt2), 
        start_x=0.0, 
        max_tokens=int(max_tokens),
        seed=int(seed), 
        num_speakers=int(num_speakers), 
        tokens_per_turn=60,
        problem_solving_mode=True
    )
    
    progress(0.95, desc="Analyzing decisions...")
    analysis = analyze_decision_preferences(metrics1, metrics2, str(prompt1), str(prompt2), text1, text2)
    
    # Create visual HTML diff
    visual_diff = create_visual_diff_html(text1, text2, str(prompt1), str(prompt2))
    
    return text1, text2, analysis, visual_diff


def build_app():
    with gr.Blocks(title="Decision Theory Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧠 NeuroSymbolic Decision Theory Analyzer\n\n"
            "Generate two different problem-solving sessions, then analyze what the system "
            "'prefers' to do for each prompt based on the differences in reasoning patterns."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Corpus Settings")
                
                use_hf = gr.Checkbox(
                    label="Use Hugging Face dataset", 
                    value=True,
                )
                
                # HuggingFace options
                hf_dataset = gr.Textbox(
                    label="HF Dataset", 
                    value="AiresPucrs/stanford-encyclopedia-philosophy",
                    visible=True
                )
                hf_split = gr.Textbox(label="Split", value="train", visible=True)
                hf_max_rows = gr.Slider(100, 5000, value=2000, step=100, label="Max rows", visible=True)
                
                # Text file upload option
                text_file = gr.File(
                    label="Upload Text File (.txt or .md)",
                    file_types=[".txt", ".md"],
                    visible=False,
                )
                file_info = gr.Markdown(
                    "📝 **Supported formats**: .txt, .md\n\n"
                    "The entire file will be used as the corpus.",
                    visible=False
                )
                
                # Toggle visibility based on use_hf checkbox
                def toggle_corpus_source(use_hf_val):
                    return (
                        gr.update(visible=use_hf_val),  # hf_dataset
                        gr.update(visible=use_hf_val),  # hf_split
                        gr.update(visible=use_hf_val),  # hf_max_rows
                        gr.update(visible=not use_hf_val),  # text_file
                        gr.update(visible=not use_hf_val),  # file_info
                    )
                
                use_hf.change(
                    toggle_corpus_source,
                    inputs=[use_hf],
                    outputs=[hf_dataset, hf_split, hf_max_rows, text_file, file_info]
                )
                
                gr.Markdown("### Generation Parameters")
                seed = gr.Number(value=42, label="Seed")
                max_tokens = gr.Slider(100, 600, value=300, step=50, label="Tokens per session")
                num_speakers = gr.Slider(2, 5, value=5, step=1, label="Number of roles")
                
                gr.Markdown("### Neural Parameters")
                steer = gr.Slider(0.5, 3, value=1.35, step=0.05, label="Steer strength")
                focus = gr.Slider(0, 1, value=0.5, step=0.05, label="Focus strength")
                pairwise = gr.Slider(0, 2, value=0.6, step=0.1, label="Pairwise strength")
                
            with gr.Column(scale=2):
                gr.Markdown("### Two Prompts for Comparison")
                
                prompt1 = gr.Textbox(
                    label="Prompt 1", 
                    value="What is the nature of consciousness?",
                    lines=2,
                    placeholder="Enter first question or problem..."
                )
                
                prompt2 = gr.Textbox(
                    label="Prompt 2",
                    value="How can we solve the mind-body problem?",
                    lines=2,
                    placeholder="Enter second question or problem..."
                )
                
                btn = gr.Button("🔬 Generate & Analyze Decisions", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                with gr.Tab("📊 Decision Analysis"):
                    analysis_output = gr.Textbox(
                        label="Comparative Decision Analysis",
                        lines=35,
                        max_lines=50
                    )
                
                with gr.Tab("🔤 Text Differences"):
                    gr.Markdown(
                        "### Vocabulary Comparison\n"
                        "Shows words and phrases that appear in one session but not the other."
                    )
                    diff_display = gr.HTML(label="Visual Difference View")
                
                with gr.Tab("📝 Session 1"):
                    output1 = gr.Textbox(label="Problem-Solving Session 1", lines=25)
                
                with gr.Tab("📝 Session 2"):
                    output2 = gr.Textbox(label="Problem-Solving Session 2", lines=25)
        
        btn.click(
            run_decision_analysis,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows, text_file,
                   prompt1, prompt2, seed, max_tokens, num_speakers,
                   steer, focus, pairwise],
            outputs=[output1, output2, analysis_output, diff_display]
        )
        
        gr.Markdown(
            "---\n"
            "### How It Works\n\n"
            "1. **Generate**: Creates two problem-solving sessions from different prompts\n"
            "2. **Compare**: Analyzes role distribution, flow positions, and communication patterns\n"
            "3. **Reveal**: Shows what the system 'prefers' to do for each prompt\n\n"
            "**Decision preferences emerge from**: prompt semantics, flow space dynamics, "
            "role-specific biases, and corpus knowledge patterns.\n\n"
            "### Corpus Options\n\n"
            "✅ **HuggingFace Datasets**: Use pre-loaded datasets (philosophy, wikipedia, etc.)\n\n"
            "📁 **Text File Upload**: Upload your own .txt or .md file as the corpus"
        )
        
        return demo


if __name__ == "__main__":
    build_app().queue().launch()
