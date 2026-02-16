#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuroSymbolic V7.0 - Professional Narrative Generation WITH CENTROID OF MEANING
New Features:
- Semantic centroid computed for each token in vocabulary
- Centroid-aware generation: tokens near current centroid are boosted
- Coherence tracking: measures semantic tightness
- Multi-scale semantics: local (pairwise) + global (centroid)
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass
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
# CENTROID OF MEANING SYSTEM
# ----------------------------

@dataclass
class SemanticCentroid:
    """Represents the centroid of meaning for a token in aesthetic space"""
    token: str
    centroid_vector: np.ndarray  # 4D: [harmony, density, momentum, resonance]
    neighbor_tokens: List[str]
    coherence: float  # How tightly clustered (0-1)
    
    def distance_to(self, other: "SemanticCentroid") -> float:
        return float(np.linalg.norm(self.centroid_vector - other.centroid_vector))
    
    def similarity_to(self, other: "SemanticCentroid") -> float:
        """Exponential decay: closer centroids = higher similarity"""
        return math.exp(-self.distance_to(other))


class CentroidComputer:
    """Computes and manages semantic centroids for tokens"""
    
    def __init__(self, max_neighbors: int = 50):
        self.max_neighbors = max_neighbors
        self.centroids: Dict[str, SemanticCentroid] = {}
    
    def compute_centroid(self, token: str, vocabulary: List[str], 
                        freq_dict: Optional[Dict[str, int]] = None) -> SemanticCentroid:
        """
        Compute semantic centroid for a token based on pairwise relationships.
        
        Process:
        1. Create SymbolicPair with each vocab token
        2. Extract 4D aesthetic vectors
        3. Weight by token frequency
        4. Take top-k most harmonious neighbors
        5. Compute mean position = centroid
        6. Measure coherence (cluster tightness)
        """
        from __main__ import SymbolicPair  # Import from current module
        
        pairs = []
        for other_token in vocabulary:
            if other_token == token:
                continue
            
            pair = SymbolicPair.from_tokens(token, other_token)
            aesthetic_vec = pair.aesthetic_vector()
            
            # Weight by frequency
            weight = 1.0
            if freq_dict is not None:
                freq = freq_dict.get(other_token, 1)
                weight = math.log(1 + freq)
            
            pairs.append((pair, weight, other_token, aesthetic_vec))
        
        # Sort by harmony (most similar first)
        pairs.sort(key=lambda x: x[0].harmony, reverse=True)
        
        # Take top N neighbors
        top_pairs = pairs[:self.max_neighbors]
        neighbor_tokens = [tok for (_, _, tok, _) in top_pairs]
        weighted_vectors = [vec * weight for (_, weight, _, vec) in top_pairs]
        
        # Compute centroid
        if len(weighted_vectors) > 0:
            vectors_array = np.stack(weighted_vectors)
            centroid_vec = np.mean(vectors_array, axis=0)
            
            # Coherence = inverse of neighbor spread
            distances = [np.linalg.norm(v - centroid_vec) for v in weighted_vectors]
            coherence = 1.0 - (np.mean(distances) / (np.max(distances) + 1e-12))
            coherence = float(np.clip(coherence, 0.0, 1.0))
        else:
            centroid_vec = np.zeros(4, dtype=np.float32)
            coherence = 0.0
        
        centroid = SemanticCentroid(
            token=token,
            centroid_vector=centroid_vec,
            neighbor_tokens=neighbor_tokens,
            coherence=coherence
        )
        
        self.centroids[token] = centroid
        return centroid
    
    def compute_all_centroids(self, tokens: List[str], 
                             freq_dict: Optional[Dict[str, int]] = None,
                             progress=None) -> Dict[str, SemanticCentroid]:
        """Compute centroids for all unique tokens"""
        unique_tokens = list(set(tokens))
        
        if progress:
            progress(0.15, desc=f"Computing centroids for {len(unique_tokens)} tokens...")
        
        for i, token in enumerate(unique_tokens):
            self.compute_centroid(token, unique_tokens, freq_dict=freq_dict)
            
            if progress and i % 100 == 0 and i > 0:
                pct = 0.15 + 0.05 * (i / len(unique_tokens))
                progress(pct, desc=f"Centroids: {i}/{len(unique_tokens)}")
        
        return self.centroids
    
    def get_centroid_boost(self, current_token: str, candidate_tokens: List[str],
                          boost_strength: float = 0.3) -> np.ndarray:
        """
        Compute centroid-based boost for candidate tokens.
        Tokens with centroids near the current token's centroid get boosted.
        
        This creates "semantic gravity" that pulls generation toward
        related concepts in the 4D aesthetic space.
        """
        if current_token not in self.centroids:
            return np.zeros(len(candidate_tokens), dtype=np.float32)
        
        current_centroid = self.centroids[current_token]
        boosts = []
        
        for candidate in candidate_tokens:
            if candidate in self.centroids:
                # Similarity decays exponentially with distance
                similarity = current_centroid.similarity_to(self.centroids[candidate])
                boost = boost_strength * similarity
            else:
                boost = 0.0
            boosts.append(boost)
        
        return np.array(boosts, dtype=np.float32)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about centroids"""
        if not self.centroids:
            return {}
        
        coherences = [c.coherence for c in self.centroids.values()]
        return {
            "total_centroids": len(self.centroids),
            "avg_coherence": np.mean(coherences),
            "min_coherence": np.min(coherences),
            "max_coherence": np.max(coherences),
            "high_coherence_tokens": [
                t for t, c in self.centroids.items() if c.coherence > 0.7
            ][:10]
        }


# ----------------------------
# CORE SYMBOLIC ENGINE
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
# ACTIVATOR NETWORK
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

        vecs = [SymbolicPair.from_tokens(a, b).aesthetic_vector() for a, b, _ in pairs]
        y = torch.tensor(np.array(vecs), dtype=torch.float32)
        
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
# CONSTANTS & UTILS
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
# FLOW & GRAPH
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
# FUZZY LOGIC
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
                osculator_strength: float = 0.0,
                trigram_positions: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
        e = entropy01.clamp(0, 1)
        p = peak01.clamp(0, 1)
        b = boost01.clamp(0, 1)
        a = aesthetic_flow01.clamp(0, 1)

        # Use trigram positions if provided, otherwise use default [0.0, 0.5, 1.0]
        if trigram_positions is not None:
            pos1, pos2, pos3 = trigram_positions
        else:
            pos1, pos2, pos3 = 0.0, 0.5, 1.0
        
        # Convert positions to tensors
        pos1_t = torch.tensor(pos1, device=e.device, dtype=torch.float32)
        pos2_t = torch.tensor(pos2, device=e.device, dtype=torch.float32)
        pos3_t = torch.tensor(pos3, device=e.device, dtype=torch.float32)

        # Position-indexed membership functions for entropy (e)
        # e_low uses pos1 to shift boundaries
        e_low  = mf_trap(e, 0.0 * (1.0 - pos1_t) + 0.1 * pos1_t, 
                            0.0 * (1.0 - pos1_t) + 0.15 * pos1_t, 
                            0.25 + 0.1 * pos1_t, 
                            0.45 + 0.05 * pos1_t)
        # e_mid uses pos2
        e_mid  = mf_tri(e, 0.25 - 0.05 * pos2_t, 
                           0.50, 
                           0.75 + 0.05 * pos2_t)
        # e_high uses pos3
        e_high = mf_trap(e, 0.55 - 0.1 * pos3_t, 
                            0.75 - 0.05 * pos3_t, 
                            1.0, 
                            1.0)
        
        # Position-indexed membership functions for peak (p)
        p_low  = mf_trap(p, 0.0, 
                            0.0, 
                            0.20 + 0.1 * pos1_t, 
                            0.40 + 0.05 * pos1_t)
        p_mid  = mf_tri(p, 0.25 - 0.1 * pos2_t, 
                           0.50, 
                           0.75 + 0.1 * pos2_t)
        p_high = mf_trap(p, 0.60 - 0.1 * pos3_t, 
                            0.80 - 0.05 * pos3_t, 
                            1.0, 
                            1.0)
        
        # Position-indexed membership functions for boost (b)
        b_low  = mf_trap(b, 0.0, 
                            0.0, 
                            0.20 + 0.15 * pos1_t, 
                            0.45 + 0.1 * pos1_t)
        b_mid  = mf_tri(b, 0.25 - 0.1 * pos2_t, 
                           0.50 + 0.05 * pos2_t, 
                           0.75 + 0.05 * pos2_t)
        b_high = mf_trap(b, 0.55 - 0.15 * pos3_t, 
                            0.80 - 0.1 * pos3_t, 
                            1.0, 
                            1.0)
        
        # Position-indexed membership for aesthetic flow (a)
        # Uses average position of trigram
        avg_pos = (pos1_t + pos2_t + pos3_t) / 3.0
        a_high = mf_trap(a, 0.5 - 0.1 * avg_pos, 
                            0.7 - 0.05 * avg_pos, 
                            1.0, 
                            1.0)

        # Fuzzy rules (unchanged)
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
# LANGUAGE MODEL
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
# DATA STRUCTURES & CACHE
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
    centroid_computer: CentroidComputer  # NEW: Added centroid system

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

@dataclass
class GenerationMetrics:
    role_token_counts: Dict[str, int]
    role_avg_flow: Dict[str, float]
    keyword_usage: Counter
    sentence_lengths: List[int]
    question_count: int
    assertion_count: int
    avg_aesthetic_flow: float
    total_tokens: int
    abstract_density: float = 0.0
    conceptual_jumps: int = 0
    unique_combinations: int = 0
    centroid_stats: Dict[str, Any] = None  # NEW: Centroid statistics


# ----------------------------
# MAIN GENERATOR
# ----------------------------

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
                 hemi_smooth_alpha: float = 0.05,
                 centroid_boost_strength: float = 0.35):  # NEW PARAMETER
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
        self.centroid_boost_strength = float(centroid_boost_strength)  # NEW

    def _pick_initial_context(self, lm: QuadgramLM, seed_words: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_'-]*$", t) and t not in COGNITIVE_TOKENS]
        if len(sw) >= 3: return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2: return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1: return (sw[-1], sw[-1], sw[-1])
        seed_tok = lm.vocab[0] if lm.vocab else "the"
        return (seed_tok, seed_tok, seed_tok)

    def build_state(self, text: str, progress=None) -> ModelState:
        tokens = basic_tokenize(text)
        flow_by_token = compute_problem_flow_by_token(tokens)
        lm = QuadgramLM(add_k=self.lm_add_k)
        lm.ingest(tokens)
        
        # NEW: Compute semantic centroids for vocabulary
        if progress:
            progress(0.1, desc="Computing semantic centroids...")
        centroid_computer = CentroidComputer(max_neighbors=50)
        centroid_computer.compute_all_centroids(tokens, freq_dict=lm.uni, progress=progress)
        
        if progress:
            progress(0.2, desc="Building token boosts...")
        
        token_boost: Dict[str, float] = {}
        for tok, boost_val in COGNITIVE_TOKENS.items():
            token_boost[tok] = boost_val
        activator = NeuronalActivator()
        if torch.cuda.is_available(): activator = activator.cuda()
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
                terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))], key=lambda x: -abs(x[1]))[:10]
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
            centroid_computer=centroid_computer,  # NEW
        )

    def prepare_corpus(self, raw_text: str, progress=None) -> PreparedCorpus:
        text = inject_cognitive_tokens(raw_text)
        text = normalize(text)
        state = self.build_state(text, progress=progress)
        tokens = basic_tokenize(text)
        if self.hemi_enable: self.hemi.reset()
        return PreparedCorpus(text=text, tokens=tokens, state=state)

    def _compute_abstract_coinage_boost(self, cand: List[str], w1: str, w2: str, w3: str, 
                                       prep: PreparedCorpus, creativity: float) -> torch.Tensor:
        """Boost tokens based on sophistication and natural novelty"""
        device = prep.state.activator.emb.weight.device
        boost = torch.zeros(len(cand), dtype=torch.float32, device=device)
        
        # Moderate complexity boost for natural sophistication
        for i, word in enumerate(cand):
            # Favor moderately complex words (natural vocabulary)
            if 5 <= len(word) <= 9:
                boost[i] += 0.2 * creativity
            elif len(word) >= 10:
                boost[i] += 0.15 * creativity  # Don't over-favor extremely long words
            
            # Modest rarity boost (prefer less common but not obscure)
            freq = prep.state.lm.uni.get(word, 0)
            if freq > 0 and prep.state.lm.total > 0:
                normalized_freq = freq / prep.state.lm.total
                # Sweet spot: not too common, not too rare
                if 0.0001 < normalized_freq < 0.01:
                    rarity_score = 1.0 - (normalized_freq / 0.01)
                    boost[i] += 0.25 * rarity_score * creativity
                elif normalized_freq <= 0.0001:
                    boost[i] += 0.15 * creativity  # Very rare words get moderate boost
            
            # Encourage natural semantic connections (moderate novelty)
            pair = SymbolicPair.from_tokens(w3, word)
            # Sweet spot: related but not too similar
            if 0.3 < pair.harmony < 0.7:
                novelty = abs(pair.harmony - 0.5) / 0.2  # Peak at 0.5
                boost[i] += 0.2 * (1.0 - novelty) * creativity
        
        return boost

    def _compute_trigram_positions(self, w1: str, w2: str, w3: str, 
                                   prep: PreparedCorpus) -> Tuple[float, float, float]:
        """Compute normalized position indexes for trigram based on vocab frequency"""
        vocab_size = len(prep.state.lm.vocab)
        if vocab_size == 0:
            return (0.0, 0.5, 1.0)
        
        # Get frequency ranks (lower rank = more common)
        def get_position(word: str) -> float:
            freq = prep.state.lm.uni.get(word, 0)
            if freq == 0:
                return 1.0  # Unknown words get highest position
            # Sort all words by frequency to get rank
            sorted_words = sorted(prep.state.lm.uni.items(), key=lambda x: -x[1])
            rank = next((i for i, (w, _) in enumerate(sorted_words) if w == word), vocab_size)
            # Normalize to [0, 1]
            return rank / max(1, vocab_size)
        
        pos1 = get_position(w1)
        pos2 = get_position(w2)
        pos3 = get_position(w3)
        
        return (pos1, pos2, pos3)

    def _final_probs(self, prep: PreparedCorpus, w1: str, w2: str, w3: str,
                     x_pos: torch.Tensor, allow_cache: bool = True,
                     abstract_coinage_mode: bool = False,
                     coinage_creativity: float = 0.8) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        cache_ok = allow_cache and (not x_pos.requires_grad) and (x_pos.numel() == 1) and (not abstract_coinage_mode)
        key = None
        if cache_ok:
            key = (self.cache_version, w1, w2, w3, float(x_pos.item()))
            cached = self.cache.get(key)
            if cached is not None: return cached

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
        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand], dtype=torch.float32, device=device)

        # Pairwise activator boost
        w_pair, _vec4 = prep.state.activator.forward_weight(w3, cand, x_pos=x_pos)
        w_pair = w_pair / (w_pair.mean() + 1e-12)
        boosts = boosts + self.pairwise_strength * w_pair

        # NEW: CENTROID BOOST - Pull toward semantically related tokens
        centroid_boost = prep.state.centroid_computer.get_centroid_boost(
            current_token=w3,
            candidate_tokens=cand,
            boost_strength=self.centroid_boost_strength
        )
        centroid_boost_t = torch.tensor(centroid_boost, dtype=torch.float32, device=device)
        boosts = boosts + centroid_boost_t  # Add centroid semantic gravity

        # NARRATIVE MODE MODIFICATIONS
        if abstract_coinage_mode:
            # Add refined vocabulary boost
            creative_boost = self._compute_abstract_coinage_boost(
                cand, w1, w2, w3, prep, coinage_creativity
            )
            boosts = boosts + creative_boost
            
            # Moderate temperature increase for natural variety
            effective_temp_mult = 1.2
        else:
            effective_temp_mult = 1.0

        context_str = f"{w1} {w2} {w3}"
        gravity = 0.0
        if "[PROBLEM]" in context_str: gravity = 0.2
        if "[SOLUTION]" in context_str: gravity = 0.3
        flow_w3 = float(prep.state.problem_flow_by_token.get(w3, 0.0))
        flow_w3_t = torch.tensor(flow_w3, dtype=torch.float32, device=device)
        boost01 = torch.tanh((boosts.abs().mean() + gravity + flow_w3_t) / 3.0).clamp(0.0, 1.0)
        base_flow = float(prep.state.semantic_graph.get_aesthetic_flow())
        base_flow_t = torch.tensor(base_flow, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        aesthetic_flow01 = (base_flow_t * (0.5 + 0.5 * x_pos)).clamp(0.0, 1.0)

        # Compute trigram position indexes
        trigram_positions = self._compute_trigram_positions(w1, w2, w3, prep)

        g = self.fuzzy_ctl(entropy01, peak01, boost01, aesthetic_flow01, 
                          osculator_strength=self.osculator_strength,
                          trigram_positions=trigram_positions)
        effective_steer = self.base_steer * g
        effective_temp = self.base_temp * (1.2 - 0.7 * g) * effective_temp_mult

        potentials = torch.log(base_p.to(device=device).clamp_min(1e-12)) + effective_steer * boosts
        potentials = potentials / torch.clamp(effective_temp, min=1e-6)
        final_probs = F.softmax(potentials, dim=-1)

        hemi_pen = torch.tensor(0.0, device=device, dtype=final_probs.dtype)
        if self.hemi_enable and not abstract_coinage_mode:  # Disable hemi for more creativity
            final_probs, hemi_pen = self.hemi.apply(cand, final_probs)
        
        flow_vec = torch.tensor([prep.state.problem_flow_by_token.get(w, 0.0) for w in cand], dtype=torch.float32, device=device).clamp(0.0, 1.0)

        out = (cand, final_probs, flow_vec)
        if cache_ok and key is not None: self.cache.put(key, (cand, final_probs.detach(), flow_vec.detach()))
        return out

    @torch.no_grad()
    def generate(self, prep: PreparedCorpus, prompt: str, start_x: float,
                 max_tokens: int = 220, seed: int = 42, num_speakers: int = 2,
                 tokens_per_turn: int = 50, problem_solving_mode: bool = True,
                 target_flow: Optional[float] = None,
                 abstract_coinage_mode: bool = False,
                 coinage_creativity: float = 0.7) -> Tuple[str, GenerationMetrics]:
        rng = np.random.default_rng(int(seed))
        seed_toks = basic_tokenize(prompt)
        w1, w2, w3 = self._pick_initial_context(prep.state.lm, seed_toks)
        device = prep.state.activator.emb.weight.device
        total_steps = int(max_tokens)
        
        if abstract_coinage_mode:
            # Professional narrative voices with natural flow
            role_definitions = [
                ("Observer", 0.0, 0.2, "noticing", ["observe", "notice", "perceive", "see", "consider"]),
                ("Questioner", 0.15, 0.35, "inquiring", ["perhaps", "might", "could", "what", "whether"]),
                ("Connector", 0.3, 0.5, "relating", ["between", "connects", "relates", "links", "bridges"]),
                ("Elaborator", 0.45, 0.65, "developing", ["further", "moreover", "building", "develops", "extends"]),
                ("Synthesizer", 0.6, 0.8, "integrating", ["together", "combines", "integrates", "unifies", "converges"]),
                ("Reflector", 0.75, 1.0, "concluding", ["ultimately", "thus", "suggests", "reveals", "illuminates"]),
            ]
        else:
            # Original problem-solving roles
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
        
        conversation: List[Tuple[str, str, List[str], float]] = []
        current_speaker_idx = 0
        current_turn_tokens: List[str] = []
        alpha_count_turn = 0
        
        role_token_counts = {role[0]: 0 for role in speaker_roles}
        role_flow_sums = {role[0]: 0.0 for role in speaker_roles}
        role_flow_counts = {role[0]: 0 for role in speaker_roles}
        keyword_usage = Counter()
        sentence_lengths = []
        question_count = 0
        assertion_count = 0
        aesthetic_flows = []
        current_sentence_length = 0
        
        # Abstract coinage tracking
        prev_tokens = [w1, w2, w3]
        conceptual_jumps = 0
        unique_pairs = set()

        for i in range(total_steps):
            role_name, x_min, x_max, mode, keywords = speaker_roles[current_speaker_idx]
            progress = i / max(1, total_steps)
            global_x = start_x + (1.0 - start_x) * progress
            role_x_bias = (x_min + x_max) / 2.0
            
            if target_flow is not None:
                curr_x_val = target_flow
            else:
                curr_x_val = 0.7 * global_x + 0.3 * role_x_bias
                curr_x_val = max(x_min, min(x_max, curr_x_val))
            
            x = torch.tensor(float(curr_x_val), dtype=torch.float32, device=device)
            cand, probs, flow_vec = self._final_probs(
                prep, w1, w2, w3, x_pos=x, allow_cache=not abstract_coinage_mode,
                abstract_coinage_mode=abstract_coinage_mode,
                coinage_creativity=coinage_creativity
            )
            
            if keywords and not abstract_coinage_mode:
                keyword_boost = torch.zeros_like(probs)
                for idx, c in enumerate(cand):
                    if c.lower() in keywords:
                        keyword_boost[idx] = 0.3
                    flow_match = 1.0 - abs(flow_vec[idx].item() - role_x_bias)
                    keyword_boost[idx] += 0.2 * flow_match
                probs = probs * torch.exp(keyword_boost)
                probs = probs / (probs.sum() + 1e-12)
            
            # Sample from distribution
            p = probs.detach().cpu().numpy()
            p = p / (p.sum() + 1e-12)
            idx = rng.choice(len(cand), p=p)
            tok = cand[idx]

            # Track abstract metrics
            if abstract_coinage_mode:
                # Track conceptual jumps (semantic distance)
                if len(prev_tokens) >= 3:
                    pair = SymbolicPair.from_tokens(prev_tokens[-1], tok)
                    if pair.harmony < 0.4:  # Low harmony = conceptual jump
                        conceptual_jumps += 1
                unique_pairs.add((prev_tokens[-1] if prev_tokens else "", tok))
                prev_tokens.append(tok)
                if len(prev_tokens) > 5:
                    prev_tokens.pop(0)

            current_turn_tokens.append(tok)
            role_token_counts[role_name] += 1
            role_flow_sums[role_name] += flow_vec[idx].item()
            role_flow_counts[role_name] += 1
            aesthetic_flows.append(curr_x_val)
            if tok.lower() in keywords: keyword_usage[tok.lower()] += 1
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
            if re.match(r"[A-Za-z]", tok): alpha_count_turn += 1
            
            should_switch = False
            if tok in [".", "!", "?"] and alpha_count_turn >= min(tokens_per_turn * 0.6, 20): 
                should_switch = True
            elif len(current_turn_tokens) >= tokens_per_turn * 1.5: 
                should_switch = True
            elif len(current_turn_tokens) >= tokens_per_turn and alpha_count_turn > 15:
                if tok in [",", ";"] or (i > 0 and rng.random() < 0.3): 
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

        role_avg_flow = {role: role_flow_sums[role] / role_flow_counts[role] if role_flow_counts[role] > 0 else 0.0 for role in role_token_counts.keys()}
        
        # Calculate abstract density (avg word length, rarity)
        all_tokens = []
        for _, _, tokens, _ in conversation:
            all_tokens.extend(tokens)
        abstract_density = sum(len(t) for t in all_tokens if re.match(r"[A-Za-z]", t)) / max(1, len(all_tokens))
        
        # Get centroid statistics
        centroid_stats = prep.state.centroid_computer.get_summary_stats()
        
        metrics = GenerationMetrics(
            role_token_counts=role_token_counts,
            role_avg_flow=role_avg_flow,
            keyword_usage=keyword_usage,
            sentence_lengths=[s for s in sentence_lengths if s > 0],
            question_count=question_count,
            assertion_count=assertion_count,
            avg_aesthetic_flow=sum(aesthetic_flows)/len(aesthetic_flows) if aesthetic_flows else 0.0,
            total_tokens=total_steps,
            abstract_density=abstract_density,
            conceptual_jumps=conceptual_jumps,
            unique_combinations=len(unique_pairs),
            centroid_stats=centroid_stats  # NEW
        )
        
        text = self.format_conversation(conversation, problem_solving_mode, abstract_coinage_mode)
        return text, metrics

    def format_conversation(self, conversation: List[Tuple[str, str, List[str], float]], 
                          problem_solving_mode: bool, abstract_coinage_mode: bool = False) -> str:
        lines = []
        
        if abstract_coinage_mode:
            # Clean, professional narrative format
            lines.append("")
        elif problem_solving_mode:
            lines.append("=" * 60)
            lines.append("SINGLE-ENTITY PROBLEM SOLVING SESSION")
            lines.append("=" * 60)
            lines.append("")
        
        for role, mode, tokens, x_bias in conversation:
            text = detokenize(tokens)
            if text.strip():
                if abstract_coinage_mode:
                    # Simple, readable format
                    lines.append(f"**{role}**")
                    lines.append(f"{text}")
                    lines.append("")
                elif problem_solving_mode:
                    lines.append(f"[{role.upper()}] ({mode}, flow: {x_bias:.2f})")
                    lines.append(f"{text}")
                    lines.append("")
                else:
                    lines.append(f"{role}: {text}")
                    lines.append("")
        
        if abstract_coinage_mode:
            lines.append("")
        elif problem_solving_mode:
            lines.append("=" * 60)
            lines.append("END OF SESSION")
            lines.append("=" * 60)
        
        return "\n".join(lines)


# ----------------------------
# RUN & UI
# ----------------------------

def _resolve_gradio_file_to_path(infile) -> str:
    if infile is None: raise ValueError("No input file provided.")
    if isinstance(infile, str): return infile
    if hasattr(infile, "name") and isinstance(infile.name, str): return infile.name
    if isinstance(infile, dict) and "path" in infile: return str(infile["path"])
    if hasattr(infile, "path"): return str(infile.path)
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
        if text_file is None: raise ValueError("No text file provided.")
        path = _resolve_gradio_file_to_path(text_file)
        return load_text(path)

def run_session(use_hf, hf_dataset, hf_split, hf_max_rows, text_file, 
                prompt, seed, maxtokens, num_speakers, 
                steer, focus, pairwise, abstract_mode, creativity, centroid_strength, 
                progress=gr.Progress()):
    try:
        progress(0.0, desc="Loading corpus...")
        corpus_text = _load_corpus(bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows), text_file)
        
        gen = NeuroSymbolicGraphGenerator(
            steer_strength=float(steer), focus_strength=float(focus), 
            pairwise_strength=float(pairwise), activator_boot_epochs=15, 
            hemi_enable=not abstract_mode,  # Disable hemi for abstract mode
            centroid_boost_strength=float(centroid_strength)  # NEW PARAMETER
        )
        
        progress(0.05, desc="Preparing corpus...")
        prep = gen.prepare_corpus(corpus_text, progress=progress)
        
        progress(0.5, desc="Generating Session...")
        text, metrics = gen.generate(
            prep, str(prompt), start_x=0.0, max_tokens=int(maxtokens), 
            seed=int(seed), num_speakers=int(num_speakers), tokens_per_turn=60,
            problem_solving_mode=not abstract_mode,
            abstract_coinage_mode=bool(abstract_mode),
            coinage_creativity=float(creativity)
        )
        
        # Stats String
        stats = []
        stats.append(f"Total Tokens: {metrics.total_tokens}")
        stats.append(f"Average Flow: {metrics.avg_aesthetic_flow:.3f}")
        
        if abstract_mode:
            stats.append(f"Abstract Density: {metrics.abstract_density:.3f}")
            stats.append(f"Conceptual Jumps: {metrics.conceptual_jumps}")
            stats.append(f"Unique Combinations: {metrics.unique_combinations}")
        else:
            stats.append(f"Questions: {metrics.question_count}, Assertions: {metrics.assertion_count}")
        
        # NEW: Centroid statistics
        if metrics.centroid_stats:
            stats.append(f"\nCentroid Statistics:")
            stats.append(f"  Total centroids: {metrics.centroid_stats.get('total_centroids', 0)}")
            stats.append(f"  Avg coherence: {metrics.centroid_stats.get('avg_coherence', 0):.3f}")
            stats.append(f"  Coherence range: [{metrics.centroid_stats.get('min_coherence', 0):.3f}, {metrics.centroid_stats.get('max_coherence', 0):.3f}]")
            high_coh = metrics.centroid_stats.get('high_coherence_tokens', [])
            if high_coh:
                stats.append(f"  High coherence tokens: {', '.join(high_coh[:5])}")
        
        stats.append("\nRole Breakdown:")
        for r, c in metrics.role_token_counts.items():
            avg_flow = metrics.role_avg_flow.get(r, 0.0)
            stats.append(f"  {r}: {c} tokens (flow: {avg_flow:.3f})")
        
        if metrics.keyword_usage:
            stats.append("\nTop Keywords:")
            for kw, count in metrics.keyword_usage.most_common(5):
                stats.append(f"  {kw}: {count}")
        
        return text, "\n".join(stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", ""

def toggle_corpus_source(use_hf_val):
    return (
        gr.update(visible=use_hf_val), # hf_dataset
        gr.update(visible=use_hf_val), # hf_split
        gr.update(visible=use_hf_val), # hf_max_rows
        gr.update(visible=not use_hf_val), # text_file
        gr.update(visible=not use_hf_val), # file_info
    )

def toggle_mode(abstract_val):
    """Show/hide creativity slider based on abstract mode"""
    return gr.update(visible=abstract_val)

def build_app():
    with gr.Blocks(title="NeuroSymbolic V7.0 - Centroid of Meaning", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # NeuroSymbolic V7.0 - Narrative Generation
        Generate flowing narratives with sophisticated language and **semantic coherence**
        
        **NEW**: Each token has a computed semantic centroid that guides generation toward related concepts!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Corpus Source")
                use_hf = gr.Checkbox(label="Use Hugging Face dataset", value=True)
                hf_dataset = gr.Textbox(label="HF Dataset", value="AiresPucrs/stanford-encyclopedia-philosophy", visible=True)
                hf_split = gr.Textbox(label="Split", value="train", visible=True)
                hf_max_rows = gr.Slider(100, 5000, value=500, step=100, label="Max rows", visible=True)
                text_file = gr.File(label="Upload Text File", file_types=[".txt", ".md"], visible=False)
                file_info = gr.Markdown("Using entire file as corpus.", visible=False)
                
                use_hf.change(toggle_corpus_source, inputs=[use_hf], 
                            outputs=[hf_dataset, hf_split, hf_max_rows, text_file, file_info])
                
                gr.Markdown("### Generation Mode")
                abstract_mode = gr.Checkbox(label="Narrative Story Mode", value=True)
                creativity = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Creative Language", 
                                      info="Higher = more sophisticated vocabulary and novel phrasing", visible=True)
                
                abstract_mode.change(toggle_mode, inputs=[abstract_mode], outputs=[creativity])
                
                gr.Markdown("### Parameters")
                seed = gr.Number(value=42, label="Seed")
                maxtokens = gr.Slider(100, 1000, value=400, step=50, label="Tokens")
                num_speakers = gr.Slider(2, 6, value=4, step=1, label="Narrative Voices")
                
                with gr.Accordion("Advanced", open=False):
                    steer = gr.Slider(0.5, 3, value=1.35, step=0.05, label="Steer strength")
                    focus = gr.Slider(0, 1, value=0.5, step=0.05, label="Focus strength")
                    pairwise = gr.Slider(0, 2, value=0.6, step=0.1, label="Pairwise strength")
                    centroid_strength = gr.Slider(0, 1, value=0.35, step=0.05, label="Centroid boost strength",
                                                 info="NEW: Semantic gravity toward related concepts")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Starting Prompt", 
                    value="Consider the nature of understanding", 
                    lines=2,
                    info="Opening phrase to begin the narrative"
                )
                btn = gr.Button("Generate Narrative", variant="primary", size="lg")
                
                output_text = gr.Textbox(label="Generated Narrative", lines=25, show_copy_button=True)
                output_stats = gr.Textbox(label="Generation Statistics", lines=15)

        btn.click(
            run_session,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows, text_file, 
                    prompt, seed, maxtokens, num_speakers, steer, focus, pairwise,
                    abstract_mode, creativity, centroid_strength],
            outputs=[output_text, output_stats]
        )
        
        gr.Markdown("""
        ### About Centroid of Meaning
        
        **NEW in V7.0**: Each token has a **semantic centroid** computed in 4D aesthetic space:
        - **Harmony**: string similarity (edit distance)
        - **Density**: combined token lengths
        - **Momentum**: relative length change
        - **Resonance**: hash-based variation
        
        During generation, tokens near the current token's centroid are **boosted**, creating:
        - **Semantic coherence**: Maintains conceptual flow
        - **Multi-scale semantics**: Local (pairwise) + global (centroid) relationships
        - **Adaptive generation**: High-coherence tokens have stable centroids, low-coherence spread out
        
        Adjust **Centroid boost strength** to control:
        - **High (0.5-0.8)**: Stay near current concept (tight semantic flow)
        - **Medium (0.3-0.4)**: Balance coherence with exploration
        - **Low (0.1-0.2)**: Allow more conceptual jumps
        - **Zero**: Disable for maximum creativity
        """)

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.queue().launch(share=False)
