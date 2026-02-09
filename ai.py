#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Theoretic Neurosymbolic Text Generator with Neuronal Activator
V5.0 - PyTorch Bootstrapping & Pairwise Symbolic Activation

Core Philosophy:
Replaces static lookup tables with a trainable 'Neuronal Activator'. 
This module bootstraps itself on the corpus structure, learning to fire 
based on aesthetic symbol properties (Harmony, Density, Momentum, Resonance)
encoded in a shared embedding space.

New Features:
1. NeuronalActivator (nn.Module): A learnable pairwise processor.
2. PyTorch Bootstrapping: Pre-trains the activator on corpus bigrams to 
   internalize symbolic rules before generation.
3. Differentiable Aesthetics: Symbolic values are now continuous gradients.
4. Osculating Fuzzy Control: Smooth 2nd-order membership functions.

Dependencies:
  pip install gradio numpy torch datasets
"""

from __future__ import annotations
import re
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from datasets import load_dataset

# ----------------------------
# NEURONAL ACTIVATOR (PyTorch)
# ----------------------------

class NeuronalActivator(nn.Module):
    """
    A learnable module that replaces the static 'Chinese Room' engine.
    It embeds tokens and fires based on pairwise aesthetic features.
    
    Bootstrapping Phase:
    Before generation, this network trains on the corpus to learn 
    classic symbolic rules (Harmony, Density, etc.), 'implanting' 
    the rules into its weights.
    """
    def __init__(self, vocab_size=50000, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Shared token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Projects concatenated pair embeddings to 4 latent aesthetic features
        self.feature_proj = nn.Linear(embed_dim * 2, 4)
        
        # Neural firing mechanism (MLP)
        self.neuron_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Global aesthetic state (learnable context)
        self.register_buffer('global_mean', torch.zeros(4))
        self.resonance_scale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        nn.init.xavier_normal_(self.feature_proj.weight)
        nn.init.orthogonal_(self.neuron_mlp[0].weight)

    def _hash_token(self, token: str) -> torch.Tensor:
        """Deterministic hash for vocabulary management."""
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return torch.tensor(h % self.token_embed.num_embeddings, device=self.token_embed.weight.device)

    def get_features(self, t1: str, t2: str) -> torch.Tensor:
        """Compute the 4D aesthetic vector for a pair."""
        with torch.no_grad():
            idx1 = self._hash_token(t1)
            idx2 = self._hash_token(t2)
        
        emb1 = self.token_embed(idx1)
        emb2 = self.token_embed(idx2)
        
        # Concatenate and project to latent aesthetic space
        concat = torch.cat([emb1, emb2])
        raw_feats = self.feature_proj(concat)
        
        # Normalize to [0,1]
        return torch.sigmoid(raw_feats)

    def forward(self, t1: str, t2: str) -> Tuple[float, torch.Tensor]:
        """
        Returns:
            firing_rate (float): 0.0-1.0 activation level
            features (Tensor): 4D aesthetic vector
        """
        features = self.get_features(t1, t2)
        
        # Calculate deviation from global mean (Lateral Inhibition concept)
        deviation = features - self.global_mean
        
        # Neuron firing logic
        # Input is features + weighted deviation
        neuron_in = features + (deviation * 0.5)
        potential = self.neuron_mlp(neuron_in)
        
        firing_rate = torch.sigmoid(potential).item()
        return firing_rate, features

    def bootstrap(self, tokens: List[str], epochs: int = 50, lr: float = 0.01, progress=None):
        """
        Self-supervised training phase.
        The network learns to predict the 'classic' symbolic rules 
        from the raw token embeddings, essentially 'compiling' the 
        rulebook into its neural weights.
        """
        if len(tokens) < 2:
            return

        optimizer = Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # 1. Generate Training Data (Classic Rules)
        pairs = []
        targets = []
        
        seen = set()
        
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if (t1, t2) in seen: continue
            seen.add((t1, t2))
            
            # --- Classic Symbolic Rules (Ground Truth) ---
            # Harmony: Edit distance approximation
            dist = self._levenshtein(t1, t2) / max(len(t1), len(t2), 1)
            harmony = 1.0 - dist
            
            # Density
            density = math.tanh((len(t1) + len(t2)) / 20.0)
            
            # Momentum
            if len(t1) > 0:
                m = (len(t2) - len(t1)) / (len(t1) + len(t2))
                momentum = (m + 1.0) / 2.0
            else:
                momentum = 0.5
                
            # Resonance (Hash-based)
            pair_str = f"{t1}|{t2}"
            h_val = int(hashlib.md5(pair_str.encode()).hexdigest(), 16)
            resonance = (h_val % 10000) / 10000.0
            
            pairs.append((t1, t2))
            targets.append([harmony, density, momentum, resonance])
            
            if len(pairs) > 2000: # Limit bootstrap size
                break
        
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.token_embed.weight.device)
        
        # 2. Training Loop
        self.train()
        batch_size = 32
        
        for epoch in range(epochs):
            total_loss = 0
            indices = torch.randperm(len(pairs))
            
            for i in range(0, len(pairs), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_targets = target_tensor[batch_idx]
                
                # Get embeddings for batch
                batch_t1_idx = torch.stack([self._hash_token(pairs[k][0]) for k in batch_idx])
                batch_t2_idx = torch.stack([self._hash_token(pairs[k][1]) for k in batch_idx])
                
                emb1 = self.token_embed(batch_t1_idx)
                emb2 = self.token_embed(batch_t2_idx)
                
                preds = torch.sigmoid(self.feature_proj(torch.cat([emb1, emb2], dim=1)))
                
                loss = loss_fn(preds, batch_targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if progress and epoch % 10 == 0:
                progress(0.2 + (0.1 * epoch/epochs), desc=f"Bootstrapping Neurons (Loss: {total_loss:.4f})")

        # 3. Update Global Mean based on corpus
        with torch.no_grad():
            self.eval()
            all_feats = []
            for t1, t2 in pairs:
                all_feats.append(self.get_features(t1, t2))
            if all_feats:
                self.global_mean.data = torch.stack(all_feats).mean(dim=0)

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return NeuronalActivator._levenshtein(s2, s1)
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

# ----------------------------
# CONSTANTS & COGNITIVE PATTERNS
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
    r"(?:solve|address|overcome|tackle|resolve)\s+(?:the\s+)?(?:problem|issue|challenge)[s]?\s*(?:of|in|with)?",
    r"(?:facing|encounter|meet|confront)\s+(?:the\s+)?(?:problem|issue|difficulty)[s]?\s*(?:of|with)?",
]

SOLUTION_PATTERNS = [
    r"(?:solution|answer|resolution|fix|remedy|approach|method|strategy|technique)[s]?\s*(?:is|are|to|for|of)?\s*(?:the\s+)?(?:\w+\s*){0,8}(?:\.|:)",
    r"(?:solved|resolved|fixed|addressed|overcome|tackled)\s+(?:by|using|through|with)\s*(?:the\s+)?(?:\w+\s*){0,10}",
    r"(?:key|best|optimal|effective)\s+(?:solution|approach|strategy|method)[s]?\s*(?:is|are|:)",
    r"(?:this|it)\s+(?:can|may|might|should|will)\s+be\s+(?:solved|addressed|resolved)\s+(?:by|using|with)",
]

# ----------------------------
# Text Processing & Utilities
# ----------------------------

def inject_cognitive_tokens(text: str) -> str:
    lines = text.split('\n')
    marked_lines = []
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
        marked_lines.append(line)
    return '\n'.join(marked_lines)

def _token_class(tok: str) -> str:
    if tok in COGNITIVE_TOKENS: return "COG"
    if tok in [".", ",", ";", ":", "!", "?", "(", ")"]: return "PUNC"
    if not re.match(r"[a-z]", tok): return "OTHER"
    L = len(tok)
    return "S" if L <= 3 else "M" if L <= 7 else "L"

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
        if t in COGNITIVE_TOKENS: continue
        if t in [".", ",", ";", ":", "!", "?", ")", "("]:
            if t in ["(", ")"]: out.append(t)
            else:
                if out: out[-1] += t
                else: out.append(t)
        else:
            if out and out[-1].endswith("("): out[-1] += t
            else: out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s

def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in [".txt", ".md"]:
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError("Unsupported file extension")

def _resolve_gradio_file_to_path(infile) -> str:
    if infile is None: raise ValueError("No input file provided.")
    if isinstance(infile, str): return infile
    if hasattr(infile, "name") and isinstance(infile.name, str): return infile.name
    if isinstance(infile, dict) and "path" in infile: return str(infile["path"])
    if hasattr(infile, "path"): return str(infile.path)
    raise ValueError(f"Unsupported infile type: {type(infile)}")

# ----------------------------
# Pure-Python TF-IDF + SVD
# ----------------------------

def pure_tfidf(docs: List[str], max_features: int = 8000) -> Tuple[np.ndarray, List[str]]:
    all_words = set()
    for doc in docs:
        words = re.findall(r"\b\w+\b", doc.lower())
        all_words.update(words)
    vocab = list(all_words)[:max_features]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        word_counts = {}
        for word in re.findall(r"\b\w+\b", doc.lower()):
            word_counts[word] = word_counts.get(word, 0) + 1
        if not word_counts: continue
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
    if k < 1: return type("SVD", (), {"components_": np.zeros((0, n))})()
    Q = np.random.randn(n, k)
    Q, _ = np.linalg.qr(Q)
    for _ in range(10):
        B = X.T @ X @ Q
        Q, _ = np.linalg.qr(B)
    B = X @ Q
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return type("SVD", (), {"components_": Vt[:k]})()

# ----------------------------
# Graph Components
# ----------------------------

@dataclass
class SimpleGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, Dict[str, Any]]]
    cognitive_map: Dict[str, List[int]]
    pairwise_aesthetics: Dict[Tuple[int, int], torch.Tensor]

    @classmethod
    def from_token_sequence(cls, tokens: List[str], activator: NeuronalActivator, max_nodes: int = 220):
        toks = tokens[:max_nodes]
        nodes = [{"id": i, "cls": _token_class(t), "token": t} for i, t in enumerate(toks)]
        edges = []
        cog_map = {"[PROBLEM]": [], "[SOLUTION]": [], "[PAIR-BEGIN]": [], "[PAIR-END]": []}
        pairwise_aesthetics = {}

        # Process edges with Neuronal Activator
        for i in range(len(toks) - 1):
            edge_data = {"rel": "adj"}
            edges.append((i, i + 1, edge_data))
            
            # Use trained activator to get features
            _, features = activator(toks[i], toks[i+1])
            pairwise_aesthetics[(i, i+1)] = features.detach().cpu()
        
        for i in range(len(toks) - 2):
            edges.append((i, i + 2, {"rel": "skip"}))
        
        for i, t in enumerate(toks):
            if t in COGNITIVE_TOKENS:
                cog_map[t].append(i)

        return cls(nodes, edges, cog_map, pairwise_aesthetics)

    def get_aesthetic_flow(self) -> float:
        if not self.pairwise_aesthetics:
            return 0.5
        vectors = list(self.pairwise_aesthetics.values())
        if not vectors: return 0.5
        stacked = torch.stack(vectors)
        return float(torch.mean(torch.norm(stacked, dim=1)).item())

    # Standard metrics
    def degree_histogram(self, max_bins: int = 16) -> np.ndarray:
        degrees = [0] * max_bins
        node_deg = {node["id"]: 0 for node in self.nodes}
        for u, v, _ in self.edges:
            node_deg[u] += 1
            node_deg[v] += 1
        for d in node_deg.values():
            if d < max_bins: degrees[d] += 1
        return np.array(degrees)

    def weisfeiler_lehman_hash(self, iterations: int = 3, digest_size: int = 16) -> str:
        labels = {node["id"]: node["cls"] for node in self.nodes}
        adj = {node["id"]: [] for node in self.nodes}
        for u, v, _ in self.edges:
            adj[u].append(v); adj[v].append(u)
        for _ in range(iterations):
            new_labels = {}
            for node_id in labels:
                neighbors = sorted([labels[n] for n in adj[node_id]])
                combined = (labels[node_id],) + tuple(neighbors)
                new_hash = hash(combined) % (10**digest_size)
                new_labels[node_id] = f"{labels[node_id]}_{new_hash}"
            labels = new_labels
        final_hash = sum(hash((k, labels[k])) for k in labels) % (10**digest_size)
        return f"{final_hash:0{digest_size}d}"

    def automorphism_estimate(self, max_count: int = 150) -> int:
        labels = {node["id"]: node["cls"] for node in self.nodes}
        counts = {}
        for l in labels.values(): counts[l] = counts.get(l, 0) + 1
        prod = 1
        for c in counts.values(): prod *= c
        return min(max_count, prod)

def graph_signature(G: SimpleGraph) -> Dict[str, object]:
    return {
        "deg_hist": G.degree_histogram(),
        "wl": G.weisfeiler_lehman_hash(),
        "aut_est": G.automorphism_estimate(),
        "cognitive_density": sum(len(v) for v in G.cognitive_map.values()),
        "aesthetic_flow": G.get_aesthetic_flow(),
    }

# ----------------------------
# Fuzzy Logic Controller
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

def tnorm_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: return a * b
def snorm_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: return torch.maximum(a, b)

class FuzzyWeightController(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_low, self.z_mid, self.z_high = 0.20, 0.55, 0.95

    @torch.no_grad()
    def forward(self, entropy01: torch.Tensor, peak01: torch.Tensor,
                boost01: torch.Tensor, aesthetic_flow: float = 0.5,
                osculator_strength: float = 0.0) -> torch.Tensor:
        e, p, b = entropy01.clamp(0, 1), peak01.clamp(0, 1), boost01.clamp(0, 1)
        a = torch.tensor(aesthetic_flow, device=e.device).clamp(0, 1)
        
        # Memberships
        e_low = mf_trap(e, 0.0, 0.0, 0.25, 0.45)
        e_mid = mf_tri(e, 0.25, 0.50, 0.75)
        e_high = mf_trap(e, 0.55, 0.75, 1.0, 1.0)
        p_low = mf_trap(p, 0.0, 0.0, 0.20, 0.40)
        p_mid = mf_tri(p, 0.25, 0.50, 0.75)
        p_high = mf_trap(p, 0.60, 0.80, 1.0, 1.0)
        b_low = mf_trap(b, 0.0, 0.0, 0.20, 0.45)
        b_mid = mf_tri(b, 0.25, 0.50, 0.75)
        b_high = mf_trap(b, 0.55, 0.80, 1.0, 1.0)
        a_high = mf_trap(a, 0.5, 0.7, 1.0, 1.0)

        # Rules
        w1 = tnorm_prod(e_high, p_low)
        w2 = tnorm_prod(e_mid, b_mid)
        w3 = snorm_max(p_high, b_high)
        w4 = tnorm_prod(e_low, p_mid)
        w5 = tnorm_prod(a_high, e_mid)

        Z = torch.tensor([self.z_high, self.z_mid, self.z_low, self.z_low, self.z_high], 
                        device=e.device).float()
        W = torch.stack([w1, w2, w3, w4, w5]).float().clamp_min(0.0)

        g = (W * Z).sum() / (W.sum() + 1e-12)
        
        # Osculating modification (Smooth 2nd order blend)
        if osculator_strength > 0.05:
            osc_val = 1.0 - ((g - 0.5) / 0.5).pow(2) # Parabola centered at 0.5
            g = (1.0 - osculator_strength) * g + osculator_strength * osc_val
            
        return g.clamp(0.0, 0.5)

# ----------------------------
# Neural Modules
# ----------------------------

class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.15, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = int(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1: x = x.view(1, 1, -1)
        elif x.dim() == 2: x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)

class SynapticPruner(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(int(n_features)))
    def forward(self, W: torch.Tensor) -> torch.Tensor: return W * self.gain.view(1, -1)

# ----------------------------
# Quadgram LM
# ----------------------------

class QuadgramLM:
    def __init__(self, add_k: float = 0.25, activator: Optional[NeuronalActivator] = None):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0
        self.activator = activator

    def ingest(self, tokens: List[str]) -> None:
        self.uni.clear(); self.bi.clear(); self.tri.clear(); self.quad.clear()
        self.total = 0
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1; self.total += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i+1]); self.bi[k] = self.bi.get(k, 0) + 1
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i+1], tokens[i+2]); self.tri[k] = self.tri.get(k, 0) + 1
        for i in range(len(tokens) - 3):
            k = (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]); self.quad[k] = self.quad.get(k, 0) + 1
        self.vocab = list(self.uni.keys())

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor, List[float]]:
        cont = []
        for (a, b, c, d), count in self.quad.items():
            if a == w1 and b == w2 and c == w3: cont.append(d)
        if not cont:
            for (a, b, c), count in self.tri.items():
                if a == w2 and b == w3: cont.append(c)
        if not cont:
            for (a, b), count in self.bi.items():
                if a == w3: cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)]

        seen = set()
        cand = []
        for w in cont:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w); cand.append(w)
        cand = cand

        V = len(self.vocab) + 1
        add_k = self.add_k

        def get_prob(w4: str) -> float:
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0: return (c1234 + add_k) / (c123 + add_k * V)
            c12 = self.bi.get((w2, w3), 0)
            c123_tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0: return (c123_tri + add_k) / (c12 + add_k * V)
            c1 = self.uni.get(w3, 0)
            c12_bi = self.bi.get((w3, w4), 0)
            if c1 > 0: return (c12_bi + add_k) / (c1 + add_k * V)
            return (self.uni.get(w4, 0) + add_k) / (self.total + add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        if probs.numel() > 0: probs = probs / (probs.sum() + 1e-12)
        else: cand = ["the"]; probs = torch.ones(1)
        
        # Get pairwise firing rates from Neuronal Activator
        pair_weights = []
        if self.activator:
            for w4 in cand:
                fire, _ = self.activator(w3, w4)
                pair_weights.append(fire)
        else:
            pair_weights = [1.0] * len(cand)
            
        return cand, probs, pair_weights

# ----------------------------
# System State
# ----------------------------

@dataclass
class Nodelet:
    idx: int; top_terms: List[Tuple[str, float]]; energy: float; narrative: str

@dataclass
class ModelState:
    nodelets: List[Nodelet]
    vocab100: List[str]
    binding_W: torch.Tensor
    bar_probs: torch.Tensor
    token_boost: Dict[str, float]
    pillar_weights: torch.Tensor
    geometric_bias: torch.Tensor
    semantic_graph: SimpleGraph
    lm_graph: Any
    activator: NeuronalActivator

@dataclass
class PreparedCorpus:
    text: str; tokens: List[str]; lm: QuadgramLM; state: ModelState; ref_sig: Dict[str, object]

class RadixLRUCache:
    def __init__(self, max_items: int = 25000):
        self.max_items = int(max(256, max_items))
        self._od = OrderedDict()
    def get(self, key):
        v = self._od.get(key, None)
        if v is None: return None
        self._od.move_to_end(key)
        return v
    def put(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.max_items: self._od.popitem(last=False)
    def clear(self): self._od.clear()

# ----------------------------
# NeuroSymbolicGraphGenerator
# ----------------------------

class NeuroSymbolicGraphGenerator:
    def __init__(
        self, nodelets_n=10, bars_n=100, svd_random_state=7, softmax_temp=0.85,
        steer_strength=1.35, lm_add_k=0.25, pillar_strength=0.85, geometric_strength=0.3,
        rfe_enabled=True, rfe_iterations=3, rfe_removal_rate=0.15, focus_strength=0.5,
        radix_cache_items=25000, pairwise_strength=0.4, osculator_strength=0.1
    ):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp, self.lm_add_k = float(softmax_temp), float(lm_add_k)
        self.pillar_strength, self.geometric_strength = float(pillar_strength), float(geometric_strength)
        self.rfe_enabled, self.rfe_iterations = bool(rfe_enabled), int(rfe_iterations)
        self.rfe_removal_rate = float(rfe_removal_rate)
        self.pairwise_strength = float(pairwise_strength)
        self.osculator_strength = float(osculator_strength)
        self.base_steer, self.base_temp = float(steer_strength), float(softmax_temp)
        
        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.fuzzy_ctl = FuzzyWeightController()
        self.pruner = None
        self.cache_version = 0
        self.radix_cache = RadixLRUCache(max_items=int(radix_cache_items))

    def _pick_initial_context(self, lm: QuadgramLM, seed_words: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_'-]*$", t) and t not in COGNITIVE_TOKENS]
        if len(sw) >= 3: return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2: return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1: return (sw[-1], sw[-1], sw[-1])
        seed_tok = lm.vocab[0] if lm.vocab else "the"
        return (seed_tok, seed_tok, seed_tok)

    def _synaptic_prune(self, W: torch.Tensor, energies: torch.Tensor, vocab100: List[str], progress=None):
        if not self.rfe_enabled or self.rfe_iterations <= 0: return W, vocab100
        k, bars_n = W.shape
        self.pruner = SynapticPruner(bars_n)
        W_curr = W.detach().clone().requires_grad_(True)
        kept_mask = torch.ones(bars_n, dtype=torch.bool)

        for iteration in range(self.rfe_iterations):
            if progress: progress(0.80 + 0.05 * (iteration / max(1, self.rfe_iterations)), desc=f"Synaptic Pruning {iteration+1}")
            W_modulated = self.pruner(W_curr)
            var_term = 0.0
            if W_modulated.size(0) >= 2: var_term = torch.var(W_modulated, dim=0, correction=1).sum()
            loss = -torch.sum(W_modulated * energies.view(-1, 1)) + 0.1 * var_term
            loss.backward()
            with torch.no_grad():
                grads = W_curr.grad.abs().sum(dim=0)
                weights = W_curr.abs().sum(dim=0)
                importance = 0.6 * weights + 0.4 * grads
                importance = importance / (importance.max() + 1e-12)
                n_keep = int(kept_mask.sum().item() * (1.0 - self.rfe_removal_rate))
                if n_keep < 10: break
                active = torch.where(kept_mask)[0]
                local_imp = importance[active]
                _, top = torch.topk(local_imp, k=min(n_keep, local_imp.numel()))
                new_mask = torch.zeros_like(kept_mask)
                new_mask[active[top]] = True
                kept_mask = new_mask
                W_curr.grad.zero_()

        with torch.no_grad():
            final_idx = torch.where(kept_mask)[0]
            W_final = W[:, final_idx]
            vocab_final = [vocab100[i] for i in final_idx.tolist()]
        return W_final, vocab_final

    def build_state(self, text: str, progress=None) -> ModelState:
        if progress: progress(0, desc="Initializing Activator")
        
        # 1. Initialize and Bootstrap Neuronal Activator
        activator = NeuronalActivator()
        tokens = basic_tokenize(text)
        activator.bootstrap(tokens, epochs=50, progress=progress)
        
        if progress: progress(0.3, desc="TF-IDF Analysis")
        clean_text = text.replace("[PROBLEM]", "").replace("[SOLUTION]", "")
        docs = re.split(r"\n\s*\n", clean_text)
        X, vocab = pure_tfidf(docs, max_features=8000)

        if X.size == 0 or len(vocab) == 0:
            vocab100 = ["the", "is", "a"]
            probs = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
            return ModelState([], vocab100, torch.zeros(0, 3), probs, {}, 
                            torch.zeros_like(probs), torch.zeros_like(probs), 
                            SimpleGraph([], [], {}, {}), None, activator)

        top_idx = np.argsort(-X.sum(axis=0))[: self.bars_n]
        vocab100 = [vocab[i] for i in top_idx]
        X_svd = X[:, top_idx]

        svd = pure_truncated_svd(X_svd, n_components=min(self.nodelets_n, min(X_svd.shape), 10))
        nodelets = []
        for i, comp in enumerate(svd.components_):
            terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))], key=lambda x: -abs(x[1]))
            eng = float(np.linalg.norm(comp))
            nodelets.append(Nodelet(i, terms, eng, f"Nodelet {i}"))

        W = torch.tensor(svd.components_, dtype=torch.float32)
        W = F.relu(W)
        if W.numel() > 0: W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)

        energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32) if nodelets else torch.ones(1)
        energies = energies / (energies.max() + 1e-12)

        if W.numel() > 0:
            W, vocab100 = self._synaptic_prune(W, energies, vocab100, progress)
            logits = (energies.view(-1, 1) * W).sum(dim=0)
            probs = F.softmax(logits / max(self.softmax_temp, 1e-6), dim=-1)
            probs = self.focus_layer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)
        else:
            probs = torch.ones(len(vocab100)) / max(1, len(vocab100))

        token_boost = {}
        for w, p in zip(vocab100, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) > 2 and subw not in STOP_WORDS:
                    token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        return ModelState(nodelets, vocab100, W, probs, token_boost, 
                         torch.zeros_like(probs), torch.zeros_like(probs), 
                         SimpleGraph([], [], {}, {}), None, activator)

    def prepare_corpus(self, text: str, progress=None) -> PreparedCorpus:
        text = inject_cognitive_tokens(text)
        text = normalize(text)
        state = self.build_state(text, progress)
        tokens = basic_tokenize(text)
        
        lm = QuadgramLM(self.lm_add_k, activator=state.activator)
        lm.ingest(tokens)
        
        for tok, boost_val in COGNITIVE_TOKENS.items():
            state.token_boost[tok] = boost_val
            
        G = SimpleGraph.from_token_sequence(tokens, state.activator)
        state.semantic_graph = G
        ref_sig = graph_signature(G)
        return PreparedCorpus(text, tokens, lm, state, ref_sig)

    def _final_probs_for_context_cached(self, prep: PreparedCorpus, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor, float]:
        key = (int(self.cache_version), str(w1), str(w2), str(w3))
        cached = self.radix_cache.get(key)
        if cached: return cached

        cand, base_probs, pair_weights = prep.lm.next_distribution(w1, w2, w3)
        if not cand:
            cand = prep.lm.vocab if prep.lm.vocab else ["the", "is", "a"]
            base_p = torch.ones(len(cand)) / max(1, len(cand))
            pair_weights = [1.0] * len(cand)
        else:
            base_p = base_probs.detach().clone().float()
            if base_p.numel() != len(cand): base_p = torch.ones(len(cand)) / max(1, len(cand))

        base_p = base_p.view(-1) / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        val = base_p.clamp_min(1e-12)
        H = -torch.sum(base_p * torch.log(val))
        entropy01 = (H / max(1e-9, math.log(max(2.0, float(base_p.numel()))))).clamp(0.0, 1.0)
        peak01 = base_p.max().clamp(0.0, 1.0)
        
        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand]).view(-1)
        
        # Inject Neural Firing Rates
        pair_tensor = torch.tensor(pair_weights, dtype=torch.float32).view(-1)
        # Normalize firing rates around 1.0
        pair_tensor = pair_tensor / (pair_tensor.mean() + 1e-12)
        boosts = boosts + self.pairwise_strength * pair_tensor
        
        context_str = f"{w1} {w2} {w3}"
        gravity = 0.3 if "[SOLUTION]" in context_str else (0.2 if "[PROBLEM]" in context_str else 0.0)
        boost01 = torch.tanh((boosts.abs().mean() + gravity) / 3.0).clamp(0.0, 1.0)
        
        aesthetic_flow = prep.state.semantic_graph.get_aesthetic_flow()
        g = self.fuzzy_ctl(entropy01, peak01, boost01, aesthetic_flow, osculator_strength=self.osculator_strength)

        eff_steer = self.base_steer * float(g.item())
        eff_temp = self.base_temp * (1.2 - 0.7 * float(g.item()))
        potentials = torch.log(base_p.clamp_min(1e-12)) + eff_steer * boosts
        final_probs = F.softmax(potentials / max(eff_temp, 1e-6), dim=-1)

        result = (cand, final_probs.detach(), float(g.item()))
        self.radix_cache.put(key, result)
        return result

# ----------------------------
# Decoder and Execution
# ----------------------------

@dataclass
class DecodeStream:
    stream_id: int; tokens_out: List[str]; w1: str; w2: str; w3: str
    done: bool = False; alpha_count: int = 0; max_steps: int = 1000
    stop_tokens: set = field(default_factory=lambda: {".", "!", "?"})
    min_alpha: int = 200

class ContinuousBatchDecoder:
    def __init__(self, gen, prep, rng, token_budget_per_round=64):
        self.gen, self.prep, self.rng = gen, prep, rng
        self.token_budget_per_round = int(max(1, token_budget_per_round))

    def _sample_fuzzy(self, cand, probs, g):
        p = probs.detach().cpu().numpy()
        p = p / (p.sum() + 1e-12)
        idx = self.rng.choice(len(cand), p=p)
        return cand[idx]

    def decode_step(self, streams: List[DecodeStream]) -> List[DecodeStream]:
        active = [s for s in streams if not s.done]
        if not active: return streams
        steps_taken = 0
        while steps_taken < self.token_budget_per_round and active:
            s = active[steps_taken % len(active)]
            cand, probs, g = self.gen._final_probs_for_context_cached(self.prep, s.w1, s.w2, s.w3)
            next_token = self._sample_fuzzy(cand, probs, g)
            s.tokens_out.append(next_token)
            s.w1, s.w2, s.w3 = s.w2, s.w3, next_token
            if re.match(r"[a-zA-Z]", next_token): s.alpha_count += 1
            if next_token in s.stop_tokens and s.alpha_count >= s.min_alpha: s.done = True
            if len(s.tokens_out) >= s.max_steps: s.done = True
            steps_taken += 1
            active = [s for s in streams if not s.done]
        return streams

# ----------------------------
# UI & Main
# ----------------------------

@dataclass
class SGPrompt: text: str
@dataclass
class SGContext:
    text: str; gen: NeuroSymbolicGraphGenerator; seed: int
    prepared: Optional[PreparedCorpus] = None
    def ensure_prepared(self):
        if not self.prepared:
            torch.manual_seed(self.seed); np.random.seed(self.seed)
            self.prepared = self.gen.prepare_corpus(self.text)

def sg_fork(ctx: SGContext, prompt: SGPrompt, n: int = 3) -> List[Tuple[SGContext, str]]:
    ctx.ensure_prepared()
    return [(ctx, prompt.text) for _ in range(n)]

def sg_gen_batched(contexts: List[SGContext], prompts: List[str], max_tokens: int = 200, stop_at_punc: bool = True) -> List[str]:
    if not contexts: return []
    gen = contexts[0].gen
    rng = np.random.default_rng(contexts[0].seed)
    streams = []
    for i, (ctx, p_txt) in enumerate(zip(contexts, prompts)):
        ctx.ensure_prepared()
        seed_toks = basic_tokenize(p_txt)
        w1, w2, w3 = gen._pick_initial_context(ctx.prepared.lm, seed_toks)
        streams.append(DecodeStream(i, [], w1, w2, w3, max_steps=max_tokens, min_alpha=10 if stop_at_punc else 9999))
    
    decoder = ContinuousBatchDecoder(gen, contexts[0].prepared, rng)
    while any(not s.done for s in streams):
        streams = decoder.decode_step(streams)
    return [detokenize(s.tokens_out) for s in streams]

def sg_join(prompts: List[str], joiner: str = "\n") -> SGPrompt: return SGPrompt(joiner.join(prompts))

def run_program(infile, use_hf, hf_dataset, hf_split, hf_max_rows, n_take, seed, steer, focus, takeaway_prompt, summary_prompt, pairwise_strength, osculator_strength, progress=gr.Progress()):
    if use_hf:
        try:
            ds = load_dataset(hf_dataset, split=hf_split)
            rows = int(hf_max_rows) if hf_max_rows > 0 else len(ds)
            corpus_text = "\n".join(str(x) for x in ds.select(range(min(rows, len(ds))))["text"])
        except Exception as e: return f"Dataset Error: {e}"
    else:
        try: corpus_text = load_text(_resolve_gradio_file_to_path(infile))
        except Exception as e: return f"File Error: {e}"

    gen = NeuroSymbolicGraphGenerator(
        nodelets_n=10, bars_n=100, svd_random_state=int(seed),
        steer_strength=float(steer), focus_strength=float(focus),
        pairwise_strength=float(pairwise_strength), osculator_strength=float(osculator_strength)
    )
    ctx = SGContext(corpus_text, gen, seed=int(seed))
    ctx.ensure_prepared()

    # Display Neuronal Activator stats
    act = ctx.prepared.state.activator
    header = (
        f"[NEURONAL ACTIVATOR STATUS]\n"
        f"Bootstrapped on corpus structure.\n"
        f"Global Aesthetic Mean (Learned): {act.global_mean.detach().cpu().numpy()}\n"
        f"Aesthetic Flow: {ctx.prepared.state.semantic_graph.get_aesthetic_flow():.3f}\n"
        f"{'-'*40}\n\n"
    )

    root = SGPrompt(str(takeaway_prompt).strip() + "\n\n")
    branches = sg_fork(ctx, root, n=int(n_take))
    branch_ctxs, branch_prompts = zip(*branches)
    
    take_texts = sg_gen_batched(list(branch_ctxs), [p + f"[Takeaway {i+1}] " for i, p in enumerate(branch_prompts)], max_tokens=620)
    merged = sg_join([p + t for p, t in zip(branch_prompts, take_texts)], joiner="\n\n")
    final_prompt = SGPrompt(summary_prompt.replace("{joined_takeaways}", merged.text))
    final_text = sg_gen_batched([ctx], [final_prompt.text], max_tokens=460)[0]

    return header + final_prompt.text + " " + final_text

def build_app():
    with gr.Blocks(title="Neurosymbolic V5.0 - Neuronal Activator") as demo:
        gr.Markdown(
            "# Neurosymbolic V5.0: Neuronal Activator Bootstrapping\n"
            "**Core Philosophy:** Replaces static symbolic rules with a trainable 'Neuronal Activator' that "
            "learns to fire based on aesthetic properties (Harmony, Density, Momentum) by bootstrapping itself on the corpus.\n\n"
            "**PyTorch Bootstrapping:** The network pre-trains on bigrams to internalize symbolic rules before generation begins."
        )
        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face dataset", value=True)
                hf_dataset = gr.Textbox(label="HF dataset", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hf_split = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(0, 20000, value=2000, label="Max rows")
                infile = gr.File(label="Input File (txt/md)")
            with gr.Column(scale=2):
                out_txt = gr.Textbox(label="Output", lines=24)
        with gr.Row():
            n_take = gr.Slider(1, 10, value=5, label="Batches")
            seed = gr.Number(value=42, label="Seed")
            steer = gr.Slider(0, 5, value=1.35, label="Steer")
            focus = gr.Slider(0, 1, value=0.5, label="Focus")
            pairwise = gr.Slider(0, 2, value=0.4, label="Neuronal Strength")
            oscs = gr.Slider(0, 1, value=0.1, label="Osculator Strength")
        
        btn = gr.Button("Run Neuronal Generator", variant="primary")
        btn.click(run_program, inputs=[infile, use_hf, hf_dataset, hf_split, hf_max_rows, n_take, seed, steer, focus, gr.Textbox(label="Prefix"), gr.Textbox(label="Summary Prompt", value="explain this?"), pairwise, oscs], outputs=out_txt)
    return demo

if __name__ == "__main__":
    build_app().queue().launch()
