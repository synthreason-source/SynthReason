#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Theoretic Neurosymbolic Text Generator with Chinese Room Pairwise Symbolics
V4.0 - Uniform Artistic Values via Symbolic Manipulation

Core Philosophy:
Following Searle's Chinese Room argument, this system performs purely syntactic 
transformations on symbolic pairs without claiming semantic understanding. All 
artistic values are uniformly derived from structural properties of symbol pairs.

New Features:
1. Pairwise Symbolic Imagery: Every token pair (bigram) is assigned uniform 
   artistic attributes based on pure syntactic rules.
2. Chinese Room Compliance: No emergent semantics - only rule-based symbol shuffling.
3. Uniform Aesthetic Values: All symbols receive identical treatment through 
   mathematical transformations.

Dependencies:
  pip install gradio numpy torch datasets
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import hashlib

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, Value

# ----------------------------
# CHINESE ROOM PAIRWISE SYMBOLICS
# ----------------------------

@dataclass
class SymbolicPair:
    """
    Represents a bigram with uniform artistic attributes derived purely 
    from syntactic properties (character counts, hash values, etc.)
    No semantic interpretation - pure symbol manipulation.
    """
    token1: str
    token2: str
    
    # Uniform artistic values (0-1 range)
    harmony: float      # Phonetic similarity (edit distance normalized)
    density: float      # Combined symbol complexity
    momentum: float     # Directional flow (length change)
    resonance: float    # Hash-based pseudo-random aesthetic
    
    @classmethod
    def from_tokens(cls, t1: str, t2: str) -> 'SymbolicPair':
        """
        Pure syntactic derivation of artistic values.
        Chinese Room compliant: no semantic understanding required.
        """
        # Harmony: normalized edit distance (0 = identical, 1 = maximally different)
        harmony = 1.0 - (cls._edit_distance(t1, t2) / max(len(t1), len(t2), 1))
        
        # Density: combined character count normalized
        density = math.tanh((len(t1) + len(t2)) / 20.0)
        
        # Momentum: direction of length change
        if len(t1) > 0:
            momentum = (len(t2) - len(t1)) / (len(t1) + len(t2))
            momentum = (momentum + 1.0) / 2.0  # normalize to [0,1]
        else:
            momentum = 0.5
            
        # Resonance: deterministic pseudo-random from hash
        pair_str = f"{t1}|{t2}"
        hash_val = int(hashlib.md5(pair_str.encode()).hexdigest(), 16)
        resonance = (hash_val % 10000) / 10000.0
        
        return cls(t1, t2, harmony, density, momentum, resonance)
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance for harmony calculation."""
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
        """Return uniform 4D aesthetic representation."""
        return np.array([self.harmony, self.density, self.momentum, self.resonance])
    
    def __repr__(self) -> str:
        return (f"SymPair({self.token1}→{self.token2} | "
                f"H:{self.harmony:.2f} D:{self.density:.2f} "
                f"M:{self.momentum:.2f} R:{self.resonance:.2f})")


class ChineseRoomImageryEngine:
    """
    Implements uniform symbolic transformation rules.
    Pure syntactic manipulation - no semantic understanding.
    
    The "room" receives symbol pairs and outputs aesthetic transformations
    based solely on lookup tables and arithmetic operations.
    """
    
    def __init__(self):
        self.pair_cache: Dict[Tuple[str, str], SymbolicPair] = {}
        self.global_aesthetics = {
            'harmony_mean': 0.5,
            'density_mean': 0.5,
            'momentum_mean': 0.5,
            'resonance_mean': 0.5,
        }
        
    def process_pair(self, t1: str, t2: str) -> SymbolicPair:
        """
        Chinese Room operation: lookup or compute symbolic pair.
        Pure rule following - no understanding of what tokens 'mean'.
        """
        key = (t1, t2)
        if key not in self.pair_cache:
            self.pair_cache[key] = SymbolicPair.from_tokens(t1, t2)
        return self.pair_cache[key]
    
    def batch_process(self, tokens: List[str]) -> List[SymbolicPair]:
        """Process all consecutive pairs in token sequence."""
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append(self.process_pair(tokens[i], tokens[i+1]))
        return pairs
    
    def update_global_aesthetics(self, pairs: List[SymbolicPair]):
        """
        Uniform statistical update - pure arithmetic on symbol properties.
        Chinese Room: we're just averaging numbers, not understanding content.
        """
        if not pairs:
            return
            
        harmonies = [p.harmony for p in pairs]
        densities = [p.density for p in pairs]
        momentums = [p.momentum for p in pairs]
        resonances = [p.resonance for p in pairs]
        
        self.global_aesthetics = {
            'harmony_mean': float(np.mean(harmonies)),
            'density_mean': float(np.mean(densities)),
            'momentum_mean': float(np.mean(momentums)),
            'resonance_mean': float(np.mean(resonances)),
        }
    
    def get_pair_weight(self, pair: SymbolicPair) -> float:
        """
        Uniform weighting function based on deviation from global means.
        Pure arithmetic - no semantic judgment.
        """
        vec = pair.aesthetic_vector()
        global_vec = np.array([
            self.global_aesthetics['harmony_mean'],
            self.global_aesthetics['density_mean'],
            self.global_aesthetics['momentum_mean'],
            self.global_aesthetics['resonance_mean'],
        ])
        
        # Euclidean distance in aesthetic space
        distance = float(np.linalg.norm(vec - global_vec))
        
        # Convert to weight (closer to mean = higher weight)
        weight = math.exp(-distance)
        return weight


# ----------------------------
# CONSTANTS & COGNITIVE PATTERNS (Enhanced)
# ----------------------------

COGNITIVE_TOKENS = {
    "[PROBLEM]": 2.5,
    "[SOLUTION]": 3.0,
    "[PAIR-BEGIN]": 1.5,  # Marks start of symbolic pair processing
    "[PAIR-END]": 1.5,    # Marks end of symbolic pair processing
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
# Text Processing & Cognitive Injection
# ----------------------------

def inject_cognitive_tokens(text: str) -> str:
    """Scans text for problem/solution patterns and prepends markers."""
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
        
        if i > 0 and '?' in marked_lines[-1] and not "[SOLUTION]" in line:
            if any(word in orig_line.lower() for word in ['answer', 'solution', 'because', 'since', 'therefore']):
                line = f"[SOLUTION] {line}"
        
        marked_lines.append(line)
    
    return '\n'.join(marked_lines)

def _token_class(tok: str) -> str:
    if tok in COGNITIVE_TOKENS:
        return "COG"
    if tok in [".", ",", ";", ":", "!", "?", "(", ")"]:
        return "PUNC"
    if not re.match(r"[a-z]", tok):
        return "OTHER"
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
        return type("SVD", (), {"components_": np.zeros((0, n))})()

    Q = np.random.randn(n, k)
    Q, _ = np.linalg.qr(Q)

    for _ in range(10):
        B = X.T @ X @ Q
        Q, _ = np.linalg.qr(B)

    B = X @ Q
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return type("SVD", (), {"components_": Vt[:k]})()

# ----------------------------
# Graph Components (Enhanced with Pairwise Tracking)
# ----------------------------

@dataclass
class SimpleGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, Dict[str, Any]]]
    cognitive_map: Dict[str, List[int]]
    pairwise_aesthetics: Dict[Tuple[int, int], np.ndarray]  # New: aesthetic vectors for edges

    @classmethod
    def from_token_sequence(cls, tokens: List[str], imagery_engine: ChineseRoomImageryEngine, max_nodes: int = 220):
        toks = tokens[:max_nodes]
        nodes = [{"id": i, "cls": _token_class(t), "token": t} for i, t in enumerate(toks)]
        edges = []
        cog_map = {"[PROBLEM]": [], "[SOLUTION]": [], "[PAIR-BEGIN]": [], "[PAIR-END]": []}
        pairwise_aesthetics = {}

        # Process all adjacent pairs through Chinese Room engine
        symbolic_pairs = imagery_engine.batch_process(toks)
        
        for i in range(len(toks) - 1):
            edge_data = {"rel": "adj"}
            edges.append((i, i + 1, edge_data))
            
            # Store aesthetic vector for this edge
            if i < len(symbolic_pairs):
                pairwise_aesthetics[(i, i+1)] = symbolic_pairs[i].aesthetic_vector()
        
        for i in range(len(toks) - 2):
            edges.append((i, i + 2, {"rel": "skip"}))
        
        for i, t in enumerate(toks):
            if t in COGNITIVE_TOKENS:
                cog_map[t].append(i)

        return cls(nodes, edges, cog_map, pairwise_aesthetics)

    def degree_histogram(self, max_bins: int = 16) -> np.ndarray:
        degrees = [0] * max_bins
        node_deg = {node["id"]: 0 for node in self.nodes}
        for u, v, _ in self.edges:
            node_deg[u] += 1
            node_deg[v] += 1
        for d in node_deg.values():
            if d < max_bins:
                degrees[d] += 1
        return np.array(degrees)

    def weisfeiler_lehman_hash(self, iterations: int = 3, digest_size: int = 16) -> str:
        labels = {node["id"]: node["cls"] for node in self.nodes}
        adj = {node["id"]: [] for node in self.nodes}
        for u, v, _ in self.edges:
            adj[u].append(v)
            adj[v].append(u)

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
        for l in labels.values():
            counts[l] = counts.get(l, 0) + 1
        prod = 1
        for c in counts.values():
            prod *= c
        return min(max_count, prod)
    
    def get_aesthetic_flow(self) -> float:
        """
        Compute uniform aesthetic flow through the graph.
        Chinese Room: pure arithmetic on cached aesthetic values.
        """
        if not self.pairwise_aesthetics:
            return 0.5
        
        vectors = list(self.pairwise_aesthetics.values())
        mean_vec = np.mean(vectors, axis=0)
        
        # Flow = average norm of aesthetic vectors
        flow = float(np.mean([np.linalg.norm(v) for v in vectors]))
        return flow

def graph_signature(G: SimpleGraph) -> Dict[str, object]:
    return {
        "deg_hist": G.degree_histogram(),
        "wl": G.weisfeiler_lehman_hash(),
        "aut_est": G.automorphism_estimate(),
        "cognitive_density": sum(len(v) for v in G.cognitive_map.values()),
        "aesthetic_flow": G.get_aesthetic_flow(),  # New metric
    }

# ----------------------------
# Fuzzy Logic Controller (Enhanced with Pairwise Awareness)
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
    """
    Enhanced with pairwise aesthetic awareness.
    Still purely syntactic - no semantic understanding.
    """
    def __init__(self):
        super().__init__()
        self.z_low = 0.20
        self.z_mid = 0.55
        self.z_high = 0.95

    @torch.no_grad()
    def forward(self, entropy01: torch.Tensor, peak01: torch.Tensor, 
                boost01: torch.Tensor, aesthetic_flow: float = 0.5) -> torch.Tensor:
        e = entropy01.clamp(0, 1)
        p = peak01.clamp(0, 1)
        b = boost01.clamp(0, 1)
        a = torch.tensor(aesthetic_flow, device=e.device, dtype=e.dtype).clamp(0, 1)

        # Memberships
        e_low  = mf_trap(e, 0.0, 0.0, 0.25, 0.45)
        e_mid  = mf_tri(e, 0.25, 0.50, 0.75)
        e_high = mf_trap(e, 0.55, 0.75, 1.0, 1.0)

        p_low  = mf_trap(p, 0.0, 0.0, 0.20, 0.40)
        p_mid  = mf_tri(p, 0.25, 0.50, 0.75)
        p_high = mf_trap(p, 0.60, 0.80, 1.0, 1.0)

        b_low  = mf_trap(b, 0.0, 0.0, 0.20, 0.45)
        b_mid  = mf_tri(b, 0.25, 0.50, 0.75)
        b_high = mf_trap(b, 0.55, 0.80, 1.0, 1.0)
        
        # Aesthetic flow memberships
        a_low = mf_trap(a, 0.0, 0.0, 0.3, 0.5)
        a_high = mf_trap(a, 0.5, 0.7, 1.0, 1.0)

        # Rules (enhanced with aesthetic awareness)
        w1 = tnorm_prod(e_high, p_low)
        w2 = tnorm_prod(e_mid, b_mid)
        w3 = snorm_max(p_high, b_high)
        w4 = tnorm_prod(e_low, p_mid)
        w5 = tnorm_prod(a_high, e_mid)  # High aesthetic flow + mid entropy

        Z = torch.tensor([self.z_high, self.z_mid, self.z_low, self.z_low, self.z_high], 
                        device=e.device, dtype=torch.float32)
        W = torch.stack([w1, w2, w3, w4, w5]).to(dtype=torch.float32).clamp_min(0.0)

        g = (W * Z).sum() / (W.sum() + 1e-12)
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
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)

class SynapticPruner(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(int(n_features)))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W * self.gain.view(1, -1)


# ----------------------------
# Quadgram LM (Enhanced with Pairwise Weighting)
# ----------------------------

class QuadgramLM:
    def __init__(self, add_k: float = 0.25, imagery_engine: Optional[ChineseRoomImageryEngine] = None):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0
        self.imagery_engine = imagery_engine  # Chinese Room for pairwise aesthetics

    def ingest(self, tokens: List[str]) -> None:
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self.quad.clear()
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

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor, List[float]]:
        """
        Enhanced to return pairwise aesthetic weights.
        Chinese Room: pure rule-based symbol manipulation.
        """
        cont = []
        for (a, b, c, d), count in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c), count in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b), count in self.bi.items():
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
            probs = torch.ones(1)
        
        # Get pairwise aesthetic weights (uniform treatment)
        pair_weights = []
        if self.imagery_engine:
            for w4 in cand:
                pair = self.imagery_engine.process_pair(w3, w4)
                weight = self.imagery_engine.get_pair_weight(pair)
                pair_weights.append(weight)
        else:
            pair_weights = [1.0] * len(cand)
            
        return cand, probs, pair_weights


# ----------------------------
# System State (Enhanced)
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
    pillar_weights: torch.Tensor
    geometric_bias: torch.Tensor
    semantic_graph: SimpleGraph
    lm_graph: Any
    imagery_engine: ChineseRoomImageryEngine  # New

@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    lm: QuadgramLM
    state: ModelState
    ref_sig: Dict[str, object]

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
# NeuroSymbolicGraphGenerator (Enhanced)
# ----------------------------

class NeuroSymbolicGraphGenerator:
    def __init__(
        self,
        nodelets_n: int = 10,
        bars_n: int = 100,
        svd_random_state: int = 7,
        softmax_temp: float = 0.85,
        steer_strength: float = 1.35,
        lm_add_k: float = 0.25,
        pillar_strength: float = 0.85,
        geometric_strength: float = 0.3,
        rfe_enabled: bool = True,
        rfe_iterations: int = 3,
        rfe_removal_rate: float = 0.15,
        focus_strength: float = 0.5,
        radix_cache_items: int = 25000,
        pairwise_strength: float = 0.4,  # New: weight for pairwise aesthetics
    ):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp = float(softmax_temp)
        self.lm_add_k = float(lm_add_k)
        self.pillar_strength = float(pillar_strength)
        self.geometric_strength = float(geometric_strength)
        self.rfe_enabled = bool(rfe_enabled)
        self.rfe_iterations = int(rfe_iterations)
        self.rfe_removal_rate = float(rfe_removal_rate)
        self.pairwise_strength = float(pairwise_strength)

        self.base_steer = float(steer_strength)
        self.base_temp = float(softmax_temp)

        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.fuzzy_ctl = FuzzyWeightController()

        self.pruner: Optional[SynapticPruner] = None
        self.cache_version = 0
        self.radix_cache = RadixLRUCache(max_items=int(radix_cache_items))

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

    def _build_token_structure_graph(self, tokens: List[str], imagery_engine: ChineseRoomImageryEngine, max_nodes: int = 220) -> SimpleGraph:
        return SimpleGraph.from_token_sequence(tokens, imagery_engine, max_nodes)

    def _graph_signature(self, G: SimpleGraph) -> Dict[str, object]:
        return graph_signature(G)

    def _synaptic_prune(self, W: torch.Tensor, energies: torch.Tensor, vocab100: List[str], progress=None):
        if not self.rfe_enabled or self.rfe_iterations <= 0:
            return W, vocab100
        k, bars_n = W.shape
        self.pruner = SynapticPruner(bars_n)
        W_curr = W.detach().clone().requires_grad_(True)
        kept_mask = torch.ones(bars_n, dtype=torch.bool)

        for iteration in range(self.rfe_iterations):
            if progress:
                progress(0.80 + 0.05 * (iteration / max(1, self.rfe_iterations)), desc=f"Synaptic Pruning {iteration+1}")
            W_modulated = self.pruner(W_curr)
            var_term = 0.0
            if W_modulated.size(0) >= 2:
                var_term = torch.var(W_modulated, dim=0, correction=1).sum()

            loss = -torch.sum(W_modulated * energies.view(-1, 1)) + 0.1 * var_term
            loss.backward()
            with torch.no_grad():
                grads = W_curr.grad.abs().sum(dim=0)
                weights = W_curr.abs().sum(dim=0)
                importance = 0.6 * weights + 0.4 * grads
                importance = importance / (importance.max() + 1e-12)
                n_keep = int(kept_mask.sum().item() * (1.0 - self.rfe_removal_rate))
                if n_keep < 10:
                    break
                active = torch.where(kept_mask)[0]
                local_importance = importance[active]
                _, top_local = torch.topk(local_importance, k=min(n_keep, local_importance.numel()))
                new_mask = torch.zeros_like(kept_mask)
                new_mask[active[top_local]] = True
                kept_mask = new_mask
                W_curr.grad.zero_()

        with torch.no_grad():
            final_idx = torch.where(kept_mask)[0]
            W_final = W[:, final_idx]
            vocab_final = [vocab100[i] for i in final_idx.tolist()]
        return W_final, vocab_final

    def build_state(self, text: str, progress=None) -> ModelState:
        if progress:
            progress(0, desc="Normalizing")
        
        # Initialize Chinese Room imagery engine
        imagery_engine = ChineseRoomImageryEngine()
        
        clean_text = text.replace("[PROBLEM]", "").replace("[SOLUTION]", "").replace("[PAIR-BEGIN]", "").replace("[PAIR-END]", "")
        docs = re.split(r"\n\s*\n", clean_text)[:500]
        X, vocab = pure_tfidf(docs, max_features=8000)

        if X.size == 0 or len(vocab) == 0:
            vocab100 = ["the", "is", "a"]
            probs = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
            return ModelState([], vocab100, torch.zeros(0, 3), probs, {}, 
                            torch.zeros_like(probs), torch.zeros_like(probs), 
                            SimpleGraph([], [], {}, {}), None, imagery_engine)

        top_idx = np.argsort(-X.sum(axis=0))[: self.bars_n]
        vocab100 = [vocab[i] for i in top_idx]
        X_svd = X[:, top_idx]

        n_rows, n_cols = X_svd.shape
        max_rank = min(n_rows, n_cols)
        k = 1 if max_rank <= 1 else min(self.nodelets_n, max_rank, 10)

        svd = pure_truncated_svd(X_svd, n_components=k, random_state=self.svd_random_state)

        nodelets = []
        for i, comp in enumerate(svd.components_):
            terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))], key=lambda x: -abs(x[1]))[:10]
            eng = float(np.linalg.norm(comp))
            nodelets.append(Nodelet(i, terms, eng, f"Nodelet {i}"))

        W = torch.tensor(svd.components_, dtype=torch.float32)
        W = F.relu(W)
        if W.numel() > 0:
            W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)

        energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32) if nodelets else torch.ones(1)
        energies = energies / (energies.max() + 1e-12)

        if W.numel() > 0:
            W, vocab100 = self._synaptic_prune(W, energies, vocab100, progress)
            logits = (energies.view(-1, 1) * W).sum(dim=0)
            probs = F.softmax(logits / max(self.softmax_temp, 1e-6), dim=-1)
            probs = self.focus_layer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)
        else:
            probs = torch.ones(len(vocab100), dtype=torch.float32) / max(1, len(vocab100))

        token_boost = {}
        for w, p in zip(vocab100, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) > 2 and subw not in STOP_WORDS:
                    token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        return ModelState(nodelets, vocab100, W, probs, token_boost, 
                         torch.zeros_like(probs), torch.zeros_like(probs), 
                         SimpleGraph([], [], {}, {}), None, imagery_engine)

    def prepare_corpus(self, text: str, progress=None) -> PreparedCorpus:
        if progress:
            progress(0.1, desc="Injecting Cognitive Tokens")
        
        text = inject_cognitive_tokens(text)
        text = normalize(text)
        
        state = self.build_state(text, progress)
        tokens = basic_tokenize(text)
        
        # Pass imagery engine to LM
        lm = QuadgramLM(self.lm_add_k, imagery_engine=state.imagery_engine)
        lm.ingest(tokens)
        
        # Manually boost cognitive tokens
        for tok, boost_val in COGNITIVE_TOKENS.items():
            state.token_boost[tok] = boost_val
        
        # Process all token pairs through Chinese Room
        symbolic_pairs = state.imagery_engine.batch_process(tokens)
        state.imagery_engine.update_global_aesthetics(symbolic_pairs)
            
        G = self._build_token_structure_graph(tokens, state.imagery_engine)
        state.semantic_graph = G
        ref_sig = self._graph_signature(G)
        return PreparedCorpus(text, tokens, lm, state, ref_sig)

    def _final_probs_for_context_cached(self, prep: PreparedCorpus, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor, float]:
        key = (int(self.cache_version), str(w1), str(w2), str(w3))
        cached = self.radix_cache.get(key)
        if cached is not None:
            return cached

        cand, base_probs, pair_weights = prep.lm.next_distribution(w1, w2, w3)
        if not cand:
            cand = prep.lm.vocab[:100] if prep.lm.vocab else ["the", "is", "a"]
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))
            pair_weights = [1.0] * len(cand)
        else:
            base_p = base_probs.detach().clone().to(dtype=torch.float32)
            if base_p.numel() != len(cand):
                base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))

        base_p = base_p.view(-1)
        base_p = base_p / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        # Fuzzy features
        val = base_p.clamp_min(1e-12)
        H = -torch.sum(base_p * torch.log(val))
        V = float(base_p.numel())
        entropy01 = (H / max(1e-9, math.log(max(2.0, V)))).clamp(0.0, 1.0)
        peak01 = base_p.max().clamp(0.0, 1.0)

        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand], dtype=torch.float32).view(-1)
        
        # Pairwise aesthetic modulation (Chinese Room uniform processing)
        pair_weight_tensor = torch.tensor(pair_weights, dtype=torch.float32).view(-1)
        pair_weight_tensor = pair_weight_tensor / (pair_weight_tensor.mean() + 1e-12)  # Normalize
        
        # Blend pairwise weights into boosts uniformly
        boosts = boosts + self.pairwise_strength * pair_weight_tensor
        
        context_str = f"{w1} {w2} {w3}"
        gravity = 0.0
        if "[PROBLEM]" in context_str:
            gravity = 0.2
        if "[SOLUTION]" in context_str:
            gravity = 0.3
        
        boost01 = torch.tanh((boosts.abs().mean() + gravity) / 3.0).clamp(0.0, 1.0)

        # Get aesthetic flow from graph
        aesthetic_flow = prep.state.semantic_graph.get_aesthetic_flow()
        
        # Fuzzy control with aesthetic awareness
        g = self.fuzzy_ctl(entropy01, peak01, boost01, aesthetic_flow)

        effective_steer = self.base_steer * float(g.item())
        effective_temp = self.base_temp * (1.2 - 0.7 * float(g.item()))

        potentials = torch.log(base_p.clamp_min(1e-12)) + effective_steer * boosts
        potentials = potentials / max(effective_temp, 1e-6)
        final_probs = F.softmax(potentials, dim=-1)

        result = (cand, final_probs.detach(), float(g.item()))
        self.radix_cache.put(key, result)
        return result


# ----------------------------
# Decoder (same structure)
# ----------------------------

@dataclass
class DecodeStream:
    stream_id: int
    tokens_out: List[str]
    w1: str
    w2: str
    w3: str
    done: bool = False
    alpha_count: int = 0
    max_steps: int = 1000
    stop_tokens: set = field(default_factory=lambda: {".", "!", "?"})
    min_alpha: int = 200

class ContinuousBatchDecoder:
    def __init__(self, gen: NeuroSymbolicGraphGenerator, prep: PreparedCorpus, rng: np.random.Generator, token_budget_per_round: int = 64):
        self.gen = gen
        self.prep = prep
        self.rng = rng
        self.token_budget_per_round = int(max(1, token_budget_per_round))

    def _sample_fuzzy(self, cand: List[str], probs: torch.Tensor, g: float) -> str:
        p = probs.detach().cpu().numpy().astype(np.float64)
        p = p / (float(p.sum()) + 1e-12)

        mu = float(np.mean(p))
        std = float(np.std(p))
        cutoff = mu + (1.0 - g) * std

        p_sparse = np.where(p > cutoff, p, 0.0)
        if float(p_sparse.sum()) < 1e-12:
            p_sparse = p

        p_sparse = p_sparse / (float(p_sparse.sum()) + 1e-12)
        return self.rng.choice(cand, p=p_sparse)

    def step_round(self, streams: List[DecodeStream]) -> None:
        active = [s for s in streams if not s.done]
        if not active:
            return
        active.sort(key=lambda s: (s.w1, s.w2, s.w3))
        active = active[: min(len(active), self.token_budget_per_round)]

        groups = {}
        for s in active:
            groups.setdefault((s.w1, s.w2, s.w3), []).append(s)

        for (w1, w2, w3), bucket in groups.items():
            cand, final_probs, g = self.gen._final_probs_for_context_cached(self.prep, w1, w2, w3)

            for s in bucket:
                nxt = self._sample_fuzzy(cand, final_probs, g)
                s.tokens_out.append(nxt)
                if nxt.isalpha():
                    s.alpha_count += 1
                s.w1, s.w2, s.w3 = s.w2, s.w3, nxt

                if s.alpha_count >= s.max_steps:
                    s.done = True
                elif nxt in s.stop_tokens and s.alpha_count > s.min_alpha:
                    s.done = True


# ----------------------------
# SG Runtime
# ----------------------------

class SGPrompt:
    def __init__(self, text: str = ""):
        self.text = str(text)
    def __iadd__(self, other):
        self.text += str(other)
        return self

class SGContext:
    def __init__(self, corpus_text: str, generator: NeuroSymbolicGraphGenerator, seed: int = 7):
        self.corpus_text = normalize(corpus_text)
        self.generator = generator
        self.seed = int(seed)
        self.prepared = None
    def ensure_prepared(self):
        if self.prepared is None:
            self.prepared = self.generator.prepare_corpus(self.corpus_text)
    def clone(self, seed_offset: int):
        ctx = SGContext(self.corpus_text, self.generator, self.seed + int(seed_offset))
        ctx.prepared = self.prepared
        return ctx

def sg_gen_batched(ctxs, prompts, max_tokens=240, seed_offsets=None, stop_at_punc=True):
    if not ctxs:
        return []
    gen = ctxs[0].generator
    ctxs[0].ensure_prepared()
    prep = ctxs[0].prepared
    rng = np.random.default_rng(int(ctxs[0].seed))

    streams = []
    for i, (ctx, prompt) in enumerate(zip(ctxs, prompts)):
        off = seed_offsets[i] if seed_offsets else i
        local_seed = int(ctx.seed + off)
        seed_words = basic_tokenize(prompt.text)
        w1, w2, w3 = gen._pick_initial_context(prep.lm, seed_words)
        streams.append(DecodeStream(
            i, [w1, w2, w3], w1, w2, w3,
            max_steps=max_tokens,
            min_alpha=max_tokens//2 if stop_at_punc else 99999
        ))

    decoder = ContinuousBatchDecoder(gen, prep, rng, token_budget_per_round=64)
    for _ in range(max_tokens * 2):
        if all(s.done for s in streams):
            break
        decoder.step_round(streams)

    return [detokenize(s.tokens_out[3:]) for s in streams]

def sg_fork(ctx, prompt, n):
    ctx.ensure_prepared()
    return [(ctx.clone(1000+i), SGPrompt(prompt.text)) for i in range(n)]

def sg_join(prompts, joiner="\n\n"):
    return SGPrompt(joiner.join(p.text for p in prompts))


# ----------------------------
# HF Dataset Loader
# ----------------------------

_PREFER_TEXT_COLS = ["text", "content", "article", "body", "markdown", "md", "html", "raw"]

def _pick_text_column(ds_split) -> str:
    feats = ds_split.features
    string_cols = [k for k, v in feats.items() if isinstance(v, Value) and v.dtype == "string"]
    if not string_cols:
        raise ValueError(f"No top-level string columns found. features={feats}")

    for pref in _PREFER_TEXT_COLS:
        for c in string_cols:
            if c.lower() == pref:
                return c
    for pref in _PREFER_TEXT_COLS:
        for c in string_cols:
            if pref in c.lower():
                return c
    return string_cols[0]

def load_hf_corpus(
    dataset_name: str,
    split_name: str = "text",
    max_rows: int = 0,
    joiner: str = "\n\n",
    cache_dir: str = ".hf_cache"
) -> Tuple[str, Dict[str, Any]]:
    
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    use_split = split_name if split_name in ds else next(iter(ds.keys()))
    split = ds[use_split]
    text_col = _pick_text_column(split)

    n_total = len(split)
    if isinstance(max_rows, float):
        max_rows = int(max_rows)
    if max_rows and max_rows > 0:
        n_take = min(int(max_rows), n_total)
        split = split.select(range(n_take))
    else:
        n_take = n_total

    def _norm(s: str) -> str:
        return normalize(s)

    parts = []
    for row in split:
        v = row.get(text_col, None)
        if isinstance(v, str):
            v = _norm(v)
            if v:
                parts.append(v)

    raw_corpus = joiner.join(parts)
    
    # Analyze injection stats
    cog_corpus = inject_cognitive_tokens(raw_corpus)
    n_prob = len(re.findall(re.escape("[PROBLEM]"), cog_corpus))
    n_sol = len(re.findall(re.escape("[SOLUTION]"), cog_corpus))

    meta = {
        "dataset": dataset_name,
        "split": use_split,
        "text_col": text_col,
        "rows_used": n_take,
        "chars": len(cog_corpus),
        "problems_found": n_prob,
        "solutions_found": n_sol
    }
    return cog_corpus, meta


# ----------------------------
# Main program function
# ----------------------------

def run_program(
    infile,
    use_hf: bool,
    hf_dataset: str,
    hf_split: str,
    hf_max_rows: int,
    n_take: int,
    seed: int,
    steer: float,
    focus: float,
    takeaway_prompt: str,
    summary_prompt: str,
    pairwise_strength: float = 0.4,
):
    if use_hf:
        corpus_text, meta = load_hf_corpus(
            dataset_name=str(hf_dataset).strip(),
            split_name=str(hf_split).strip(),
            max_rows=int(hf_max_rows) if hf_max_rows else 0,
        )
        header = (
            f"[HF DATASET] {meta['dataset']} (split: {meta['split']})\n"
            f"Rows: {meta['rows_used']} | Chars: {meta['chars']}\n"
            f"Cognitive Tokens: [PROBLEM]: {meta['problems_found']}, [SOLUTION]: {meta['solutions_found']}\n"
            f"Chinese Room Pairwise Symbolics: ENABLED (Uniform Artistic Values)\n"
            f"{'-'*40}\n\n"
        )
    else:
        path = _resolve_gradio_file_to_path(infile)
        raw_text = load_text(path)
        corpus_text = inject_cognitive_tokens(raw_text)
        n_prob = len(re.findall(re.escape("[PROBLEM]"), corpus_text))
        n_sol = len(re.findall(re.escape("[SOLUTION]"), corpus_text))
        
        header = (
            f"[FILE] {Path(path).name}\n"
            f"Chars: {len(corpus_text)}\n"
            f"Cognitive Tokens: [PROBLEM]: {n_prob}, [SOLUTION]: {n_sol}\n"
            f"Chinese Room Pairwise Symbolics: ENABLED (Uniform Artistic Values)\n"
            f"{'-'*40}\n\n"
        )

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        radix_cache_items=30000,
        pairwise_strength=float(pairwise_strength)
    )
    ctx = SGContext(corpus_text, gen, seed=int(seed))
    ctx.ensure_prepared()

    # Display Chinese Room symbolic pair statistics
    imagery_engine = ctx.prepared.state.imagery_engine
    header += (
        f"[CHINESE ROOM STATISTICS]\n"
        f"Pairwise Symbolic Cache: {len(imagery_engine.pair_cache)} pairs\n"
        f"Global Aesthetic Means:\n"
        f"  Harmony: {imagery_engine.global_aesthetics['harmony_mean']:.3f}\n"
        f"  Density: {imagery_engine.global_aesthetics['density_mean']:.3f}\n"
        f"  Momentum: {imagery_engine.global_aesthetics['momentum_mean']:.3f}\n"
        f"  Resonance: {imagery_engine.global_aesthetics['resonance_mean']:.3f}\n"
        f"Aesthetic Flow: {ctx.prepared.state.semantic_graph.get_aesthetic_flow():.3f}\n"
        f"{'-'*40}\n\n"
    )

    root = SGPrompt(str(takeaway_prompt).strip() + "\n\n")
    branches = sg_fork(ctx, root, n=int(n_take))
    branch_ctxs, branch_prompts = zip(*branches)
    branch_ctxs = list(branch_ctxs)
    branch_prompts = list(branch_prompts)

    for i, bp in enumerate(branch_prompts):
        bp += f"[Takeaway {i+1}] "

    take_texts = sg_gen_batched(branch_ctxs, branch_prompts, max_tokens=620, stop_at_punc=True)

    for i, txt in enumerate(take_texts):
        branch_prompts[i] += txt

    merged = sg_join(branch_prompts, joiner="\n\n")
    final_prompt = SGPrompt(summary_prompt.replace("{joined_takeaways}", merged.text))
    final_text = sg_gen_batched([ctx], [final_prompt], max_tokens=460)[0]

    return header + final_prompt.text + " " + final_text


# ----------------------------
# Gradio UI
# ----------------------------

def build_app():
    with gr.Blocks(title="Neurosymbolic V4.0 - Chinese Room Pairwise Symbolics") as demo:
        gr.Markdown(
            "# Neurosymbolic V4.0: Chinese Room Pairwise Symbolic Imagery\n"
            "**Core Philosophy:** Following Searle's Chinese Room argument, this system performs "
            "purely syntactic transformations on symbolic pairs without claiming semantic understanding. "
            "All artistic values are uniformly derived from structural properties.\n\n"
            "**Pairwise Symbolics:** Every token bigram receives uniform aesthetic attributes "
            "(Harmony, Density, Momentum, Resonance) via deterministic rule-based computation.\n\n"
            "**Uniform Treatment:** No symbol receives special treatment - all transformations "
            "are mathematical and consistent."
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face dataset", value=True)

                hf_dataset = gr.Textbox(
                    label="HF dataset name",
                    value="AiresPucrs/stanford-encyclopedia-philosophy",
                    interactive=True,
                )
                hf_split = gr.Textbox(label="HF split (optional)", value="text", interactive=True)
                hf_max_rows = gr.Slider(0, 20000, value=2000, step=100, label="HF max rows (0 = all)")

                infile = gr.File(label="Input File (txt/md) if not using HF", file_types=[".txt", ".md"])

            with gr.Column(scale=2):
                out_txt = gr.Textbox(label="Output", lines=24)

        with gr.Row():
            n_take = gr.Slider(1, 10, value=5, label="Parallel Batches")
            seed = gr.Number(value=42, label="Seed")

        with gr.Row():
            steer = gr.Slider(0, 5, value=1.35, label="Base Steer")
            focus = gr.Slider(0, 1, value=0.5, label="Focus")
            pairwise = gr.Slider(0, 2, value=0.4, label="Pairwise Strength", 
                                info="Weight for uniform symbolic aesthetics")

        p_takeaway = gr.Textbox(label="Prefix", value="", lines=2)
        p_summary = gr.Textbox(
            label="Summary Prompt",
            value="explain this?",
            lines=4
        )

        btn = gr.Button("Run Chinese Room Generator", variant="primary")
        btn.click(
            run_program,
            inputs=[infile, use_hf, hf_dataset, hf_split, hf_max_rows, n_take, seed, 
                   steer, focus, p_takeaway, p_summary, pairwise],
            outputs=out_txt
        )
        
        gr.Markdown(
            "---\n"
            "**Chinese Room Compliance Notes:**\n"
            "- No semantic understanding claimed - pure symbol manipulation\n"
            "- All aesthetic values computed via deterministic rules\n"
            "- Uniform treatment: edit distance, character count, hash-based pseudo-randomness\n"
            "- Pairwise symbolic imagery provides artistic coherence without meaning"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()