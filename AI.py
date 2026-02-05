#!/usr/bin/env python3
"""
Neurosymbolic Generator (Patched) — Separate Train / Generate

Fixes from your pasted script + tracebacks:
- basictokenize() missing -> alias to basictokenizetext(). [file:1]
- sgfork() broken list-comprehension -> fixed and returns list of (ctx,prompt). [file:1]
- backward() crash: "tensor does not require grad" -> make trainable params and ensure
  training path returns non-detached probabilities (no caching in train). [file:1]
- Fuzzy gain g sometimes 0 due to mftrap endpoint handling when a==b or c==d -> fixed.
- "No inference points found" -> robust sentence splitting + adaptive filters + fallback.
- Gradio UI: separate Train button and Generate button (not combined). [file:1]
"""

import re
import math
import random
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gradio as gr
from datasets import load_dataset, Value


# =============================================================================
# Utilities
# =============================================================================

STOPWORDS = set(
    "a an and are as at be by for from has have he her hers him his i in is it its me my "
    "of on or our ours she so that the their them they this to was we were what when where "
    "which who will with you your yours".split()
)

PUNCT = set(list(".,!?;:"))


def normalizetext(text: str) -> str:
    text = (text or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{3,}", "...", text)
    return text.strip()


def basictokenizetext(text: str) -> List[str]:
    text = (text or "").replace("_", " ")
    # words, alnum-hyphen, punctuation
    tokens = re.findall(r"[A-Za-z]+|[A-Za-z0-9-]+|[.,!?;:]", text)
    out: List[str] = []
    for t in tokens:
        if re.match(r"^[A-Za-z]", t):
            out.append(t.lower())
        else:
            out.append(t)
    return out


# Compatibility alias (your original code calls basictokenize everywhere). [file:1]
def basictokenize(text: str) -> List[str]:
    return basictokenizetext(text)


def detokenizetokens(tokens: List[str]) -> str:
    toks = tokens or []
    out: List[str] = []
    for t in toks:
        if t in PUNCT:
            if out:
                out[-1] = out[-1] + t
            else:
                out.append(t)
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s([.,!?;:])", r"\1", s)
    # sentence-case after .,!,?
    s = re.sub(r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), s)
    return s.strip()


def detokenize(tokens: List[str]) -> str:
    return detokenizetokens(tokens)


def loadtext(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError("Unsupported file extension; use .txt or .md")


def puretfidfdocs(docs: List[str], maxfeatures: int = 8000) -> Tuple[np.ndarray, List[str]]:
    allwords = set()
    for doc in docs:
        allwords.update(re.findall(r"\w+", (doc or "").lower()))
    vocab = list(allwords)[: maxfeatures]
    wordtoidx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)

    for i, doc in enumerate(docs):
        wc: Dict[str, int] = {}
        for w in re.findall(r"\w+", (doc or "").lower()):
            wc[w] = wc.get(w, 0) + 1
        if not wc:
            continue
        uniq = max(1, len(wc))
        for w, c in wc.items():
            j = wordtoidx.get(w, None)
            if j is None:
                continue
            tf = c / uniq
            df = sum(1 for d in docs if w in (d or "").lower())
            idf = math.log(len(docs) / (1 + df))
            X[i, j] = tf * idf
    return X, vocab


def puretruncatedsvd(X: np.ndarray, ncomponents: int, randomstate: int = 42) -> Any:
    np.random.seed(int(randomstate))
    m, n = X.shape
    k = int(min(ncomponents, min(m, n)))
    if k <= 1:
        return type("SVD", (), {"components": np.zeros((0, n), dtype=np.float32)})

    Q = np.random.randn(n, k).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    for _ in range(8):
        B = X.T @ X @ Q
        Q, _ = np.linalg.qr(B)
    B = X @ Q
    _U, _S, Vt = np.linalg.svd(B, full_matrices=False)
    comps = Vt[:k].astype(np.float32)
    return type("SVD", (), {"components": comps})


def tokenclasstok(tok: str) -> str:
    if tok in PUNCT:
        return "PUNC"
    t = (tok or "").lower()
    if not re.match(r"^[a-z]", t):
        return "OTHER"
    L = len(t)
    return "S" if L <= 3 else ("M" if L <= 7 else "L")


# =============================================================================
# Graph signature
# =============================================================================

@dataclass
class SimpleGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, Dict[str, Any]]]

    @classmethod
    def fromtokensequence(cls, tokens: List[str], maxnodes: int = 220):
        toks = (tokens or [])[: maxnodes]
        nodes = [{"id": i, "cls": tokenclasstok(t)} for i, t in enumerate(toks)]
        edges: List[Tuple[int, int, Dict[str, Any]]] = []
        for i in range(len(toks) - 1):
            edges.append((i, i + 1, {"rel": "adj"}))
        for i in range(len(toks) - 2):
            edges.append((i, i + 2, {"rel": "skip"}))
        return cls(nodes, edges)

    def degreehistogram(self, maxbins: int = 16) -> np.ndarray:
        degrees = np.zeros((maxbins,), dtype=np.int32)
        nodedeg = {node["id"]: 0 for node in self.nodes}
        for u, v, _ in self.edges:
            nodedeg[u] += 1
            nodedeg[v] += 1
        for d in nodedeg.values():
            if 0 <= d < maxbins:
                degrees[d] += 1
        return degrees

    def weisfeilerlehmanhash(self, iterations: int = 3, digestsize: int = 16) -> str:
        labels = {node["id"]: node["cls"] for node in self.nodes}
        adj = {node["id"]: [] for node in self.nodes}
        for u, v, _ in self.edges:
            adj[u].append(v)
            adj[v].append(u)

        for _ in range(int(iterations)):
            newlabels = {}
            for nodeid in labels:
                neighbors = sorted(labels[n] for n in adj[nodeid])
                combined = (labels[nodeid], tuple(neighbors))
                newhash = hash(combined) % (10 ** int(digestsize))
                newlabels[nodeid] = f"{labels[nodeid]}{newhash}"
            labels = newlabels

        finalhash = sum(hash(k) + hash(labels[k]) for k in labels) % (10 ** int(digestsize))
        return f"{finalhash:0{int(digestsize)}}"

    def automorphismestimate(self, maxcount: int = 150) -> int:
        counts: Dict[str, int] = {}
        for node in self.nodes:
            c = node["cls"]
            counts[c] = counts.get(c, 0) + 1
        prod = 1
        for v in counts.values():
            prod *= v
        return min(int(maxcount), int(prod))


def graphsignature(G: SimpleGraph) -> Dict[str, object]:
    return {
        "deghist": G.degreehistogram(),
        "wl": G.weisfeilerlehmanhash(),
        "autest": G.automorphismestimate(),
    }


# =============================================================================
# Fuzzy logic (FIXED membership endpoints)
# =============================================================================

def mftrix(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    # triangle: 0 at <=a and >=c, 1 at b
    x = x.to(dtype=torch.float32)
    y = torch.zeros_like(x)

    # rising edge (a->b)
    if b > a:
        y = torch.where((x > a) & (x < b), (x - a) / (b - a), y)

    # falling edge (b->c)
    if c > b:
        y = torch.where((x >= b) & (x < c), torch.maximum(y, (c - x) / (c - b)), y)

    # peak
    y = torch.where(x == b, torch.ones_like(x), y)
    return y.clamp(0.0, 1.0)


def mftrap(x: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    # trapezoid: 0 <=a, rises to 1 at b, plateau b..c, falls to 0 at d
    # handles shoulders a==b and/or c==d correctly (important for your controller).
    x = x.to(dtype=torch.float32)
    y = torch.zeros_like(x)

    # plateau (inclusive)
    y = torch.where((x >= b) & (x <= c), torch.ones_like(x), y)

    # rising edge (a->b)
    if b > a:
        y = torch.where((x > a) & (x < b), torch.maximum(y, (x - a) / (b - a)), y)
    else:
        # left shoulder (a==b): jump to 1 at x>=b
        y = torch.where(x >= b, torch.maximum(y, torch.ones_like(x)), y)

    # falling edge (c->d)
    if d > c:
        y = torch.where((x > c) & (x < d), torch.maximum(y, (d - x) / (d - c)), y)
    else:
        # right shoulder (c==d): stay 1 until c, then 0 after d; plateau already set
        y = y

    # outside support
    y = torch.where(x <= a, torch.zeros_like(x), y)
    y = torch.where(x >= d, torch.zeros_like(x), y)
    return y.clamp(0.0, 1.0)


def tnormprod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def snormmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


class FuzzyWeightController(nn.Module):
    def __init__(self):
        super().__init__()
        self.zlow = 0.20
        self.zmid = 0.55
        self.zhigh = 0.95

    def forward(self, entropy01: torch.Tensor, peak01: torch.Tensor, boost01: torch.Tensor) -> torch.Tensor:
        e = entropy01.clamp(0, 1)
        p = peak01.clamp(0, 1)
        b = boost01.clamp(0, 1)

        elow = mftrap(e, 0.0, 0.0, 0.25, 0.45)
        emid = mftrix(e, 0.25, 0.50, 0.75)
        ehigh = mftrap(e, 0.55, 0.75, 1.0, 1.0)

        plow = mftrap(p, 0.0, 0.0, 0.20, 0.40)
        pmid = mftrix(p, 0.25, 0.50, 0.75)
        phigh = mftrap(p, 0.60, 0.80, 1.0, 1.0)

        blow = mftrap(b, 0.0, 0.0, 0.20, 0.45)
        bmid = mftrix(b, 0.25, 0.50, 0.75)
        bhigh = mftrap(b, 0.55, 0.80, 1.0, 1.0)

        w1 = tnormprod(ehigh, plow)   # explore
        w2 = tnormprod(emid, bmid)    # balanced
        w3 = snormmax(phigh, bhigh)   # confident/steered
        w4 = tnormprod(elow, pmid)    # stable

        Z = torch.tensor([self.zhigh, self.zmid, self.zlow, self.zlow], device=e.device, dtype=torch.float32)
        W = torch.stack([w1, w2, w3, w4]).to(dtype=torch.float32).clamp(min=0.0)
        g = (W @ Z) / (W.sum() + 1e-12)
        return g.clamp(0.0, 1.0)


class LateralInhibition(nn.Module):
    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        mod = F.conv1d(x, self.kernel, padding=self.pad)
        out = F.relu(x + self.strength * mod)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)


# =============================================================================
# LM
# =============================================================================

class QuadgramLM:
    def __init__(self, addk: float = 0.25):
        self.addk = float(addk)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self.quad.clear()
        self.total = 0

        toks = tokens or []
        for t in toks:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1

        for i in range(len(toks) - 1):
            k = (toks[i], toks[i + 1])
            self.bi[k] = self.bi.get(k, 0) + 1

        for i in range(len(toks) - 2):
            k = (toks[i], toks[i + 1], toks[i + 2])
            self.tri[k] = self.tri.get(k, 0) + 1

        for i in range(len(toks) - 3):
            k = (toks[i], toks[i + 1], toks[i + 2], toks[i + 3])
            self.quad[k] = self.quad.get(k, 0) + 1

        self.vocab = list(self.uni.keys())

    def nextdistribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        cont: List[str] = []
        for (a, b, c, d), _cnt in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c), _cnt in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b), _cnt in self.bi.items():
                if a == w3:
                    cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]

        seen = set()
        cand: List[str] = []
        for w in cont:
            if w not in seen:
                seen.add(w)
                cand.append(w)
        cand = cand[:500]

        V = len(self.vocab) + 1
        addk = self.addk

        def getprob(w4: str) -> float:
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0:
                return (c1234 + addk) / (c123 + addk * V)

            c12 = self.bi.get((w2, w3), 0)
            c123tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0:
                return (c123tri + addk) / (c12 + addk * V)

            c1 = self.uni.get(w3, 0)
            c12bi = self.bi.get((w3, w4), 0)
            if c1 > 0:
                return (c12bi + addk) / (c1 + addk * V)

            return (self.uni.get(w4, 0) + addk) / (self.total + addk * V)

        probs = torch.tensor([getprob(w) for w in cand], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# =============================================================================
# State objects
# =============================================================================

@dataclass
class Nodelet:
    idx: int
    topterms: List[Tuple[str, float]]
    energy: float
    narrative: str


@dataclass
class ModelState:
    nodelets: List[Nodelet]
    vocab100: List[str]
    bindingW: torch.Tensor
    barprobs: torch.Tensor
    tokenboost: Dict[str, float]
    pillarweights: torch.Tensor
    geometricbias: torch.Tensor
    semanticgraph: SimpleGraph
    lmgraph: Any


@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    lm: QuadgramLM
    state: ModelState
    refsig: Dict[str, object]


class RadixLRUCache:
    def __init__(self, maxitems: int = 25000):
        self.maxitems = int(max(256, maxitems))
        self.od = OrderedDict()

    def get(self, key):
        v = self.od.get(key, None)
        if v is not None:
            self.od.move_to_end(key)
        return v

    def put(self, key, value):
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.maxitems:
            self.od.popitem(last=False)

    def clear(self):
        self.od.clear()


# =============================================================================
# Main generator (now actually trainable)
# =============================================================================

class NeuroSymbolicGraphGenerator(nn.Module):
    def __init__(
        self,
        nodeletsn=10,
        barsn=100,
        svdrandomstate=7,
        softmaxtemp=0.85,
        steerstrength=1.35,
        lmaddk=0.25,
        focusstrength=0.5,
        radixcacheitems=25000,
    ):
        super().__init__()
        self.nodeletsn = int(nodeletsn)
        self.barsn = int(barsn)
        self.svdrandomstate = int(svdrandomstate)
        self.softmaxtemp = float(softmaxtemp)
        self.lmaddk = float(lmaddk)
        self.basesteer = float(steerstrength)
        self.basetemp = float(softmaxtemp)

        self.focuslayer = LateralInhibition(strength=float(focusstrength))
        self.fuzzyctl = FuzzyWeightController()

        self.cacheversion = 0
        self.radixcache = RadixLRUCache(maxitems=int(radixcacheitems))

        # Trainable scalars (prevents backward() crash and lets "Train" do something).
        self.steer_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.temp_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.boost_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def bumpcacheversion(self):
        self.cacheversion += 1
        self.radixcache.clear()

    def pickinitialcontext(self, lm: QuadgramLM, seedwords: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in (seedwords or []) if re.match(r"^[a-z0-9-]+$", t)]
        if len(sw) >= 3:
            return sw[-3], sw[-2], sw[-1]
        if len(sw) == 2:
            return sw[0], sw[1], sw[1]
        if len(sw) == 1:
            return sw[0], sw[0], sw[0]
        seedtok = lm.vocab[0] if lm.vocab else "the"
        return seedtok, seedtok, seedtok

    def buildstate(self, text: str) -> ModelState:
        text = normalizetext(text)
        docs = re.split(r"\.\s+", text)[:500]
        X, vocab = puretfidfdocs(docs, maxfeatures=8000)

        if X.size == 0 or not vocab:
            vocab100 = ["the", "is", "a"]
            probs = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)
            return ModelState([], vocab100, torch.zeros(0, 3), probs, {}, torch.zeros_like(probs), torch.zeros_like(probs), SimpleGraph([], []), None)

        topidx = np.argsort(-X.sum(axis=0))[: self.barsn]
        vocab100 = [vocab[i] for i in topidx]
        Xsvd = X[:, topidx]

        nrows, ncols = Xsvd.shape
        maxrank = min(nrows, ncols)
        k = 1 if maxrank <= 1 else min(self.nodeletsn, maxrank, 10)

        svd = puretruncatedsvd(Xsvd, ncomponents=k, randomstate=self.svdrandomstate)
        nodelets: List[Nodelet] = []
        for i, comp in enumerate(svd.components):
            terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))], key=lambda x: -abs(x[1]))[:10]
            eng = float(np.linalg.norm(comp))
            nodelets.append(Nodelet(i, terms, eng, f"Nodelet {i}"))

        W = torch.tensor(svd.components, dtype=torch.float32)
        W = F.relu(W)
        if W.numel() > 0:
            W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)

        energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32)
        if energies.numel() > 0:
            energies = energies / (energies.max() + 1e-12)

        if W.numel() == 0 or energies.numel() == 0:
            probs = torch.ones(len(vocab100), dtype=torch.float32)
            probs = probs / (probs.sum() + 1e-12)
        else:
            logits = energies.view(-1, 1) @ W.sum(dim=0).view(1, -1)
            probs = F.softmax(logits / max(self.softmaxtemp, 1e-6), dim=-1)
            probs = self.focuslayer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)

        tokenboost: Dict[str, float] = {}
        for w, p in zip(vocab100, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) >= 2 and subw not in STOPWORDS:
                    tokenboost[subw] = max(tokenboost.get(subw, 0.0), math.log(p + 1e-12) * 5.0)

        return ModelState(nodelets, vocab100, W, probs, tokenboost, torch.zeros_like(probs), torch.zeros_like(probs), SimpleGraph([], []), None)

    def preparecorpus(self, text: str) -> PreparedCorpus:
        text = normalizetext(text)
        state = self.buildstate(text)
        tokens = basictokenize(text)
        lm = QuadgramLM(self.lmaddk)
        lm.ingest(tokens)
        G = SimpleGraph.fromtokensequence(tokens, maxnodes=220)
        refsig = graphsignature(G)
        return PreparedCorpus(text, tokens, lm, state, refsig)

    def finalprobsforcontextcached(self, prep: PreparedCorpus, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor, float]:
        key = (int(self.cacheversion), str(w1), str(w2), str(w3))

        # IMPORTANT: cache only in eval mode; training must not detach/cached.
        if not self.training:
            cached = self.radixcache.get(key)
            if cached is not None:
                return cached

        cand, baseprobs = prep.lm.nextdistribution(w1, w2, w3)
        if not cand:
            cand = prep.lm.vocab if prep.lm.vocab else ["the", "is", "a"]
            basep = torch.ones(len(cand), dtype=torch.float32)
            basep = basep / max(1, len(cand))
        else:
            basep = baseprobs.clone().to(dtype=torch.float32).view(-1)
            basep = basep / (basep.sum() + 1e-12)
            basep = self.focuslayer(basep.view(1, 1, -1)).squeeze(0).squeeze(0)

        val = basep.clamp(min=1e-12)
        H = -torch.sum(basep * torch.log(val))
        V = float(basep.numel())
        denom = max(1e-9, math.log(max(2.0, V)))
        entropy01 = (H / denom).clamp(0.0, 1.0)
        peak01 = basep.max().clamp(0.0, 1.0)

        boosts = torch.tensor([prep.state.tokenboost.get(w, 0.0) for w in cand], dtype=torch.float32).view(-1)
        boost01 = (torch.tanh(boosts.abs()).mean() / 3.0).clamp(0.0, 1.0)

        g = self.fuzzyctl(entropy01, peak01, boost01)  # tensor

        # Trainable scaling
        steer_scale = torch.clamp(self.steer_scale, 0.0, 10.0)
        temp_scale = torch.clamp(self.temp_scale, 0.05, 10.0)
        boost_scale = torch.clamp(self.boost_scale, 0.0, 10.0)

        effectivesteer = (self.basesteer * steer_scale) * g
        effectivetemp = (self.basetemp * temp_scale) * (1.2 - 0.7 * g)

        potentials = torch.log(basep.clamp(min=1e-12)) + effectivesteer * (boost_scale * boosts)
        potentials = potentials / effectivetemp.clamp(min=1e-6)
        finalprobs = F.softmax(potentials, dim=-1)

        g_float = float(g.detach().item())

        if not self.training:
            result = (cand, finalprobs.detach(), g_float)
            self.radixcache.put(key, result)
            return result

        return (cand, finalprobs, g_float)


# =============================================================================
# Dataset + sentence splitting + inference point extraction
# =============================================================================

@dataclass
class TextSample:
    text: str
    tokens: List[str]
    split: str


class NeurosymbolicDataset(Dataset):
    def __init__(self, texts: List[str], split_ratio: float = 0.8, max_samples: int = 100000):
        self.samples: List[TextSample] = []
        texts = (texts or [])[: int(max_samples)]
        split_idx = int(len(texts) * float(split_ratio))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        for text_list, split_name in [(train_texts, "train"), (val_texts, "val")]:
            for t in text_list:
                toks = basictokenize(t)
                # less strict than original (>10 was too aggressive for some corpora)
                if len(toks) >= 6:
                    self.samples.append(TextSample(t, toks, split_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def splitsentences(text: str, maxsentences: int = 20000) -> List[str]:
    t = normalizetext(text)
    if not t:
        return []
    # robust: split on punctuation or explicit separators; allow missing whitespace
    parts = re.split(r"[.?!]+|(?:\s-\s)|(?:\s•\s)|\n+", t)
    out: List[str] = []
    for s in parts:
        s = s.strip()
        if s:
            out.append(s)
        if len(out) >= int(maxsentences):
            break
    return out


def scoresentence(sentence: str, prep: PreparedCorpus) -> Tuple[float, Dict[str, float]]:
    toks = basictokenize(sentence)
    if not toks:
        return -1e9, {}

    boostvals = [prep.state.tokenboost.get(t, 0.0) for t in toks if t.isalpha()]
    boostsum = float(sum(v for v in boostvals if v > 0))
    boostuniq = float(sum(1 for t in toks if prep.state.tokenboost.get(t, 0.0) > 0))

    Gs = SimpleGraph.fromtokensequence(toks, maxnodes=220)
    sentsig = graphsignature(Gs)
    structsim = abs(float(np.sum(prep.refsig["deghist"])) - float(np.sum(sentsig["deghist"])))
    structsim = float(max(0.0, 1.0 - 0.5 * structsim))

    nalpha = sum(1 for t in toks if t.isalpha())
    ntotal = len(toks)
    alpharatio = nalpha / max(1, ntotal)

    score = (1.00 * boostsum + 0.65 * boostuniq + 1.20 * structsim + 0.40 * alpharatio - 0.01 * ntotal)
    meta = {"boostsum": boostsum, "boostuniq": boostuniq, "len": float(ntotal), "struct": float(structsim), "alpharatio": float(alpharatio)}
    return float(score), meta


def extractpertinentpoints(
    sourcetext: str,
    prep: PreparedCorpus,
    kpoints: int = 12,
    minwords: int = 3,
    maxwords: int = 140,
    maxscansentences: int = 20000,
) -> List[str]:
    sents = splitsentences(sourcetext, maxsentences=int(maxscansentences))
    if not sents:
        return []

    def count_words(s: str) -> int:
        return len(re.findall(r"[A-Za-z]+", s))

    scored: List[Tuple[float, str]] = []
    for s in sents:
        w = count_words(s)
        if w < int(minwords) or w > int(maxwords):
            continue
        sc, _meta = scoresentence(s, prep)
        scored.append((sc, s))

    # Adaptive fallback if nothing passed the filters:
    if not scored:
        # loosen automatically
        for s in sents:
            w = count_words(s)
            if w >= 2:
                sc, _meta = scoresentence(s, prep)
                scored.append((sc, s))
            if len(scored) >= int(kpoints) * 5:
                break

    if not scored:
        # final fallback: just return first k non-empty sentences
        return [s.strip() for s in sents[: int(kpoints)] if s.strip()]

    scored.sort(key=lambda x: x[0], reverse=True)

    def keynorm(ss: str) -> str:
        toks = [t for t in basictokenize(ss) if t.isalpha() and t not in STOPWORDS][:80]
        return " ".join(sorted(set(toks)))

    seen = set()
    out: List[str] = []
    for _sc, s in scored[: int(kpoints) * 6]:
        sig = keynorm(s)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        out.append(s.strip())
        if len(out) >= int(kpoints):
            break
    return out


def formatpoints(points: List[str]) -> str:
    if not points:
        return "No inference points found (input text may be empty)."
    lines = ["Pertinent points of inference:"]
    for i, p in enumerate(points, 1):
        lines.append(f"{i}. {p}")
    return "\n".join(lines)


# =============================================================================
# Batched generation
# =============================================================================

@dataclass
class DecodeStream:
    streamid: int
    tokensout: List[str] = field(default_factory=list)
    w1: str = ""
    w2: str = ""
    w3: str = ""
    done: bool = False
    alphacount: int = 0
    maxsteps: int = 240
    stoptokens: set = field(default_factory=lambda: {".", "!", "?"})
    minalpha: int = 80


class ContinuousBatchDecoder:
    def __init__(self, gen: NeuroSymbolicGraphGenerator, prep: PreparedCorpus, rng: np.random.Generator, tokenbudgetperround: int = 64):
        self.gen = gen
        self.prep = prep
        self.rng = rng
        self.tokenbudgetperround = int(max(1, tokenbudgetperround))

    def samplefuzzy(self, cand: List[str], probs: torch.Tensor, g: float) -> str:
        p = probs.detach().cpu().numpy().astype(np.float64)
        p = p / float(p.sum() + 1e-12)
        mu = float(np.mean(p))
        std = float(np.std(p))
        cutoff = mu + (1.0 - float(g)) * std
        psparse = np.where(p > cutoff, p, 0.0)
        if float(psparse.sum()) < 1e-12:
            psparse = p
        psparse = psparse / float(psparse.sum() + 1e-12)
        return str(self.rng.choice(cand, p=psparse))

    def stepround(self, streams: List[DecodeStream]) -> None:
        active = [s for s in streams if not s.done]
        if not active:
            return

        active.sort(key=lambda s: (s.w1, s.w2, s.w3))
        active = active[: min(len(active), self.tokenbudgetperround)]

        buckets: Dict[Tuple[str, str, str], List[DecodeStream]] = {}
        for s in active:
            buckets.setdefault((s.w1, s.w2, s.w3), []).append(s)

        self.gen.eval()
        for (w1, w2, w3), bucket in buckets.items():
            cand, probs, g = self.gen.finalprobsforcontextcached(self.prep, w1, w2, w3)
            for s in bucket:
                nxt = self.samplefuzzy(cand, probs, g)
                s.tokensout.append(nxt)
                if nxt.isalpha():
                    s.alphacount += 1
                s.w1, s.w2, s.w3 = s.w2, s.w3, nxt

                if s.alphacount >= int(s.maxsteps):
                    s.done = True
                elif nxt in s.stoptokens and s.alphacount >= int(s.minalpha):
                    s.done = True


class SGPrompt:
    def __init__(self, text: str):
        self.text = str(text or "")


class SGContext:
    def __init__(self, corpustext: str, generator: NeuroSymbolicGraphGenerator, seed: int = 7):
        self.corpustext = normalizetext(corpustext)
        self.generator = generator
        self.seed = int(seed)
        self.prepared: Optional[PreparedCorpus] = None

    def ensureprepared(self):
        if self.prepared is None:
            self.prepared = self.generator.preparecorpus(self.corpustext)

    def clone(self, seedoffset: int) -> "SGContext":
        ctx = SGContext(self.corpustext, self.generator, self.seed + int(seedoffset))
        ctx.prepared = self.prepared
        return ctx


def sggenbatched(ctxs: List[SGContext], prompts: List[SGPrompt], maxtokens: int = 240) -> List[str]:
    if not ctxs or not prompts:
        return [""] * len(prompts)

    ctxs[0].ensureprepared()
    prep = ctxs[0].prepared
    if prep is None:
        return [""] * len(prompts)

    rng = np.random.default_rng(int(ctxs[0].seed))
    streams: List[DecodeStream] = []

    for i, (_ctx, prompt) in enumerate(zip(ctxs, prompts)):
        seedwords = basictokenize(prompt.text)
        w1, w2, w3 = ctxs[0].generator.pickinitialcontext(prep.lm, seedwords)
        streams.append(
            DecodeStream(
                i,
                w1=w1,
                w2=w2,
                w3=w3,
                maxsteps=int(maxtokens),
                minalpha=max(20, int(maxtokens) // 3),
            )
        )

    decoder = ContinuousBatchDecoder(ctxs[0].generator, prep, rng, tokenbudgetperround=64)
    for _ in range(max(1, int(maxtokens))):
        if all(s.done for s in streams):
            break
        decoder.stepround(streams)

    return [detokenize(s.tokensout) for s in streams]


def sgfork(ctx: SGContext, prompt: SGPrompt, n: int) -> List[Tuple[SGContext, SGPrompt]]:
    # Fixed comprehension (your original was syntactically invalid). [file:1]
    ctx.ensureprepared()
    return [(ctx.clone(1000 + i), SGPrompt(prompt.text)) for i in range(int(n))]


# =============================================================================
# HF loading helpers
# =============================================================================

PREFERTEXTCOLS = ["text", "content", "article", "body", "markdown", "md", "html", "raw"]


def picktextcolumndssplit(split):
    feats = split.features
    stringcols = [k for k, v in feats.items() if isinstance(v, Value) and v.dtype == "string"]
    if not stringcols:
        raise ValueError(f"No top-level string columns found. features={feats}")

    for pref in PREFERTEXTCOLS:
        for c in stringcols:
            if c.lower() == pref:
                return c
    for pref in PREFERTEXTCOLS:
        for c in stringcols:
            if pref in c.lower():
                return c
    return stringcols[0]


def loadhfcorpus(datasetname: str, splitname: str = "train", maxrows: int = 0, joiner: str = " ") -> Tuple[str, Dict[str, Any]]:
    ds = load_dataset(datasetname)
    usesplit = splitname if splitname in ds else next(iter(ds.keys()))
    split = ds[usesplit]
    textcol = picktextcolumndssplit(split)

    ntotal = len(split)
    maxrows = int(maxrows) if maxrows else 0
    if maxrows > 0:
        ntake = min(maxrows, ntotal)
        split = split.select(range(ntake))
    else:
        ntake = ntotal

    parts: List[str] = []
    for row in split:
        v = row.get(textcol, None)
        if isinstance(v, str):
            v = normalizetext(v)
            if v:
                parts.append(v)

    corpus = joiner.join(parts)
    meta = {
        "dataset": datasetname,
        "split": usesplit,
        "textcol": textcol,
        "rowsused": ntake,
        "rowsnonempty": len(parts),
        "chars": len(corpus),
        "columns": list(split.column_names),
    }
    return corpus, meta


def resolvegradiofiletopath(infile) -> str:
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


def _load_corpus(infile, usehf: bool, hfdataset: str, hfsplit: str, hfmaxrows: int) -> Tuple[str, str]:
    if usehf:
        corpustext, meta = loadhfcorpus(hfdataset.strip(), hfsplit.strip(), maxrows=int(hfmaxrows) if hfmaxrows else 0)
        header = (
            f"HF dataset {meta['dataset']} split {meta['split']} textcol {meta['textcol']} "
            f"rows_used {meta['rowsused']} nonempty {meta['rowsnonempty']} chars {meta['chars']}"
        )
        return corpustext, header
    path = resolvegradiofiletopath(infile)
    corpustext = loadtext(path)
    header = f"FILE {Path(path).name} chars {len(corpustext)}"
    return corpustext, header


# =============================================================================
# Train / Validate
# =============================================================================

def train_and_evaluate(
    infile=None,
    usehf=True,
    hfdataset="AiresPucrs/stanford-encyclopedia-philosophy",
    hfsplit="train",
    hfmaxrows=50000,
    seed=42,
    epochs=3,
    steer=1.35,
    focus=0.5,
    batch_size=32,
    lr=0.001,
) -> Tuple[str, NeuroSymbolicGraphGenerator, str, str]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))

    corpustext, header = _load_corpus(infile, bool(usehf), str(hfdataset), str(hfsplit), int(hfmaxrows))

    sents = splitsentences(corpustext, maxsentences=20000)
    dataset = NeurosymbolicDataset(sents, split_ratio=0.8, max_samples=100000)

    train_ds = [s for s in dataset.samples if s.split == "train"]
    val_ds = [s for s in dataset.samples if s.split == "val"]

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=lambda x: x)

    model = NeuroSymbolicGraphGenerator(steerstrength=float(steer), focusstrength=float(focus), radixcacheitems=30000)
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-5)

    def nll_from_probs(probs_1d: torch.Tensor, target_idx: int) -> torch.Tensor:
        logp = torch.log(probs_1d.clamp(min=1e-12)).unsqueeze(0)  # [1,C]
        target = torch.tensor([int(target_idx)], dtype=torch.long)
        return F.nll_loss(logp, target)

    # TRAIN
    model.train()
    train_losses: List[float] = []

    for ep in range(int(epochs)):
        epoch_loss = 0.0
        nb = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses: List[torch.Tensor] = []

            for sample in batch[:8]:
                try:
                    prep = model.preparecorpus(sample.text)
                    toks = sample.tokens[:10]
                    if len(toks) < 4:
                        continue
                    w1, w2, w3 = model.pickinitialcontext(prep.lm, toks[:3])
                    cand, probs, _g = model.finalprobsforcontextcached(prep, w1, w2, w3)
                    target = toks[3]
                    if target in cand:
                        losses.append(nll_from_probs(probs, cand.index(target)))
                except Exception:
                    continue

            if losses:
                batch_loss = torch.stack(losses).mean()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(batch_loss.detach().item())
                nb += 1

        train_losses.append(epoch_loss / max(1, nb))
        model.bumpcacheversion()

    # VALIDATE
    model.eval()
    vloss = 0.0
    vcount = 0
    last_g = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for sample in batch[:8]:
                try:
                    prep = model.preparecorpus(sample.text)
                    toks = sample.tokens[:10]
                    if len(toks) < 4:
                        continue
                    w1, w2, w3 = model.pickinitialcontext(prep.lm, toks[:3])
                    cand, probs, g = model.finalprobsforcontextcached(prep, w1, w2, w3)
                    last_g = float(g)
                    target = toks[3]
                    if target in cand:
                        vloss += float(nll_from_probs(probs, cand.index(target)).item())
                        vcount += 1
                except Exception:
                    continue

    avg_vloss = vloss / max(1, vcount)
    ppl = math.exp(avg_vloss) if vcount > 0 else 0.0

    # INFERENCE POINTS from validation corpus (larger slice to ensure content)
    val_corpus = " ".join(s.text for s in val_ds[:800])
    prep_val = model.preparecorpus(val_corpus)
    points = extractpertinentpoints(val_corpus, prep_val, kpoints=12)

    report = (
        f"{header}\n\n"
        f"TRAIN DONE: epochs={int(epochs)}\n"
        f"Training samples={len(train_ds)}  Validation samples={len(val_ds)}\n"
        f"Train loss={train_losses[-1] if train_losses else 0.0:.4f}\n\n"
        f"VAL: loss={avg_vloss:.4f} perplexity={ppl:.2f} scored_samples={vcount}\n"
        f"Fuzzy gain g={last_g:.3f}\n\n"
        f"{formatpoints(points)}\n\n"
        f"Learned: steer_scale={float(model.steer_scale.detach().item()):.4f} "
        f"temp_scale={float(model.temp_scale.detach().item()):.4f} "
        f"boost_scale={float(model.boost_scale.detach().item()):.4f}"
    )
    return report, model, corpustext, header


# =============================================================================
# Gradio (separate Train / Generate)
# =============================================================================

APP_STATE = {"model": None, "corpustext": "", "header": ""}


def ui_train(infile, usehf, hfdataset, hfsplit, hfmaxrows, seed, epochs, steer, focus, lr, batch_size):
    report, model, corpustext, header = train_and_evaluate(
        infile=infile,
        usehf=usehf,
        hfdataset=hfdataset,
        hfsplit=hfsplit,
        hfmaxrows=hfmaxrows,
        seed=seed,
        epochs=epochs,
        steer=steer,
        focus=focus,
        lr=lr,
        batch_size=batch_size,
    )
    APP_STATE["model"] = model
    APP_STATE["corpustext"] = corpustext
    APP_STATE["header"] = header
    return "OK: trained in memory. Now click Generate.", report


def ui_generate(prompt_text: str, nsamples: int, maxtokens: int, seed: int, use_trained_context: bool):
    prompt_text = str(prompt_text or "").strip()
    if not prompt_text:
        return "Enter a prompt."

    model = APP_STATE.get("model", None)
    corpustext = APP_STATE.get("corpustext", "")

    if model is None:
        model = NeuroSymbolicGraphGenerator()
        corpustext = corpustext or prompt_text

    context_text = corpustext if (bool(use_trained_context) and corpustext.strip()) else prompt_text

    ctx0 = SGContext(context_text, model, seed=int(seed))
    pairs = sgfork(ctx0, SGPrompt(prompt_text), int(nsamples))
    ctxs = [c for (c, _p) in pairs]
    prompts = [_p for (_c, _p) in pairs]
    outs = sggenbatched(ctxs, prompts, maxtokens=int(maxtokens))

    blocks: List[str] = []
    for i, o in enumerate(outs, 1):
        blocks.append(f"=== Sample {i} ===\n{o}")
    return "\n\n".join(blocks)


def buildapp():
    with gr.Blocks(title="Neurosymbolic — Separate Train / Generate") as demo:
        gr.Markdown("# Neurosymbolic (Patched) — Separate Train and Generate")

        status = gr.Textbox(label="Status", value="Idle", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Train")
                usehf = gr.Checkbox(label="Use Hugging Face dataset", value=True)
                hfdataset = gr.Textbox(label="HF dataset name", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hfsplit = gr.Textbox(label="HF split", value="train")
                hfmaxrows = gr.Slider(0, 200000, value=50000, step=500, label="HF max rows (0 = all; bigger = better points)")
                infile = gr.File(label="Input file (.txt/.md) if not using HF", file_types=[".txt", ".md"])

                seed = gr.Number(value=42, label="Seed")
                epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                steer = gr.Slider(0, 5, value=1.35, step=0.05, label="Steer strength")
                focus = gr.Slider(0, 1, value=0.5, step=0.05, label="Focus strength")
                lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning rate")
                batch_size = gr.Slider(8, 128, value=32, step=8, label="Batch size")

                btn_train = gr.Button("Train", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## Generate")
                prompt = gr.Textbox(label="Prompt", lines=5)
                nsamples = gr.Slider(1, 10, value=3, step=1, label="Samples")
                maxtokens = gr.Slider(20, 600, value=200, step=10, label="Max tokens")
                gseed = gr.Number(value=7, label="Generation seed")
                use_trained_context = gr.Checkbox(label="Use trained corpus as context", value=True)
                btn_gen = gr.Button("Generate", variant="secondary")

        train_out = gr.Textbox(label="Train report", lines=18, max_lines=60)
        gen_out = gr.Textbox(label="Generated text", lines=18, max_lines=80)

        btn_train.click(
            ui_train,
            inputs=[infile, usehf, hfdataset, hfsplit, hfmaxrows, seed, epochs, steer, focus, lr, batch_size],
            outputs=[status, train_out],
        )
        btn_gen.click(
            ui_generate,
            inputs=[prompt, nsamples, maxtokens, gseed, use_trained_context],
            outputs=[gen_out],
        )

    return demo


if __name__ == "__main__":
    demo = buildapp()
    demo.queue().launch(share=True)
