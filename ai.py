#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V6.2 - Vector Spotting Edition
Complete working version with all fixes applied.
"""
from __future__ import annotations
import re
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np
from scipy.optimize import curve_fit
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# =============================================================================
# 1. SYMBOLIC PAIR & ACTIVATOR
# =============================================================================

@dataclass
class SymbolicPair:
    token1: str
    token2: str
    harmony: float
    density: float
    momentum: float
    resonance: float

    @classmethod
    def fromtokens(cls, t1: str, t2: str) -> "SymbolicPair":
        lent1, lent2 = len(t1), len(t2)
        harmony = 1.0 - cls.editdistance(t1, t2) / max(lent1, lent2, 1)
        density = math.tanh((lent1 + lent2) / 20.0) if lent1 + lent2 > 0 else 0.0
        if lent1 and lent2:
            momentum = (lent2 - lent1) / (lent1 + lent2)
            momentum = max(min(momentum, 1.0), -2.0)
        else:
            momentum = 0.5
        pairstr = f"{t1}|{t2}"
        hashval = int(hashlib.md5(pairstr.encode()).hexdigest(), 16)
        resonance = (hashval % 10000) / 10000.0
        return cls(t1, t2, harmony, density, momentum, resonance)

    @staticmethod
    def editdistance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return SymbolicPair.editdistance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def aestheticvector(self) -> np.ndarray:
        return np.array([self.harmony, self.density, self.momentum, self.resonance], dtype=np.float32)


class NeuronalActivator(nn.Module):
    """Differentiable aesthetic weight predictor with FIXED dimension handling."""
    def __init__(self, vocabsize: int = 50000, embdim: int = 64, hidden: int = 96, posfourier: int = 16):
        super().__init__()
        self.vocabsize = int(vocabsize)
        self.embdim = int(embdim)
        self.num_posfourier = int(posfourier)  # FIXED: Renamed to avoid conflict with method
        self.emb = nn.Embedding(self.vocabsize, self.embdim)
        indim = 2 * self.embdim + 2 * self.num_posfourier
        self.mlp = nn.Sequential(
            nn.Linear(indim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("globalmean4", torch.full((4,), 0.5, dtype=torch.float32))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def hashtokentoid(tok: str, mod: int) -> int:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        return int(h % mod)

    def tokenids(self, toks: List[str], device: torch.device) -> torch.LongTensor:
        ids = [self.hashtokentoid(t, self.vocabsize) for t in toks]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def posfourier(self, x: torch.Tensor) -> torch.Tensor:
        """FIXED: Always returns 2D [B, 2*num_posfourier]."""
        if x.dim() == 0:
            x = x.view(1)
        freqs = torch.arange(1, self.num_posfourier + 1, device=x.device, dtype=x.dtype) * (2.0 * math.pi)
        ang = x.view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def predictvec4(self, t1ids: torch.LongTensor, t2ids: torch.LongTensor, xpos: torch.Tensor) -> torch.Tensor:
        """FIXED: Bulletproof dimension handling."""
        device = self.emb.weight.device
        
        # Ensure xpos is tensor
        if not isinstance(xpos, torch.Tensor):
            xpos = torch.as_tensor(xpos, dtype=torch.float32, device=device)
        else:
            xpos = xpos.to(device=device, dtype=torch.float32)
        
        # Make xpos at least 1D
        if xpos.dim() == 0:
            xpos = xpos.view(1)
        
        # Ensure ids are 1D
        t1ids = t1ids.to(device=device, dtype=torch.long).view(-1)
        t2ids = t2ids.to(device=device, dtype=torch.long).view(-1)
        
        # Handle broadcasting: 1 context vs many candidates
        if t1ids.shape[0] == 1 and t2ids.shape[0] > 1:
            t1ids = t1ids.expand(t2ids.shape[0])
        elif t2ids.shape[0] == 1 and t1ids.shape[0] > 1:
            t2ids = t2ids.expand(t1ids.shape[0])
        
        B = t1ids.shape[0]
        
        # Broadcast xpos to match batch size
        if xpos.shape[0] == 1 and B > 1:
            xpos = xpos.expand(B)
        elif xpos.shape[0] != B:
            # Truncate to min length
            m = min(B, xpos.shape[0])
            t1ids = t1ids[:m]
            t2ids = t2ids[:m]
            xpos = xpos[:m]
            B = m
        
        # Get embeddings [B, embdim]
        e1 = self.emb(t1ids)
        e2 = self.emb(t2ids)
        
        # Get positional features [B, 2*num_posfourier]
        pf = self.posfourier(xpos)
        
        # Safety check: ensure all 2D
        if pf.dim() == 1:
            pf = pf.unsqueeze(0)
        if e1.shape[0] != pf.shape[0]:
            raise RuntimeError(f"Batch mismatch: e1={e1.shape}, pf={pf.shape}")
        
        # Concatenate [B, 2*embdim + 2*num_posfourier]
        z = torch.cat([e1, e2, pf], dim=-1)
        vec4 = torch.sigmoid(self.mlp(z))
        return vec4

    def weightfromvec4(self, vec4: torch.Tensor) -> torch.Tensor:
        d = torch.linalg.norm(vec4 - self.globalmean4.view(1, 4), dim=-1)
        return torch.exp(-d)

    @torch.no_grad()
    def updateglobalmean(self, vec4all: torch.Tensor):
        self.globalmean4.copy_(vec4all.mean(dim=0).clamp(0.0, 1.0))

    def forwardweight(self, t1: str, t2list: List[str], xpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.emb.weight.device
        t1ids = self.tokenids([t1], device)
        t2ids = self.tokenids(t2list, device)
        # Expand t1 to match t2
        t1ids = t1ids.expand(len(t2list))
        vec4 = self.predictvec4(t1ids, t2ids, xpos)
        w = self.weightfromvec4(vec4)
        return w, vec4

    def bootstrapontokens(self, tokens: List[str], epochs: int = 25, lr: float = 3e-3, maxpairs: int = 4000, progress=None) -> Dict[str, float]:
        """FIXED: Fast tensor creation using np.stack."""
        if len(tokens) < 2:
            return {"pairs": 0, "loss": 0.0}
        
        N = len(tokens)
        pairs = []
        for i in range(N - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if t1 in COGNITIVETOKENS or t2 in COGNITIVETOKENS:
                continue
            x = float(i) / max(1, N - 2)
            pairs.append((t1, t2, x))
            if len(pairs) >= maxpairs:
                break
        
        if not pairs:
            return {"pairs": 0, "loss": 0.0}
        
        # FIXED: Use np.stack instead of list of ndarrays
        y_np = np.stack([SymbolicPair.fromtokens(a, b).aestheticvector() for a, b, _ in pairs], axis=0).astype(np.float32)
        y = torch.from_numpy(y_np)
        self.updateglobalmean(y)
        
        device = self.emb.weight.device
        y = y.to(device)
        t1ids = self.tokenids([a for a, _, _ in pairs], device)
        t2ids = self.tokenids([b for _, b, _ in pairs], device)
        xpos = torch.tensor([x for _, _, x in pairs], dtype=torch.float32, device=device)
        
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        lossfn = nn.MSELoss()
        self.train()
        
        bs = 128
        losses = []
        for ep in range(int(epochs)):
            perm = torch.randperm(len(pairs), device=device)
            eploss = 0.0
            for k in range(0, len(pairs), bs):
                idx = perm[k:k + bs]
                # FIXED: Ensure xpos batch is proper 1D tensor
                xpos_batch = xpos[idx].view(-1)
                pred = self.predictvec4(t1ids[idx], t2ids[idx], xpos_batch)
                loss = lossfn(pred, y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                eploss += float(loss.item())
            
            losses.append(eploss / max(1, math.ceil(len(pairs) / bs)))
            if progress and (ep == 0 or (ep + 1) % 5 == 0):
                progress(0.10 + 0.15 * (ep + 1) / max(1, epochs), desc=f"Bootstrapping activator (loss {losses[-1]:.4f})")
        
        self.eval()
        return {"pairs": len(pairs), "loss": float(losses[-1]) if losses else 0.0}


# =============================================================================
# 2. COGNITIVE TOKENS & TEXT PROCESSING
# =============================================================================

COGNITIVETOKENS: Dict[str, float] = {
    "PROBLEM": 2.5,
    "SOLUTION": 3.0,
    "PAIR-BEGIN": 1.5,
    "PAIR-END": 1.5,
}

STOPWORDS = set("a an and are as at be by for from has have he her hers him his i in is it "
                "its me my of on or our ours she so that the their them they this to was we "
                "were what when where which who will with you your yours".split())

PROBLEMPATTERNS = [
    r"(problem|issue|challenge|difficulty|question)",
    r"(what|how|why|when|where).*(problem|issue|challenge)",
]

SOLUTIONPATTERNS = [
    r"(solution|answer|resolution|fix|remedy)",
    r"(solved|resolved|fixed|addressed)",
]


def injectcognitivetokens(text: str) -> str:
    lines = text.split("\n")
    marked = []
    for i, line in enumerate(lines):
        modified = False
        for pat in SOLUTIONPATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                line = f"SOLUTION {line}"
                modified = True
                break
        if not modified:
            for pat in PROBLEMPATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    line = f"PROBLEM {line}"
                    break
        marked.append(line)
    return "\n".join(marked)


def normalizetext(text: str) -> str:
    text = text
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def basictokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    tokens = re.findall(r"[A-Z]+-[A-Z]+|[a-zA-Z]+|[0-9]+|[\.,!\?]", text)
    out = []
    for t in tokens:
        if t in COGNITIVETOKENS:
            out.append(t)
        elif re.match(r"[A-Za-z]", t):
            out.append(t.lower())
        else:
            out.append(t)
    return out


def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in COGNITIVETOKENS:
            continue
        if t in [".", ",", "!", "?"]:
            if out:
                out[-1] += t
            else:
                out.append(t)
        else:
            if out and out[-1].endswith("-"):
                out[-1] += t
            else:
                out.append(t)
    s = " ".join(out)
    s = re.sub(r"\s+([.,!\?])", r"\1", s)
    s = re.sub(r"([.!\?])\s([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), s)
    return s


def computeproblemflowbytoken(tokens: List[str]) -> Dict[str, float]:
    """Estimate tokens' position between PROBLEM and SOLUTION (0=problem, 1=solution)."""
    acc: Dict[str, List[float]] = {}
    inseg, segstart = False, None
    for i, tok in enumerate(tokens):
        if tok == "PROBLEM":
            inseg, segstart = True, i
        elif tok == "SOLUTION" and inseg and segstart is not None:
            L = i - segstart
            if L > 1:
                for j in range(segstart + 1, i):
                    t = tokens[j]
                    if t not in COGNITIVETOKENS:
                        pos = (j - segstart) / max(1, L)
                        acc.setdefault(t, []).append(float(pos))
            inseg, segstart = False, None
    return {t: float(sum(v) / len(v)) for t, v in acc.items()}


# =============================================================================
# 3. FUZZY LOGIC & LATERAL INHIBITION
# =============================================================================

def mftri(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    x = x.clamp(min=min(a, c), max=max(a, c))
    left = (x - a) / max(1e-9, b - a)
    right = (c - x) / max(1e-9, c - b)
    return torch.clamp(torch.minimum(left, right), 0.0, 1.0)


def mftrap(x: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
    x = x.clamp(min=min(a, d), max=max(a, d))
    up = (x - a) / max(1e-9, b - a)
    down = (d - x) / max(1e-9, d - c)
    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.clamp(torch.minimum(torch.minimum(up, one), down), 0.0, 1.0)


def tnormprod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def snormmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


class FuzzyWeightController(nn.Module):
    def __init__(self):
        super().__init__()
        self.zlow, self.zmid, self.zhigh = 0.20, 0.55, 0.95

    def forward(self, entropy01, peak01, boost01, aestheticflow01, osculatorstrength=0.0):
        e = entropy01.clamp(0, 1)
        p = peak01.clamp(0, 1)
        b = boost01.clamp(0, 1)
        a = aestheticflow01.clamp(0, 1)

        elow = mftrap(e, 0.0, 0.0, 0.25, 0.45)
        emid = mftri(e, 0.25, 0.50, 0.75)
        ehigh = mftrap(e, 0.55, 0.75, 1.0, 1.0)

        plow = mftrap(p, 0.0, 0.0, 0.20, 0.40)
        pmid = mftri(p, 0.25, 0.50, 0.75)
        phigh = mftrap(p, 0.60, 0.80, 1.0, 1.0)

        blow = mftrap(b, 0.0, 0.0, 0.20, 0.45)
        bmid = mftri(b, 0.25, 0.50, 0.75)
        bhigh = mftrap(b, 0.55, 0.80, 1.0, 1.0)

        ahigh = mftrap(a, 0.5, 0.7, 1.0, 1.0)

        w1 = tnormprod(ehigh, plow)
        w2 = tnormprod(emid, bmid)
        w3 = snormmax(phigh, bhigh)
        w4 = tnormprod(elow, pmid)
        w5 = tnormprod(ahigh, emid)

        Z = torch.tensor([self.zhigh, self.zmid, self.zlow, self.zlow, self.zhigh], device=e.device, dtype=torch.float32)
        W = torch.stack([w1, w2, w3, w4, w5]).clamp(min=0.0)
        g = (W * Z).sum() / (W.sum() + 1e-12)

        s = max(0.0, min(1.0, float(osculatorstrength)))
        osc = 1.0 - ((g - 0.5) / 0.5) ** 2
        g = (1.0 - s) * g + s * osc.clamp(0.0, 1.0)
        return g.clamp(0.0, 0.5)


class LateralInhibition(nn.Module):
    def __init__(self, kernelsize=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.15, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = kernelsize // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)


# =============================================================================
# 4. QUADGRAM LM
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

    def ingest(self, tokens: List[str]):
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

    def nextdistribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        cont = []
        for (a, b, c, d) in self.quad:
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c) in self.tri:
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b) in self.bi:
                if a == w3:
                    cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]
        
        seen, cand = set(), []
        for w in cont:
            if w not in seen and w not in COGNITIVETOKENS:
                seen.add(w); cand.append(w)
        cand = cand[:500]
        
        V, addk = len(self.vocab) or 1, self.addk
        
        def getprob(w4):
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
        if probs.numel() == 0:
            cand, probs = ["the"], torch.ones(1, dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# =============================================================================
# 5. GRAPH & STATE
# =============================================================================

@dataclass
class SimpleGraph:
    nodes: List[Dict]
    edges: List[Tuple]
    cognitivemap: Dict[str, List[int]]
    pairwiseaesthetics: Dict

    def getaestheticflow(self) -> float:
        if not self.pairwiseaesthetics:
            return 0.5
        vecs = list(self.pairwiseaesthetics.values())
        if not vecs:
            return 0.5
        V = torch.stack(vecs, dim=0)
        return float(torch.mean(torch.linalg.norm(V, dim=1)).item())


@dataclass
class ModelState:
    lm: QuadgramLM
    activator: NeuronalActivator
    tokenboost: Dict[str, float]
    semanticgraph: SimpleGraph
    problemflowbytoken: Dict[str, float]


@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    state: ModelState


class RadixLRUCache:
    def __init__(self, maxitems: int = 25000):
        self.maxitems = int(max(256, maxitems))
        self.od = OrderedDict()

    def get(self, key):
        v = self.od.get(key)
        if v is None:
            return None
        self.od.move_to_end(key)
        return v

    def put(self, key, value):
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.maxitems:
            self.od.popitem(last=False)


# =============================================================================
# 6. GENERATOR WITH VECTOR SPOTTING
# =============================================================================

class NeuroSymbolicGraphGenerator:
    def __init__(self, steerstrength=1.35, focusstrength=0.5, pairwisestrength=0.4, osculatorstrength=0.1, activatorbootepochs=25):
        self.basesteer = float(steerstrength)
        self.pairwisestrength = float(pairwisestrength)
        self.osculatorstrength = float(osculatorstrength)
        self.activatorbootepochs = int(activatorbootepochs)
        self.basetemp = 0.85
        self.focuslayer = LateralInhibition(strength=float(focusstrength))
        self.fuzzyctl = FuzzyWeightController()
        self.cache = RadixLRUCache()

    def pickinitialcontext(self, lm: QuadgramLM, seedwords: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seedwords if re.match(r"[a-zA-Z0-9-]", t) and t not in COGNITIVETOKENS]
        if len(sw) >= 3:
            return sw[-3], sw[-2], sw[-1]
        if len(sw) == 2:
            return sw[-2], sw[-1], sw[-1]
        if len(sw) == 1:
            return sw[-1], sw[-1], sw[-1]
        return "the", "the", "the"

    def preparecorpus(self, rawtext: str, progress=None) -> PreparedCorpus:
        text = normalizetext(injectcognitivetokens(rawtext))
        tokens = basictokenize(text)
        
        lm = QuadgramLM()
        lm.ingest(tokens)
        
        activator = NeuronalActivator()
        if torch.cuda.is_available():
            activator = activator.cuda()
        
        if progress:
            progress(0.02, desc="Bootstrapping activator")
        activator.bootstrapontokens(tokens, epochs=self.activatorbootepochs, progress=progress)
        
        tokenboost = dict(COGNITIVETOKENS)
        flowbytoken = computeproblemflowbytoken(tokens)
        
        # Simple graph
        graph = SimpleGraph([], [], {}, {})
        state = ModelState(lm, activator, tokenboost, graph, flowbytoken)
        
        return PreparedCorpus(text, tokens, state)

    def base_lm(self, prep: PreparedCorpus, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        """Isolated LM probs."""
        cand, baseprobs = prep.state.lm.nextdistribution(w1, w2, w3)
        basep = baseprobs.to(dtype=torch.float32)
        basep = basep / (basep.sum() + 1e-12)
        basep = self.focuslayer(basep.view(1, -1)).view(-1)
        basep = basep / (basep.sum() + 1e-12)
        return cand, basep

    def apply_flow(self, prep: PreparedCorpus, cand: List[str], basep: torch.Tensor, xpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply aesthetic flow over vectorized x."""
        if xpos.dim() == 0:
            xpos = xpos.view(1)
        
        B, C = xpos.shape[0], basep.shape[0]
        device = prep.state.activator.emb.weight.device
        basep = basep.to(device).view(1, C).expand(B, C)
        
        # Entropy/peak
        val = basep[0].clamp(min=1e-12)
        H = -torch.sum(basep[0] * torch.log(val))
        V = float(C)
        entropy01 = (H / max(1e-9, math.log(max(2.0, V)))).clamp(0.0, 1.0)
        peak01 = basep[0].max().clamp(0.0, 1.0)
        
        # Boosts with activator
        boosts = torch.tensor([prep.state.tokenboost.get(w, 0.0) for w in cand], dtype=torch.float32, device=device)
        w3 = prep.tokens[-1] if prep.tokens else "the"
        
        wpair_list = []
        for b in range(B):
            wpair_b, _ = prep.state.activator.forwardweight(w3, cand, xpos=xpos[b:b+1])
            wpair_list.append(wpair_b)
        wpair = torch.stack(wpair_list, dim=0)  # [B, C]
        
        boosts_row = boosts.view(1, C).expand(B, C) + self.pairwisestrength * wpair
        
        # Fuzzy control
        aestheticflow01 = (0.5 + 0.5 * xpos).clamp(0.0, 1.0)
        boost01 = torch.tanh(boosts_row.mean(dim=-1)).clamp(0.0, 1.0)
        
        g_list = []
        for b in range(B):
            g_b = self.fuzzyctl(entropy01, peak01, boost01[b], aestheticflow01[b], self.osculatorstrength)
            g_list.append(g_b)
        g = torch.stack(g_list, dim=0).view(B, 1)
        
        effectivesteer = self.basesteer * g
        effectivetemp = self.basetemp * (1.2 - 0.7 * g)
        
        potentials = torch.log(basep.clamp(min=1e-12)) * effectivesteer + boosts_row
        potentials = potentials / torch.clamp(effectivetemp, min=1e-6)
        finalprobs = F.softmax(potentials, dim=-1)
        
        flowvec = torch.tensor([prep.state.problemflowbytoken.get(w, 0.0) for w in cand], dtype=torch.float32, device=device).view(1, C).expand(B, C)
        
        return finalprobs.detach(), flowvec.detach()

    @torch.no_grad()
    def generate_with_vector_spotting(self, prep: PreparedCorpus, prompt: str, startx: float, maxtokens: int = 220, seed: int = 42, trajectory_steps: int = 20) -> Tuple[str, Dict]:
        rng = np.random.default_rng(int(seed))
        toks = basictokenize(prompt)
        w1, w2, w3 = self.pickinitialcontext(prep.state.lm, toks)
        outtokens = []
        device = prep.state.activator.emb.weight.device
        alphacount = 0
        
        for step in range(maxtokens):
            xs = torch.linspace(startx, 1.0, trajectory_steps, device=device)
            cand, basep = self.base_lm(prep, w1, w2, w3)
            probs_traj, _ = self.apply_flow(prep, cand, basep, xs)
            
            # Simple peak selection
            progress = step / max(1, maxtokens)
            currxval = startx + (1.0 - startx) * progress
            idx_x = int(currxval * (trajectory_steps - 1))
            sample_probs = probs_traj[idx_x]
            
            p_np = sample_probs.cpu().numpy()
            p_np = p_np / (p_np.sum() + 1e-12)
            idx = rng.choice(len(cand), p=p_np)
            tok = cand[idx]
            
            outtokens.append(tok)
            w1, w2, w3 = w2, w3, tok
            
            if re.match(r"[A-Za-z]", tok):
                alphacount += 1
            if tok in [".", "!", "?"] and alphacount > 40 and step > maxtokens * 0.75:
                break
        
        return detokenize(outtokens), {"tokens": len(outtokens)}

    @torch.no_grad()
    def generate(self, prep: PreparedCorpus, prompt: str, startx: float, maxtokens: int = 220, seed: int = 42) -> str:
        """Standard generation (non-spotting)."""
        rng = np.random.default_rng(int(seed))
        toks = basictokenize(prompt)
        w1, w2, w3 = self.pickinitialcontext(prep.state.lm, toks)
        outtokens = []
        device = prep.state.activator.emb.weight.device
        alphacount = 0
        
        for i in range(maxtokens):
            progress = i / max(1, maxtokens)
            currxval = startx + (1.0 - startx) * progress
            x = torch.tensor(float(currxval), dtype=torch.float32, device=device)
            
            cand, basep = self.base_lm(prep, w1, w2, w3)
            probs, _ = self.apply_flow(prep, cand, basep, x)
            probs = probs.view(-1)
            
            p = probs.cpu().numpy()
            p = p / (p.sum() + 1e-12)
            idx = rng.choice(len(cand), p=p)
            tok = cand[idx]
            
            outtokens.append(tok)
            w1, w2, w3 = w2, w3, tok
            
            if re.match(r"[A-Za-z]", tok):
                alphacount += 1
            if tok in [".", "!", "?"] and alphacount > 40 and i > maxtokens * 0.75:
                break
        
        return detokenize(outtokens)


# =============================================================================
# 7. GRADIO APP
# =============================================================================

def loadcorpus(usehf, hfdataset, hfsplit, hfmaxrows, infile) -> str:
    if usehf:
        ds = load_dataset(hfdataset, split=hfsplit)
        rows = int(hfmaxrows) if int(hfmaxrows) > 0 else len(ds)
        rows = min(rows, len(ds))
        if "text" in ds.column_names:
            return "\n".join(str(x) for x in ds.select(range(rows))["text"])
        return "\n".join(str(ds[i]) for i in range(rows))
    else:
        if infile is None:
            raise ValueError("No input file")
        if hasattr(infile, 'name'):
            return Path(infile.name).read_text(encoding='utf-8', errors='replace')
        return str(infile)


def rungenerate(infile, usehf, hfdataset, hfsplit, hfmaxrows, prompt, seed, xstart, maxtokens, steer, focus, pairwise, oscs, bootepochs, use_spotting, progress=gr.Progress()):
    try:
        corpustext = loadcorpus(bool(usehf), str(hfdataset), str(hfsplit), int(hfmaxrows), infile)
    except Exception as e:
        return f"Corpus load error: {e}"
    
    gen = NeuroSymbolicGraphGenerator(
        steerstrength=float(steer),
        focusstrength=float(focus),
        pairwisestrength=float(pairwise),
        osculatorstrength=float(oscs),
        activatorbootepochs=int(bootepochs),
    )
    
    prep = gen.preparecorpus(corpustext, progress=progress)
    
    header = f"## STATE\n- Tokens: {len(prep.tokens)}\n" \
             f"- Activator mean: {prep.state.activator.globalmean4.detach().cpu().numpy()}\n" \
             f"----------------------------------------\n"
    
    if use_spotting:
        txt, _ = gen.generate_with_vector_spotting(prep, str(prompt), float(xstart), int(maxtokens), int(seed))
    else:
        txt = gen.generate(prep, str(prompt), float(xstart), int(maxtokens), int(seed))
    
    return header + txt


def buildapp():
    with gr.Blocks(title="NeuroSymbolic V6.2") as demo:
        gr.Markdown("# NeuroSymbolic V6.2 - Vector Spotting Edition")
        
        with gr.Row():
            with gr.Column(scale=1):
                usehf = gr.Checkbox(label="Use HF dataset", value=True)
                hfdataset = gr.Textbox(label="HF dataset", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hfsplit = gr.Textbox(label="HF split", value="train")
                hfmaxrows = gr.Slider(0, 5000, value=500, step=100, label="HF max rows")
                infile = gr.File(label="Input File (.txt/.md)", file_types=[".txt", ".md"])
                
                steer = gr.Slider(0, 5, value=1.35, step=0.05, label="Steer strength")
                focus = gr.Slider(0, 1, value=0.5, step=0.01, label="Focus strength")
                pairwise = gr.Slider(0, 2, value=0.4, step=0.05, label="Pairwise strength")
                oscs = gr.Slider(0, 1, value=0.1, step=0.05, label="Osculator strength")
                bootepochs = gr.Slider(0, 50, value=15, step=5, label="Bootstrap epochs")
            
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value="explain this?", lines=3)
                seed = gr.Number(value=42, label="Seed")
                maxtokens = gr.Slider(10, 1500, value=450, step=10, label="Max tokens")
                xstart = gr.Slider(0, 1, value=0.3, step=0.05, label="Start x")
               
                
                outtxt = gr.Textbox(label="Output", lines=20)
                btn = gr.Button("Run Generator", variant="primary")
        
        btn.click(rungenerate,
                  inputs=[infile, usehf, hfdataset, hfsplit, hfmaxrows, prompt, seed, xstart, maxtokens, steer, focus, pairwise, oscs, bootepochs],
                  outputs=outtxt)
    
    return demo


if __name__ == "__main__":
    buildapp().queue().launch()
