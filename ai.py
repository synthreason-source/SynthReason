#!/usr/bin/env python3
"""
NeuroSymbolic-Coho v1.2  — PROMPT CONTEXT FIX
See inline comments for the three bugs fixed.
"""
from __future__ import annotations
import re, numpy as np
from typing import Dict, List, Tuple, Set
import gradio as gr
import torch, torch.nn.functional as F
from datasets import load_dataset

SW = set("a an and are as at be by for from has have he her him his i in is it its me my "
         "of on or our she so that the their them they this to was we were what when where "
         "which who will with you your yours".split())

class CochainComplex:
    def __init__(self, add_k: float = 0.3):
        self.k = add_k
        self.C: List[Dict] = [{}, {}, {}, {}]
        self.vocab: List[str] = []
        self._content: List[str] = []
        self.h1: Dict[str, float] = {}
        self._bigram_starters: Set[str] = set()

    def fit(self, tokens: List[str]) -> None:
        for d in self.C: d.clear()
        for i, t in enumerate(tokens):
            self.C[0][t] = self.C[0].get(t, 0) + 1
            for n in range(1, 4):
                if i + n < len(tokens):          # strict < : always full (n+1)-tuples
                    key = tuple(tokens[i : i + n + 1])
                    self.C[n][key] = self.C[n].get(key, 0) + 1
        self.vocab = list(self.C[0])
        self.C[0]['__total__'] = sum(self.C[0].values())
        self._content = sorted(
            [w for w in self.vocab if len(w) > 3 and w not in SW],
            key=lambda w: self.C[0].get(w, 0), reverse=True)[:400]
        self._bigram_starters = {a for (a, b) in self.C[1]}  # BUG 3 fix
        self._compute_h1()

    def _coboundary1(self, w: str) -> float:
        uni = self.C[0].get(w, 0)
        if uni == 0: return 0.0
        bi_as_boundary = sum(v for (a, b), v in self.C[1].items() if b == w)
        return bi_as_boundary / (uni + 1e-9)

    def _compute_h1(self) -> None:
        total = self.C[0].get('__total__', 1)
        for w in self.vocab:
            freq = self.C[0].get(w, 0) / total
            br   = self._coboundary1(w)
            topo = any(k in w for k in ("homolog","cohomolog","betti","filtrat",
                                         "simplic","persist","homotop","sheaf","manifold"))
            self.h1[w] = float(freq * (1.0 - 0.7 * br) * (3.0 if topo else 1.0))

    def next_dist(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        k = self.k; V = len(self.vocab) + 1
        cands: List[str] = []
        for (a,b,c,d),_ in self.C[3].items():
            if (a,b,c)==(w1,w2,w3): cands.append(d)
        if len(set(cands))<8:
            for (a,b,c),_ in self.C[2].items():
                if (a,b)==(w2,w3): cands.append(c)
        if len(set(cands))<8:
            for (a,b),_ in self.C[1].items():
                if a==w3: cands.append(b)
        if len(set(cands))<8:
            cands += self._content[:200]
        seen: set = set(); out: List[str] = []
        for w in cands:
            if w not in seen and w!='__total__': seen.add(w); out.append(w)
        out = out[:500] or self._content[:200] or ['structure']

        def p(w4):
            tri = self.C[2].get((w1,w2,w3),0)
            if tri: return (self.C[3].get((w1,w2,w3,w4),0)+k)/(tri+k*V)
            bi = self.C[1].get((w2,w3),0)
            if bi: return (self.C[2].get((w2,w3,w4),0)+k)/(bi+k*V)
            u = self.C[0].get(w3,0)
            if u: return (self.C[1].get((w3,w4),0)+k)/(u+k*V)
            return (self.C[0].get(w4,0)+k)/(self.C[0].get('__total__',1)+k*V)

        probs = torch.tensor([p(w) for w in out], dtype=torch.float32)
        return out, probs/(probs.sum()+1e-12)

    def h1_boost(self, cands: List[str], strength: float = 0.4) -> torch.Tensor:
        return torch.tensor([self.h1.get(w,0.)*strength for w in cands], dtype=torch.float32)

    # BUG 1 FIX: persistent prompt bias applied every step
    def prompt_bias(self, cands: List[str], prompt_words: Set[str],
                    strength: float = 0.6) -> torch.Tensor:
        return torch.tensor(
            [strength if w in prompt_words else 0.0 for w in cands], dtype=torch.float32)


ROLES = [
    ("Observer",    ["observe","notice","perceive"]),
    ("Questioner",  ["perhaps","might","could","whether"]),
    ("Connector",   ["between","connects","relates","bridges"]),
    ("Elaborator",  ["further","moreover","extends","develops"]),
    ("Synthesizer", ["together","combines","integrates","unifies"]),
    ("Reflector",   ["ultimately","thus","suggests","reveals"]),
]

def _saw_temp(step, total, teeth=8, base=0.9):
    period = max(1, total//teeth); phase = (step%period)/period
    return base*(1.5-0.9*phase)

def _stop_gate(cands, probs, thresh=0.55, penalty=3.0):
    top = int(probs.argmax())
    if probs[top]>thresh and cands[top] in SW:
        mask = torch.tensor([1./penalty if w in SW else 1. for w in cands])
        probs = probs*mask; probs /= probs.sum()+1e-12
    return probs

def _seed_ctx(cc: CochainComplex, words: List[str]) -> Tuple[str,str,str]:
    # BUG 2+3 FIX: prefer words with bigram continuations; pad with distinct content words
    connected = [w for w in words if w in cc._bigram_starters and w not in SW and re.match(r'[a-z]',w)]
    in_vocab  = [w for w in words if w in cc.C[0] and w not in SW and re.match(r'[a-z]',w) and w not in connected]
    ordered   = connected + in_vocab
    if len(ordered)>=3: return ordered[-3], ordered[-2], ordered[-1]
    pad = [w for w in cc._content if w not in ordered]
    combined = (ordered + pad + ['structure','analysis','pattern'])[:3]
    return combined[0], combined[1], combined[2]

def _parse_prompt_words(prompt: str, cc: CochainComplex) -> Set[str]:
    # BUG 1 FIX: full set of in-corpus prompt words for persistent bias
    raw = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    return {w for w in raw if w in cc.C[0] and w not in SW and len(w)>2}

def _detok(toks):
    out=[]
    for t in toks:
        if t in '.,;:!?)': out[-1]+=t if out else out.append(t)
        else: out.append(t)
    s=' '.join(out)
    return re.sub(r'(^|[.!?] )([a-z])', lambda m:m.group(1)+m.group(2).upper(), s)

def generate(cc: CochainComplex, prompt: str, n_tokens=400, seed=42,
             n_voices=4, steer=1.2, h1_str=0.4, prompt_strength=0.6) -> str:
    rng  = np.random.default_rng(seed)
    toks = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    prompt_words = _parse_prompt_words(prompt, cc)   # BUG 1 FIX
    w1,w2,w3 = _seed_ctx(cc, toks)                   # BUG 2+3 FIX
    roles = [ROLES[i%len(ROLES)] for i in range(n_voices)]
    turns: List[Tuple[str,List[str]]] = []
    cur: List[str] = []; ri = 0

    for step in range(n_tokens):
        cands, probs = cc.next_dist(w1,w2,w3)
        logits = torch.log(probs.clamp(1e-12)) + steer*cc.h1_boost(cands,h1_str)
        # BUG 1 FIX: add persistent prompt bias EVERY step
        if prompt_words:
            logits = logits + cc.prompt_bias(cands, prompt_words, prompt_strength)
        probs = F.softmax(logits, dim=-1)
        probs = _stop_gate(cands, probs)
        temp  = _saw_temp(step, n_tokens)
        probs = F.softmax(torch.log(probs.clamp(1e-12))/temp, dim=-1)
        p = probs.numpy(); p /= p.sum()
        tok = cands[rng.choice(len(cands), p=p)]
        cur.append(tok); w1,w2,w3 = w2,w3,tok
        if tok in '.!?' and len(cur)>30:
            turns.append((roles[ri][0], cur[:])); cur=[]; ri=(ri+1)%n_voices
    if cur: turns.append((roles[ri][0], cur))
    return "\n".join(f"**{role}**\n{_detok(tks)}\n" for role,tks in turns)


def load_corpus(use_hf, dataset, split, maxrows, file) -> str:
    if use_hf:
        ds = load_dataset(dataset, split=split)
        rows = min(int(maxrows), len(ds))
        col = 'text' if 'text' in ds.column_names else ds.column_names[0]
        return "\n".join(str(ds[i][col]) for i in range(rows))
    if file is None: raise ValueError("No file provided.")
    path = file if isinstance(file,str) else getattr(file,'name',file.get('path',''))
    return open(path, encoding='utf-8', errors='replace').read()

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9\'-]*', text)]

def run(use_hf, dataset, split, maxrows, file, prompt,
        seed, ntok, voices, steer, h1_str, prompt_strength,
        progress=gr.Progress()):
    progress(0.1, desc="Loading corpus…")
    text = load_corpus(use_hf, dataset, split, maxrows, file)
    progress(0.3, desc="Building cochain complex…")
    cc = CochainComplex(); cc.fit(tokenize(text))
    progress(0.5, desc="Generating…")
    prompt_words = _parse_prompt_words(prompt, cc)
    oov = [w for w in re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
           if w not in SW and len(w)>2 and w not in cc.C[0]]
    toks = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    w1,w2,w3 = _seed_ctx(cc, toks)
    out = generate(cc, prompt, n_tokens=int(ntok), seed=int(seed),
                   n_voices=int(voices), steer=float(steer), h1_str=float(h1_str),
                   prompt_strength=float(prompt_strength))
    stats = "\n".join([
        f"Vocab: {len(cc.vocab)}  |  C¹ bigrams: {len(cc.C[1])}",
        f"Seed context: ({w1}, {w2}, {w3})",
        f"Prompt words active (bias every step): {sorted(prompt_words) or '—'}",
        f"Prompt words OOV (not in corpus): {sorted(oov) or '—'}",
        f"H¹ top 8: {', '.join(sorted(cc.h1, key=cc.h1.get, reverse=True)[:8])}",
    ])
    progress(1.0)
    return out, stats

def ui():
    with gr.Blocks(title="NeuroSymbolic-Coho", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NeuroSymbolic-Coho v1.2\n"
                    "**Prompt persists throughout generation** — not just the first 3 tokens.\n\n"
                    "H¹ cocycle scores + continuous prompt bias. Sawtooth temperature, no logit collapse.")
        with gr.Row():
            with gr.Column(scale=1):
                use_hf  = gr.Checkbox(label="HuggingFace dataset", value=True)
                dataset = gr.Textbox(value="AiresPucrs/stanford-encyclopedia-philosophy", label="Dataset")
                split   = gr.Textbox(value="train", label="Split")
                maxrows = gr.Slider(100, 5000, value=1000, step=100, label="Max rows")
                file    = gr.File(label="Or upload .txt", file_types=[".txt",".md"], visible=False)
                use_hf.change(lambda v:(gr.update(visible=v),gr.update(visible=not v)),
                              [use_hf],[maxrows,file])
                prompt  = gr.Textbox(value="Consider the nature of persistent homology",
                                     label="Prompt",
                                     info="Content words exert a bias on every generation step.")
                seed    = gr.Number(value=42, label="Seed")
                ntok    = gr.Slider(100,1000,value=400,step=50,label="Tokens")
                voices  = gr.Slider(2,6,value=4,step=1,label="Voices")
                steer   = gr.Slider(0.5,3.0,value=1.2,step=0.1,label="Steer strength")
                h1_str  = gr.Slider(0.0,1.0,value=0.4,step=0.05,label="H¹ boost (cohomology)")
                prompt_strength = gr.Slider(0.0,2.0,value=0.6,step=0.1,
                                            label="Prompt strength",
                                            info="0 = seed only (old). 0.6 = recommended. 1.5+ = strong pull.")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                out_txt  = gr.Textbox(label="Narrative", lines=28, show_copy_button=True)
                out_stat = gr.Textbox(label="Cohomology + prompt stats", lines=6)
        btn.click(run,
                  [use_hf,dataset,split,maxrows,file,prompt,seed,ntok,voices,steer,h1_str,prompt_strength],
                  [out_txt,out_stat])
    return demo

if __name__ == "__main__":
    ui().queue().launch(share=False)
