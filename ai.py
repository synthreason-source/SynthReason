#!/usr/bin/env python3
"""
NeuroSymbolic-Coho v1.1  — BUG-FIXED
─────────────────────────────────────────────────────────────
FIX: fit() stored short (incomplete) n-grams in C[n] when the
token window ran past the end of the list.  For example, at the
last token i=N-1 with n=1 the slice tokens[N-1 : N+1] has only
ONE element, producing a 1-tuple key ('word',) inside C[1].
_coboundary1 then tried to unpack every C[1] key as (a, b) and
crashed with "not enough values to unpack (expected 2, got 1)".

The one-line fix is a length guard in fit():
    if len(key) == n + 1:        # only store complete n-grams
This also silently fixes C[2] and C[3] boundary conditions.

Everything else is identical to v1.0.
─────────────────────────────────────────────────────────────
COHOMOLOGICAL REDUCTION PRINCIPLE
  Treat the corpus as a cochain complex  C⁰ → C¹ → C² → C³
    C⁰ = unigrams,  C¹ = bigrams,  C² = trigrams,  C³ = quadgrams
  Coboundary δⁿ: Cⁿ → Cⁿ⁺¹ lifts probability mass upward.
  H¹ = ker δ¹ / im δ⁰  identifies tokens that are "cocycles" —
    they recur across contexts without being mere boundaries of
    higher-order patterns.  These are boosted at generation time.
  Sawtooth filtration modulates temperature (not logits) so the
  wave creates exploration/exploitation cycles without collapse.
  Stop-word gate kills runaway high-frequency tokens.
"""

from __future__ import annotations
import re, numpy as np
from typing import Dict, List, Tuple
import gradio as gr
import torch, torch.nn.functional as F
from datasets import load_dataset

# ── stop words ────────────────────────────────────────────────
SW = set("a an and are as at be by for from has have he her him his i in is it its me my "
         "of on or our she so that the their them they this to was we were what when where "
         "which who will with you your yours".split())

# ══════════════════════════════════════════════════════════════
#  COCHAIN COMPLEX
# ══════════════════════════════════════════════════════════════
class CochainComplex:
    """C⁰→C¹→C²→C³ with coboundary-lifted boosts and H¹ persistence."""

    def __init__(self, add_k: float = 0.3):
        self.k = add_k
        self.C: List[Dict] = [{}, {}, {}, {}]   # grade-n n-gram counts
        self.vocab: List[str] = []
        self._content: List[str] = []            # non-stop content words
        self.h1: Dict[str, float] = {}           # H¹ cocycle scores

    # ── ingest ────────────────────────────────────────────────
    def fit(self, tokens: List[str]) -> None:
        for d in self.C:
            d.clear()

        for i, t in enumerate(tokens):
            self.C[0][t] = self.C[0].get(t, 0) + 1
            for n in range(1, 4):
                # FIX: only store a key when the slice is exactly (n+1) tokens long.
                # Without this guard, the last few positions produce short tuples
                # (e.g. a 1-tuple in C[1]) that crash the (a, b) unpack in _coboundary1.
                if i + n < len(tokens):          # ← was:  i + n <= len(tokens)
                    key = tuple(tokens[i : i + n + 1])   # always exactly (n+1) elements
                    self.C[n][key] = self.C[n].get(key, 0) + 1

        self.vocab = list(self.C[0])
        self.C[0]['__total__'] = sum(self.C[0].values())

        # content vocab for anti-lock fallback
        self._content = sorted(
            [w for w in self.vocab if len(w) > 3 and w not in SW],
            key=lambda w: self.C[0].get(w, 0),
            reverse=True,
        )[:400]

        self._compute_h1()

    # ── δ¹: lift unigram mass to bigram boundary ──────────────
    def _coboundary1(self, w: str) -> float:
        """Fraction of w's unigram mass that is a bigram boundary."""
        uni = self.C[0].get(w, 0)
        if uni == 0:
            return 0.0
        # Safe: fit() guarantees every key in C[1] is a 2-tuple.
        bi_as_boundary = sum(v for (a, b), v in self.C[1].items() if b == w)
        return bi_as_boundary / (uni + 1e-9)

    def _compute_h1(self) -> None:
        """H¹ score = high frequency AND low coboundary ratio → true cocycle."""
        total = self.C[0].get('__total__', 1)
        for w in self.vocab:
            freq           = self.C[0].get(w, 0) / total
            boundary_ratio = self._coboundary1(w)
            topo = any(k in w for k in (
                "homolog", "cohomolog", "betti", "filtrat",
                "simplic", "persist", "homotop", "sheaf", "manifold",
            ))
            h1 = freq * (1.0 - 0.7 * boundary_ratio) * (3.0 if topo else 1.0)
            self.h1[w] = float(h1)

    # ── next-token distribution ───────────────────────────────
    def next_dist(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        k = self.k
        V = len(self.vocab) + 1

        # cascade: quad → tri → bi → content fallback
        cands: List[str] = []
        for (a, b, c, d), _ in self.C[3].items():
            if (a, b, c) == (w1, w2, w3):
                cands.append(d)
        if len(set(cands)) < 8:
            for (a, b, c), _ in self.C[2].items():
                if (a, b) == (w2, w3):
                    cands.append(c)
        if len(set(cands)) < 8:
            for (a, b), _ in self.C[1].items():
                if a == w3:
                    cands.append(b)
        if len(set(cands)) < 8:          # anti-lock: content words only
            cands += self._content[:200]

        seen: set = set()
        out: List[str] = []
        for w in cands:
            if w not in seen and w != '__total__':
                seen.add(w)
                out.append(w)
        out = out[:500] or self._content[:200] or ['structure']

        def p(w4: str) -> float:
            tri = self.C[2].get((w1, w2, w3), 0)
            if tri:
                return (self.C[3].get((w1, w2, w3, w4), 0) + k) / (tri + k * V)
            bi = self.C[1].get((w2, w3), 0)
            if bi:
                return (self.C[2].get((w2, w3, w4), 0) + k) / (bi + k * V)
            u = self.C[0].get(w3, 0)
            if u:
                return (self.C[1].get((w3, w4), 0) + k) / (u + k * V)
            return (self.C[0].get(w4, 0) + k) / (self.C[0].get('__total__', 1) + k * V)

        probs = torch.tensor([p(w) for w in out], dtype=torch.float32)
        return out, probs / (probs.sum() + 1e-12)

    # ── H¹ boost vector ───────────────────────────────────────
    def h1_boost(self, cands: List[str], strength: float = 0.4) -> torch.Tensor:
        return torch.tensor(
            [self.h1.get(w, 0.0) * strength for w in cands],
            dtype=torch.float32,
        )


# ══════════════════════════════════════════════════════════════
#  GENERATOR
# ══════════════════════════════════════════════════════════════
ROLES = [
    ("Observer",    ["observe", "notice", "perceive"]),
    ("Questioner",  ["perhaps", "might", "could", "whether"]),
    ("Connector",   ["between", "connects", "relates", "bridges"]),
    ("Elaborator",  ["further", "moreover", "extends", "develops"]),
    ("Synthesizer", ["together", "combines", "integrates", "unifies"]),
    ("Reflector",   ["ultimately", "thus", "suggests", "reveals"]),
]


def _saw_temp(step: int, total: int, teeth: int = 8, base: float = 0.9) -> float:
    """Sawtooth on TEMPERATURE: hot→cool per tooth, then reset.  Prevents collapse."""
    period = max(1, total // teeth)
    phase  = (step % period) / period        # 0 → 1
    return base * (1.5 - 0.9 * phase)        # 1.35 → 0.45


def _stop_gate(cands: List[str], probs: torch.Tensor,
               thresh: float = 0.55, penalty: float = 3.0) -> torch.Tensor:
    """Suppress runaway stop-word dominance."""
    top = int(probs.argmax())
    if probs[top] > thresh and cands[top] in SW:
        mask  = torch.tensor([1.0 / penalty if w in SW else 1.0 for w in cands])
        probs = probs * mask
        probs = probs / (probs.sum() + 1e-12)
    return probs


def _seed_ctx(cc: CochainComplex, words: List[str]) -> Tuple[str, str, str]:
    sw = [w for w in words if w in cc.C[0] and w not in SW and re.match(r'[a-z]', w)]
    if len(sw) >= 3:
        return sw[-3], sw[-2], sw[-1]
    if len(sw) >= 1:
        return sw[0], sw[0], sw[0]
    fb = (cc._content + ['structure', 'analysis', 'pattern'])[:3]
    return fb[0], fb[1], fb[2]


def _detok(toks: List[str]) -> str:
    out: List[str] = []
    for t in toks:
        if t in '.,;:!?)':
            if out:
                out[-1] += t
            else:
                out.append(t)
        else:
            out.append(t)
    s = ' '.join(out)
    return re.sub(r'(^|[?] )([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)


def generate(cc: CochainComplex, prompt: str, n_tokens: int = 400, seed: int = 42,
             n_voices: int = 4, steer: float = 1.2, h1_str: float = 0.4) -> str:
    rng  = np.random.default_rng(seed)
    toks = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    w1, w2, w3 = _seed_ctx(cc, toks)
    roles       = [ROLES[i % len(ROLES)] for i in range(n_voices)]
    turns: List[Tuple[str, List[str]]] = []
    cur:  List[str] = []
    ri = 0

    for step in range(n_tokens):
        cands, probs = cc.next_dist(w1, w2, w3)

        # H¹ cohomology boost
        probs = F.softmax(
            torch.log(probs.clamp(1e-12)) + steer * cc.h1_boost(cands, h1_str),
            dim=-1,
        )
        probs = _stop_gate(cands, probs)

        # Sawtooth temperature modulation
        temp  = _saw_temp(step, n_tokens)
        probs = F.softmax(torch.log(probs.clamp(1e-12)) / temp, dim=-1)

        p   = probs.numpy(); p /= p.sum()
        tok = cands[rng.choice(len(cands), p=p)]
        cur.append(tok)
        w1, w2, w3 = w2, w3, tok

        # Role switch on sentence boundary
        if tok in '.!?' and len(cur) > 30:
            turns.append((roles[ri][0], cur[:]))
            cur = []
            ri  = (ri + 1) % n_voices

    if cur:
        turns.append((roles[ri][0], cur))

    lines = [f"**{role}**\n{_detok(tks)}\n" for role, tks in turns]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  CORPUS LOADER
# ══════════════════════════════════════════════════════════════
def load_corpus(use_hf: bool, dataset: str, split: str, maxrows: int, file) -> str:
    if use_hf:
        ds   = load_dataset(dataset, split=split)
        rows = min(int(maxrows), len(ds))
        col  = 'text' if 'text' in ds.column_names else ds.column_names[0]
        return "\n".join(str(ds[i][col]) for i in range(rows))
    if file is None:
        raise ValueError("No file provided.")
    path = file if isinstance(file, str) else getattr(file, 'name', file.get('path', ''))
    return open(path, encoding='utf-8', errors='replace').read()


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split()]


# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
def run(use_hf, dataset, split, maxrows, file, prompt,
        seed, ntok, voices, steer, h1_str,
        progress=gr.Progress()):
    progress(0.1, desc="Loading corpus…")
    text = load_corpus(use_hf, dataset, split, maxrows, file)
    progress(0.3, desc="Building cochain complex…")
    cc = CochainComplex()
    cc.fit(tokenize(text))
    progress(0.5, desc="Generating…")
    out = generate(cc, prompt, n_tokens=int(ntok), seed=int(seed),
                   n_voices=int(voices), steer=float(steer), h1_str=float(h1_str))
    stats = (
        f"Vocab: {len(cc.vocab)}  |  "
        f"C¹ bigrams: {len(cc.C[1])}  |  "
        f"H¹ cocycles (top 8): "
        + ', '.join(sorted(cc.h1, key=cc.h1.get, reverse=True)[:8])
    )
    progress(1.0)
    return out, stats


def ui():
    with gr.Blocks(title="NeuroSymbolic-Coho", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# NeuroSymbolic-Coho v1.1\n"
            "Cohomological narrative generation — **10 % of V7.2**\n\n"
            "Uses the cochain complex **C⁰→C¹→C²→C³** (uni/bi/tri/quad-grams). "
            "H¹ cocycle scores boost tokens that persist across filtration levels. "
            "Sawtooth wave modulates temperature, not logits, preventing collapse."
        )
        with gr.Row():
            with gr.Column(scale=1):
                use_hf  = gr.Checkbox(label="HuggingFace dataset", value=True)
                dataset = gr.Textbox(value="AiresPucrs/stanford-encyclopedia-philosophy",
                                     label="Dataset")
                split   = gr.Textbox(value="train", label="Split")
                maxrows = gr.Slider(100, 5000, value=1000, step=100, label="Max rows")
                file    = gr.File(label="Or upload .txt", file_types=[".txt", ".md"],
                                  visible=False)
                use_hf.change(
                    lambda v: (gr.update(visible=v), gr.update(visible=not v)),
                    [use_hf], [maxrows, file],
                )
                prompt  = gr.Textbox(value="what is the meaning of life?",
                                     label="Prompt")
                seed    = gr.Number(value=42, label="Seed")
                ntok    = gr.Slider(100, 1000, value=400, step=50, label="Tokens")
                voices  = gr.Slider(2, 6, value=4, step=1, label="Voices")
                steer   = gr.Slider(0.5, 3.0, value=1.2, step=0.1, label="Steer strength")
                h1_str  = gr.Slider(0.0, 1.0, value=0.4, step=0.05,
                                    label="H¹ boost (cohomology)")
                btn     = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                out_txt  = gr.Textbox(label="Narrative", lines=28, show_copy_button=True)
                out_stat = gr.Textbox(label="Cohomology stats", lines=3)

        btn.click(
            run,
            [use_hf, dataset, split, maxrows, file, prompt,
             seed, ntok, voices, steer, h1_str],
            [out_txt, out_stat],
        )
    return demo


if __name__ == "__main__":
    ui().queue().launch(share=False)
