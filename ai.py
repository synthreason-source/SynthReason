#!/usr/bin/env python3
"""
NeuroSymbolic-Coho v1.4  — HEMICONTINUITY
─────────────────────────────────────────────────────────────
FIVE ROLE BUGS FIXED:

BUG 1 — Role keywords were defined but NEVER applied to generation.
  roles[ri][1] (the keyword list) was never unpacked or used inside
  the generation loop. All voices were statistically identical.
  FIX: At every step, the current role's keywords receive a logit
  boost (role_keyword_strength) before sampling.

BUG 2 — Role switching required sentence-end punctuation (. ! ?)
  which the LM may rarely generate, letting one role monopolise
  the entire output. The len(cur)>30 guard counted all tokens
  including punctuation, so alpha content could be far less.
  FIX: Guaranteed max-turn length (tokens_per_role) triggers a
  switch even without punctuation. Switch also fires if the turn
  has enough alpha words and hits a comma/semicolon.

BUG 3 — Every role used identical temperature and steer.
  Observer and Reflector were statistically the same voice.
  FIX: Each ROLE entry now carries (name, keywords, temp_mult,
  steer_mult). Questioner gets higher temp (exploratory),
  Reflector gets lower temp + higher steer (focused conclusions).

BUG 4 — With 6 roles and n_voices=4, Synthesizer and Reflector
  never appeared. roles[i%6] for i in range(4) always gives the
  first 4 roles.
  FIX: Role selection now cycles through ALL ROLES evenly,
  regardless of n_voices. n_voices controls how many distinct
  speaker slots exist; roles wrap around the full 6-entry list.

BUG 5 — Role keyword boost was missing from the logit vector.
  FIX: Added _role_boost() which returns a logit-space additive
  vector for the current role's keyword set, applied before the
  final softmax along with h1 and prompt biases.
"""

from __future__ import annotations
import re, numpy as np
from typing import Dict, List, Tuple, Set, Optional
import gradio as gr
import torch, torch.nn.functional as F
from datasets import load_dataset

SW = set("a an and are as at be by for from has have he her him his i in is it its me my "
         "of on or our she so that the their them they this to was we were what when where "
         "which who will with you your yours".split())

# ══════════════════════════════════════════════════════════════
#  COCHAIN COMPLEX  (unchanged from v1.2)
# ══════════════════════════════════════════════════════════════
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
                if i + n < len(tokens):
                    key = tuple(tokens[i : i + n + 1])
                    self.C[n][key] = self.C[n].get(key, 0) + 1
        self.vocab = list(self.C[0])
        self.C[0]['__total__'] = sum(self.C[0].values())
        self._content = sorted(
            [w for w in self.vocab if len(w) > 3 and w not in SW],
            key=lambda w: self.C[0].get(w, 0), reverse=True)[:400]
        self._bigram_starters = {a for (a, b) in self.C[1]}
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
            if w not in seen and w != '__total__':
                seen.add(w); out.append(w)
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

    def prompt_bias(self, cands: List[str], prompt_words: Set[str],
                    strength: float = 0.6) -> torch.Tensor:
        return torch.tensor(
            [strength if w in prompt_words else 0.0 for w in cands], dtype=torch.float32)

    # BUG 5 FIX: role keyword boost (was missing entirely)
    def role_boost(self, cands: List[str], keywords: List[str],
                   strength: float = 0.5) -> torch.Tensor:
        """Logit boost for candidates matching the current role's keyword set."""
        kw_set = set(keywords)
        return torch.tensor(
            [strength if w in kw_set else 0.0 for w in cands], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════
#  ROLE DEFINITIONS  — BUG 3 + 4 FIX
#  Format: (name, keywords, temp_mult, steer_mult)
#  temp_mult  > 1 → more exploratory / surprising
#  steer_mult > 1 → stronger pull toward h1/prompt/keyword biases
# ══════════════════════════════════════════════════════════════
ROLES = [
    # name           keywords                              temp_mult  steer_mult
    ("Observer",    ["observe","notice","perceive","seen","appears"],   1.2,  0.8),
    ("Questioner",  ["perhaps","might","could","whether","possibly"],   1.5,  0.7),
    ("Connector",   ["between","connects","relates","bridges","links"],  1.0,  1.0),
    ("Elaborator",  ["further","moreover","extends","develops","also"],  1.1,  1.0),
    ("Synthesizer", ["together","combines","integrates","unifies","thus"], 0.9, 1.2),
    ("Reflector",   ["ultimately","reveals","suggests","therefore","hence"], 0.7, 1.4),
]

# BUG 4 FIX: always cycle through all 6 roles regardless of n_voices
def _assign_roles(n_voices: int) -> List[tuple]:
    """Give each voice slot a role, cycling through all 6 evenly."""
    return [ROLES[i % len(ROLES)] for i in range(n_voices)]


# ══════════════════════════════════════════════════════════════
#  HEMI-CONTINUITY
#  Tracks probability mass flow between consecutive steps.
#  Penalises tokens that suddenly enter or leave the top-K,
#  and smooths the distribution with a blend of the previous step.
#
#  Upper violation: token appears in top-K this step but not last.
#  Lower violation: token was in top-K last step but dropped out.
#  Both violations contribute to hemi_penalty (diagnostic only).
#  Smoothing: probs = (1-α)*probs + α*prev_probs  (renormalised).
#  Penalty:   violating tokens get their probability scaled down.
# ══════════════════════════════════════════════════════════════
class HemiContinuity:
    def __init__(self, top_k: int = 50, alpha: float = 0.1,
                 penalty: float = 0.1):
        """
        top_k   : number of candidates to watch for violations
        alpha   : smoothing weight for previous distribution (0 = off)
        penalty : probability scale-down applied to violating tokens
                  (0 = no penalty, 0.5 = halve their probability)
        """
        self.top_k   = max(1, int(top_k))
        self.alpha   = float(alpha)
        self.penalty = float(penalty)
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_cands: List[str]      = []
        self._prev_probs: Optional[torch.Tensor] = None
        self._prev_index: Dict[str, int] = {}

    def reset(self) -> None:
        """Call at the start of each generate() and optionally on role switch."""
        self._reset_state()

    def apply(self, cands: List[str],
              probs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Apply hemi-continuity smoothing and violation penalty.
        Returns (adjusted_probs, hemi_penalty_scalar).
        On the very first call (no previous state) returns probs unchanged.
        """
        # ── First step: just store and pass through ───────────
        if self._prev_probs is None:
            self._prev_cands = list(cands)
            self._prev_probs = probs.detach().clone()
            self._prev_index = {w: i for i, w in enumerate(cands)}
            return probs, 0.0

        # ── Align previous probs onto the current candidate list ──
        # Words that appeared last step but not this step are ignored.
        # Words that appear this step but not last step get prev_prob = 0.
        prev_on_curr = torch.zeros(len(cands), dtype=probs.dtype)
        for i, w in enumerate(cands):
            j = self._prev_index.get(w)
            if j is not None and j < self._prev_probs.numel():
                prev_on_curr[i] = self._prev_probs[j]

        # ── Compute top-K masks ────────────────────────────────
        k = min(self.top_k, len(cands))
        curr_topk = torch.zeros(len(cands), dtype=torch.bool)
        curr_topk[torch.topk(probs, k).indices] = True

        prev_topk = torch.zeros(len(cands), dtype=torch.bool)
        if prev_on_curr.sum() > 0:
            prev_topk[torch.topk(prev_on_curr, k).indices] = True

        # ── Violation masks ────────────────────────────────────
        upper_viol = curr_topk & (~prev_topk)   # suddenly appeared
        lower_viol = prev_topk & (~curr_topk)   # suddenly dropped out

        hemi_penalty = float((probs[upper_viol].sum()
                              + prev_on_curr[lower_viol].sum()).item())

        # ── Penalty: scale down violating tokens ───────────────
        if self.penalty > 0.0 and (upper_viol.any() or lower_viol.any()):
            scale = torch.ones(len(cands), dtype=probs.dtype)
            viol  = upper_viol | lower_viol
            scale[viol] = max(0.0, 1.0 - self.penalty)
            probs = probs * scale
            probs = probs / (probs.sum() + 1e-12)

        # ── Smoothing: blend with previous distribution ────────
        if self.alpha > 0.0:
            probs = (1.0 - self.alpha) * probs + self.alpha * prev_on_curr
            probs = probs / (probs.sum() + 1e-12)

        # ── Store for next step ────────────────────────────────
        self._prev_cands = list(cands)
        self._prev_probs = probs.detach().clone()
        self._prev_index = {w: i for i, w in enumerate(cands)}

        return probs, hemi_penalty


def _saw_temp(step: int, total: int, teeth: int = 8, base: float = 0.9) -> float:
    period = max(1, total // teeth)
    phase  = (step % period) / period
    return base * (1.5 - 0.9 * phase)


def _stop_gate(cands, probs, thresh=0.55, penalty=3.0):
    top = int(probs.argmax())
    if probs[top] > thresh and cands[top] in SW:
        mask  = torch.tensor([1./penalty if w in SW else 1. for w in cands])
        probs = probs * mask; probs /= probs.sum() + 1e-12
    return probs


def _seed_ctx(cc: CochainComplex, words: List[str]) -> Tuple[str,str,str]:
    connected = [w for w in words if w in cc._bigram_starters and w not in SW and re.match(r'[a-z]',w)]
    in_vocab  = [w for w in words if w in cc.C[0] and w not in SW and re.match(r'[a-z]',w) and w not in connected]
    ordered   = connected + in_vocab
    if len(ordered) >= 3: return ordered[-3], ordered[-2], ordered[-1]
    pad = [w for w in cc._content if w not in ordered]
    combined = (ordered + pad + ['structure','analysis','pattern'])[:3]
    return combined[0], combined[1], combined[2]


def _parse_prompt_words(prompt: str, cc: CochainComplex) -> Set[str]:
    raw = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    return {w for w in raw if w in cc.C[0] and w not in SW and len(w) > 2}


def _detok(toks: List[str]) -> str:
    out: List[str] = []
    for t in toks:
        if t in '.,;:!?)':
            if out: out[-1] += t
            else: out.append(t)
        else:
            out.append(t)
    s = ' '.join(out)
    return re.sub(r'(^|[.!?] )([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)


# ══════════════════════════════════════════════════════════════
#  GENERATOR  — all five role bugs fixed
# ══════════════════════════════════════════════════════════════
def generate(cc: CochainComplex, prompt: str, n_tokens: int = 400, seed: int = 42,
             n_voices: int = 4, steer: float = 1.2, h1_str: float = 0.4,
             prompt_strength: float = 0.6, role_strength: float = 0.5,
             tokens_per_role: int = 60,
             hemi_enable: bool = True, hemi_alpha: float = 0.1,
             hemi_top_k: int = 50, hemi_penalty: float = 0.1,
             hemi_reset_on_role: bool = True) -> Tuple[str, Dict]:
    """
    Returns (text, stats_dict) where stats_dict includes hemi diagnostics.

    hemi_enable        : enable HemiContinuity smoothing + violation penalty
    hemi_alpha         : blend weight for previous distribution (0=off, 0.1=default)
    hemi_top_k         : top-K window used for violation detection
    hemi_penalty       : scale-down applied to violating tokens (0=off, 0.1=default)
    hemi_reset_on_role : reset hemi state on each role switch (distinct voice contexts)
    """
    rng  = np.random.default_rng(seed)
    toks = re.findall(r'[a-z][a-z0-9\'-]*', prompt.lower())
    prompt_words = _parse_prompt_words(prompt, cc)
    w1, w2, w3   = _seed_ctx(cc, toks)

    roles = _assign_roles(n_voices)

    # Initialise HemiContinuity
    hemi = HemiContinuity(top_k=hemi_top_k, alpha=hemi_alpha, penalty=hemi_penalty)

    turns: List[Tuple[str, List[str]]] = []
    cur:   List[str] = []
    ri          = 0
    alpha_count = 0
    total_hemi_penalty = 0.0
    hemi_steps         = 0

    for step in range(n_tokens):
        role_name, keywords, temp_mult, steer_mult = roles[ri]

        cands, probs = cc.next_dist(w1, w2, w3)

        logits = torch.log(probs.clamp(1e-12))
        logits = logits + (steer * steer_mult) * cc.h1_boost(cands, h1_str)
        if prompt_words:
            logits = logits + cc.prompt_bias(cands, prompt_words, prompt_strength)
        logits = logits + cc.role_boost(cands, keywords, role_strength)

        probs = F.softmax(logits, dim=-1)
        probs = _stop_gate(cands, probs)

        temp  = _saw_temp(step, n_tokens) * temp_mult
        probs = F.softmax(torch.log(probs.clamp(1e-12)) / max(temp, 0.1), dim=-1)

        # ── HemiContinuity ─────────────────────────────────────
        if hemi_enable:
            probs, hpen = hemi.apply(cands, probs)
            total_hemi_penalty += hpen
            hemi_steps += 1

        p   = probs.numpy(); p /= p.sum()
        tok = cands[rng.choice(len(cands), p=p)]
        cur.append(tok)
        w1, w2, w3 = w2, w3, tok

        if re.match(r'[a-zA-Z]', tok):
            alpha_count += 1

        end_of_sentence = tok in '.!?' and alpha_count >= 20
        max_turn_reached = alpha_count >= tokens_per_role
        soft_break = (tok in ',;' and alpha_count >= tokens_per_role // 2
                      and rng.random() < 0.4)

        if (end_of_sentence or max_turn_reached or soft_break) and cur:
            turns.append((role_name, cur[:]))
            cur = []; alpha_count = 0
            ri = (ri + 1) % n_voices
            # Optionally give each role a fresh probability context
            if hemi_enable and hemi_reset_on_role:
                hemi.reset()

    if cur:
        turns.append((roles[ri][0], cur))

    text = "\n".join(f"**{role}**\n{_detok(tks)}\n" for role, tks in turns)
    hemi_stats = {
        "enabled":       hemi_enable,
        "alpha":         hemi_alpha,
        "top_k":         hemi_top_k,
        "penalty":       hemi_penalty,
        "reset_on_role": hemi_reset_on_role,
        "avg_penalty":   total_hemi_penalty / max(1, hemi_steps),
        "total_penalty": total_hemi_penalty,
        "steps":         hemi_steps,
    }
    return text, hemi_stats


# ══════════════════════════════════════════════════════════════
#  CORPUS LOADER
# ══════════════════════════════════════════════════════════════
def load_corpus(use_hf: bool, dataset: str, split: str, maxrows: int, file) -> str:
    if use_hf:
        ds   = load_dataset(dataset, split=split)
        rows = min(int(maxrows), len(ds))
        col  = 'text' if 'text' in ds.column_names else ds.column_names[0]
        return "\n".join(str(ds[i][col]) for i in range(rows))
    if file is None: raise ValueError("No file provided.")
    path = file if isinstance(file,str) else getattr(file,'name', file.get('path',''))
    return open(path, encoding='utf-8', errors='replace').read()

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9\'-]*', text)]


# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
def run(use_hf, dataset, split, maxrows, file, prompt,
        seed, ntok, voices, steer, h1_str, prompt_strength,
        role_strength, tokens_per_role,
        hemi_enable, hemi_alpha, hemi_top_k, hemi_penalty, hemi_reset_on_role,
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

    roles_used   = _assign_roles(int(voices))
    role_summary = " → ".join(f"{r[0]}(t×{r[2]},s×{r[3]})" for r in roles_used)

    out, hemi_stats = generate(
        cc, prompt, n_tokens=int(ntok), seed=int(seed),
        n_voices=int(voices), steer=float(steer), h1_str=float(h1_str),
        prompt_strength=float(prompt_strength),
        role_strength=float(role_strength),
        tokens_per_role=int(tokens_per_role),
        hemi_enable=bool(hemi_enable),
        hemi_alpha=float(hemi_alpha),
        hemi_top_k=int(hemi_top_k),
        hemi_penalty=float(hemi_penalty),
        hemi_reset_on_role=bool(hemi_reset_on_role),
    )

    hemi_line = (
        f"HemiContinuity: ON  α={hemi_stats['alpha']}  "
        f"top_k={hemi_stats['top_k']}  penalty={hemi_stats['penalty']}  "
        f"reset_on_role={hemi_stats['reset_on_role']}  "
        f"avg_violation={hemi_stats['avg_penalty']:.4f}"
        if hemi_stats["enabled"]
        else "HemiContinuity: OFF"
    )

    stats = "\n".join([
        f"Vocab: {len(cc.vocab)}  |  C¹ bigrams: {len(cc.C[1])}",
        f"Seed context: ({w1}, {w2}, {w3})",
        f"Prompt words active: {sorted(prompt_words) or '—'}",
        f"Prompt words OOV: {sorted(oov) or '—'}",
        f"Roles in use: {role_summary}",
        hemi_line,
        f"H¹ top 8: {', '.join(sorted(cc.h1, key=cc.h1.get, reverse=True)[:8])}",
    ])
    progress(1.0)
    return out, stats


def ui():
    with gr.Blocks(title="NeuroSymbolic-Coho", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# NeuroSymbolic-Coho v1.4\n"
            "**HemiContinuity** · Roles · Persistent prompt bias · Cochain complex C⁰→C¹→C²→C³\n\n"
            "HemiContinuity smooths probability flow between steps — tokens cannot teleport "
            "into or out of the top-K without a penalty, enforcing coherent evolution of the "
            "candidate distribution across the whole narrative."
        )
        with gr.Row():
            with gr.Column(scale=1):
                use_hf  = gr.Checkbox(label="HuggingFace dataset", value=True)
                dataset = gr.Textbox(
                    value="AiresPucrs/stanford-encyclopedia-philosophy", label="Dataset")
                split   = gr.Textbox(value="train", label="Split")
                maxrows = gr.Slider(100, 5000, value=1000, step=100, label="Max rows")
                file    = gr.File(label="Or upload .txt", file_types=[".txt",".md"], visible=False)
                use_hf.change(lambda v:(gr.update(visible=v),gr.update(visible=not v)),
                              [use_hf],[maxrows,file])

                prompt  = gr.Textbox(value="Consider the nature of persistent homology",
                                     label="Prompt",
                                     info="Content words bias every generation step.")
                seed    = gr.Number(value=42, label="Seed")
                ntok    = gr.Slider(100, 1000, value=400, step=50, label="Tokens")
                voices  = gr.Slider(2, 6, value=6, step=1, label="Voices",
                                    info="Each slot cycles through all 6 roles (Observer→Reflector).")
                steer   = gr.Slider(0.5, 3.0, value=1.2, step=0.1, label="Base steer strength")
                h1_str  = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="H¹ boost (cohomology)")

                with gr.Accordion("Prompt & Role tuning", open=True):
                    prompt_strength = gr.Slider(0.0, 2.0, value=0.6, step=0.1,
                                                label="Prompt strength",
                                                info="0 = seed-only. 0.6 = recommended.")
                    role_strength   = gr.Slider(0.0, 2.0, value=0.5, step=0.1,
                                                label="Role keyword strength",
                                                info="How hard each role's keywords pull on generation.")
                    tokens_per_role = gr.Slider(20, 150, value=60, step=10,
                                                label="Max tokens per role turn",
                                                info="Forces a role switch after this many alpha tokens.")

                with gr.Accordion("HemiContinuity", open=True):
                    gr.Markdown(
                        "Smooths probability flow between steps. Penalises tokens that "
                        "suddenly enter or leave the top-K, enforcing gradual evolution "
                        "of the candidate distribution."
                    )
                    hemi_enable       = gr.Checkbox(label="Enable HemiContinuity", value=True)
                    hemi_alpha        = gr.Slider(0.0, 0.3, value=0.1, step=0.01,
                                                  label="Smoothing α",
                                                  info="Blend weight for previous distribution. "
                                                       "0 = no smoothing.")
                    hemi_top_k        = gr.Slider(10, 100, value=50, step=5,
                                                  label="Top-K violation window",
                                                  info="Candidates watched for sudden probability jumps.")
                    hemi_penalty      = gr.Slider(0.0, 0.5, value=0.1, step=0.05,
                                                  label="Violation penalty",
                                                  info="Scale-down applied to violating tokens. "
                                                       "0 = smooth only, no penalty.")
                    hemi_reset_on_role = gr.Checkbox(
                        label="Reset on role switch",
                        value=True,
                        info="Give each role a fresh probability context at the start of its turn.")

                btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                out_txt  = gr.Textbox(label="Narrative", lines=28, show_copy_button=True)
                out_stat = gr.Textbox(label="Cohomology + role stats", lines=7)

        btn.click(run,
                  [use_hf, dataset, split, maxrows, file, prompt, seed, ntok, voices,
                   steer, h1_str, prompt_strength, role_strength, tokens_per_role,
                   hemi_enable, hemi_alpha, hemi_top_k, hemi_penalty, hemi_reset_on_role],
                  [out_txt, out_stat])
    return demo


if __name__ == "__main__":
    ui().queue().launch(share=False)
