#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Callable, Optional
import gradio as gr
import numpy as np

# ================================================================
# SEED CONTROL
# ================================================================
def seed_everything(seed: int = 42):
    """Unified seeding for all RNGs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.numel() == 0:
        return x
    z = x - x.max()
    ez = torch.exp(z)
    s = ez.sum()
    return ez / (s if s > 0 else 1.0)

# ================================================================
# HOMOMORPHIC KERNELS (Seed-aware)
# ================================================================
class SurjectionKernels:
    def __init__(self, tokens: List[str], ngram_model: Dict, global_seed: int = 42):
        seed_everything(global_seed)
        self.tokens = tokens
        self.model = ngram_model
        self.keys = list(ngram_model.keys())
        
        self.feat_map = self._build_feature_homomorphism()
        self.anchors = self._build_anchor_homomorphism()
        self.anchor_hits = torch.zeros(len(self.anchors), dtype=torch.int32)
        self.generation_state = []
        
        self.K_SIM = 0.35
        self.K_LIN = 0.45  
        self.K_ALT = 14
        self.global_seed = global_seed
    
    def _build_feature_homomorphism(self) -> Dict[str, torch.Tensor]:
        freq = Counter(self.tokens)
        total = max(1, len(self.tokens))
        
        def φ(w: str) -> torch.Tensor:
            L, f = len(w), freq.get(w, 1)
            feats = torch.tensor([
                L / 10.0,                           
                sum(1 for c in w if c.isalpha()) / (L+1),  
                sum(1 for c in w if c in "aeiou") / (L+1), 
                torch.log(torch.tensor(f+1.0)) / torch.log(torch.tensor(total+1.0)), 
                1.0 / (f+1.0)                         
            ], dtype=torch.float32)
            norm = torch.linalg.norm(feats)
            if norm > 1e-9:
                return 2.0 * (feats / norm) - feats
            return feats
        
        unique_tokens = list(set(self.tokens))[:5000]
        return {w: φ(w) for w in unique_tokens}
    
    def _build_anchor_homomorphism(self) -> torch.Tensor:
        counts = Counter(self.tokens)
        top_words = [w for w, _ in counts.most_common(64)]
        anchors = []
        
        for w in top_words:
            if w in self.feat_map:
                v = self.feat_map[w] / (torch.linalg.norm(self.feat_map[w]) + 1e-9)
                if not anchors or min(torch.linalg.norm(v - a).item() for a in anchors) > 0.35:
                    anchors.append(v)
                if len(anchors) >= 18:
                    break
        
        if anchors:
            return torch.stack(anchors)
        return torch.eye(5, dtype=torch.float32)
    
    def _nearest_anchor_idx(self, vec: torch.Tensor) -> int:
        v = vec / (torch.linalg.norm(vec) + 1e-9)
        sims = self.anchors @ v
        return int(torch.argmax(sims).item())
    
    def K_similarity(self, context: Tuple[str, str], candidates: List[str]) -> torch.Tensor:
        if not candidates:
            return torch.tensor([1.0])
        
        def surject(u: torch.Tensor, v: torch.Tensor) -> float:
            dot = float(torch.dot(u, v))
            nv2 = float(torch.dot(v, v) + 1e-9)
            return float(torch.clamp(0.5 * (torch.tanh(torch.tensor(dot / nv2)) + 1), 0, 1))
        
        scores = []
        prev_word = context[0] if len(context) > 0 else ""
        for c in candidates:
            if prev_word in self.feat_map and c in self.feat_map:
                score = surject(self.feat_map[prev_word], self.feat_map[c])
            else:
                score = 0.5
            scores.append(score)
        
        return softmax_torch(torch.tensor(scores))
    
    def K_linearize(self, context: Tuple[str, str], candidates: List[str]) -> torch.Tensor:
        base_p = self.K_similarity(context, candidates)
        
        v1 = self.feat_map.get(context[0] if len(context) > 0 else "", torch.zeros(5))
        v2 = self.feat_map.get(context[1] if len(context) > 1 else "", torch.zeros(5))
        ctx = (v1 + v2) / 2.0
        
        anchor_idx = self._nearest_anchor_idx(ctx)
        anchor = self.anchors[anchor_idx]
        
        cand_feats = [self.feat_map.get(c, torch.zeros(5)) / (torch.linalg.norm(self.feat_map.get(c, torch.zeros(5))) + 1e-9) 
                     for c in candidates]
        cand_feats = torch.stack(cand_feats)
        align_weights = torch.clamp(torch.matmul(cand_feats, anchor), min=0.0)
        align_weights = align_weights / (align_weights.sum() + 1e-9)
        
        return (1.0 - self.K_LIN) * base_p + self.K_LIN * align_weights
    
    def K_onto_balance(self, base_p: torch.Tensor, candidates: List[str], step: int) -> torch.Tensor:
        if (step + 1) % self.K_ALT != 0:
            return base_p
        
        min_hits = self.anchor_hits.min().item()
        under_indices = torch.where(self.anchor_hits == min_hits)[0]
        if len(under_indices) == 0:
            return base_p
        
        target_anchor = int(under_indices[step % len(under_indices)].item())
        anchor = self.anchors[target_anchor]
        
        cand_feats = [self.feat_map.get(c, torch.zeros(5)) / (torch.linalg.norm(self.feat_map.get(c, torch.zeros(5))) + 1e-9) 
                     for c in candidates]
        cand_feats = torch.stack(cand_feats)
        onto_weights = torch.clamp(torch.matmul(cand_feats, anchor), min=0.0)
        onto_weights = onto_weights / (onto_weights.sum() + 1e-9)
        
        return (1.0 - self.K_SIM) * base_p + self.K_SIM * onto_weights

# Generation (seed-aware)
def generation_homomorphism(kernel: SurjectionKernels, seed_words: List[str], length: int, gen_seed: int) -> str:
    seed_everything(gen_seed)  # Per-generation seed
    state = seed_words[-2:] if len(seed_words) >= 2 else ["the", "a"]
    output = state.copy()
    kernel.generation_state = state.copy()
    
    for step in range(length):
        candidates = kernel.model.get(tuple(state[-2:]), kernel.tokens[:10])
        if not candidates:
            candidates = ["the", "a", "is", "in", "to"]
        
        p_sim = kernel.K_similarity(tuple(state[-2:]), candidates)
        p_lin = kernel.K_linearize(tuple(state[-2:]), candidates)
        p_final = kernel.K_onto_balance(p_lin, candidates, step)
        
        p_final = torch.clamp(p_final, min=1e-12)
        p_final = p_final / p_final.sum()
        next_idx = torch.multinomial(p_final, 1).item()
        next_word = candidates[next_idx]
        
        if next_word in kernel.feat_map:
            feat = kernel.feat_map[next_word]
            anchor_idx = kernel._nearest_anchor_idx(feat)
            kernel.anchor_hits[anchor_idx] += 1
        
        state.append(next_word)
        output.append(next_word)
        kernel.generation_state = state[-10:]
    
    return " ".join(output)

def build_ngram(tokens: List[str], n=2) -> Dict:
    if len(tokens) < n + 1:
        return {("the",): ["a", "is"], ("a",): ["the", "cat"]}
    
    model = defaultdict(list)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i+n])
        next_token = tokens[i+n]
        model[context].append(next_token)
    return dict(model)

# MAIN HANDLER (with seed)
def generate_from_file(file_obj, seed_text, length, random_seed):
    try:
        if file_obj is None:
            return "Please upload a file"
        
        path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        if not os.path.exists(path):
            return "File not found"
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().lower()
        
        tokens = text.split()
        if len(tokens) < 10:
            return "Corpus too small. Need at least 10 tokens."
        
        model = build_ngram(tokens)
        kernels = SurjectionKernels(tokens, model, int(random_seed))
        
        seed_words = seed_text.split()
        result = generation_homomorphism(kernels, seed_words, int(length), int(random_seed))
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

# ENHANCED GRADIO APP
def build_app():
    with gr.Blocks(title="🔮 Homomorphic Text Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # **Φ = K₃ ∘ K₂ ∘ K₁** *(Reproducible)*
        
        **Three tensor kernels + Full seed control**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                infile = gr.File(label="📄 Corpus (.txt)", file_types=[".txt"])
            
            with gr.Column(scale=1):
                seed_text = gr.Textbox(label="🌱 Seed Words", value="the quick brown", 
                                     placeholder="Enter 2-4 words")
                length_slider = gr.Slider(50, 500, value=150, step=25, label="📏 Length")
                random_seed = gr.Number(label="🎲 Random Seed", value=42, precision=0, 
                                      info="Same seed = same output!")
        
        with gr.Row():
            btn_generate = gr.Button("✨ Generate Φ(output)", variant="primary", size="lg")
        
        output = gr.Textbox(label="🎭 Generated Text", lines=12)
        
        # SEED REPRODUCIBILITY DEMO
        with gr.Row():
            gr.Markdown("### 🔄 **Reproducibility Test**")
            with gr.Column():
                seed_42 = gr.Number(value=42, label="Seed: 42", precision=0, interactive=True)
                btn_seed42 = gr.Button("Generate 42", variant="secondary")
            with gr.Column():
                seed_123 = gr.Number(value=123, label="Seed: 123", precision=0, interactive=True)
                btn_seed123 = gr.Button("Generate 123", variant="secondary")
        
        # Event handlers
        btn_generate.click(generate_from_file, 
                          [infile, seed_text, length_slider, random_seed], output)
        btn_seed42.click(generate_from_file, 
                        [infile, seed_text, length_slider, seed_42], output)
        btn_seed123.click(generate_from_file, 
                         [infile, seed_text, length_slider, seed_123], output)
        
        gr.Markdown("""
        **Reproducibility guaranteed:** Same seed + same inputs = identical output
        
        **Kernels:** K₁(similarity) → K₂(linearize) → K₃(balance)
        """)
    
    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
