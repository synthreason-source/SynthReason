#!/usr/bin/env python3
# MATH CONCEPT LLM BUILDER V4.1 - FULL CODE (100 concepts) + SINGLE-SHOT DATASET LOADING
# Perplexity API (optional) → generate concept-infused model code → load dataset (ONE attempt per alias per run) → train → ROUGE-L
#
# Install:
#   pip install torch datasets rouge-score numpy pplx
#
# Env:
#   set PPLX_API_KEY=... (Windows) / export PPLX_API_KEY=... (Linux/macOS)
#
# HF cache control (recommended) [web:92]
#   set HF_HOME=.\hf_home
#   set HF_HUB_CACHE=.\hf_hub_cache
#   set HF_DATASETS_CACHE=.\hf_datasets_cache

import os
import sys
import time
import random
import warnings
import traceback
import itertools
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------
# Optional: Perplexity API
# ---------------------------
PPLX_AVAILABLE = False
try:
    import pplx  # pip install pplx
    PPLX_AVAILABLE = True
except Exception:
    pass

API_KEY = os.getenv("PPLX_API_KEY", "").strip()

# ---------------------------
# Stable HF cache dirs [web:92]
# ---------------------------
os.environ.setdefault("HF_HOME", os.path.abspath("./hf_home"))
os.environ.setdefault("HF_HUB_CACHE", os.path.abspath("./hf_hub_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("./hf_datasets_cache"))

# ---------------------------
# 100 concepts mapping
# ---------------------------
MATH_CONCEPTS_100 = {
    # Linear Algebra (20)
    "Vectors": "gsm8k",
    "Matrices": "competition_math",
    "MatrixMultiplication": "codealpaca",
    "DotProduct": "gsm8k",
    "Eigenvalues": "competition_math",
    "Eigenvectors": "math_strong",
    "SingularValueDecomposition": "codealpaca",
    "PrincipalComponentAnalysis": "competition_math",
    "VectorNorms": "gsm8k",
    "MatrixRank": "math_strong",
    "LinearTransformations": "codealpaca",
    "Determinant": "competition_math",
    "InverseMatrix": "math_strong",
    "OrthogonalMatrices": "codealpaca",
    "SymmetricMatrices": "competition_math",
    "LUDecomposition": "math_strong",
    "QRDecomposition": "codealpaca",
    "CholeskyDecomposition": "competition_math",
    "TraceMatrix": "gsm8k",
    "Transpose": "math_strong",

    # Calculus (15)
    "PartialDerivatives": "gsm8k",
    "Gradient": "competition_math",
    "ChainRule": "codealpaca",
    "HessianMatrix": "math_strong",
    "TaylorSeries": "competition_math",
    "Integration": "gsm8k",
    "MultivariateCalculus": "math_strong",
    "JacobianMatrix": "codealpaca",
    "Laplacian": "competition_math",
    "DirectionalDerivative": "gsm8k",
    "VectorCalculus": "math_strong",
    "Divergence": "competition_math",
    "Curl": "codealpaca",
    "GreenTheorem": "gsm8k",
    "StokesTheorem": "math_strong",

    # Probability & Stats (20)
    "Probability": "gsm8k",
    "BayesTheorem": "competition_math",
    "ConditionalProbability": "math_strong",
    "RandomVariables": "codealpaca",
    "ExpectedValue": "gsm8k",
    "Variance": "competition_math",
    "StandardDeviation": "math_strong",
    "Covariance": "codealpaca",
    "Correlation": "gsm8k",
    "GaussianDistribution": "competition_math",
    "BinomialDistribution": "math_strong",
    "PoissonDistribution": "codealpaca",
    "BernoulliDistribution": "gsm8k",
    "CentralLimitTheorem": "competition_math",
    "LawLargeNumbers": "math_strong",
    "ConfidenceIntervals": "codealpaca",
    "HypothesisTesting": "gsm8k",
    "PValue": "competition_math",
    "MarkovChains": "math_strong",
    "Entropy": "codealpaca",

    # Optimization (15)
    "GradientDescent": "gsm8k",
    "StochasticGradientDescent": "competition_math",
    "Momentum": "codealpaca",
    "AdamOptimizer": "math_strong",
    "RMSprop": "gsm8k",
    "LearningRate": "competition_math",
    "ConvexOptimization": "codealpaca",
    "LagrangeMultipliers": "math_strong",
    "KKTConditions": "competition_math",
    "LineSearch": "gsm8k",
    "TrustRegion": "codealpaca",
    "ConjugateGradient": "math_strong",
    "NewtonsMethod": "competition_math",
    "QuasiNewton": "gsm8k",
    "BatchNormalization": "codealpaca",

    # Neural Networks (20)
    "Backpropagation": "gsm8k",
    "ForwardPropagation": "competition_math",
    "ActivationFunctions": "codealpaca",
    "ReLU": "math_strong",
    "SigmoidActivation": "gsm8k",
    "TanhActivation": "competition_math",
    "SoftmaxFunction": "codealpaca",
    "AttentionMechanism": "competition_math",
    "SelfAttention": "math_strong",
    "MultiHeadAttention": "codealpaca",
    "LayerNormalization": "gsm8k",
    "Dropout": "competition_math",
    "WeightInitialization": "math_strong",
    "LossFunctions": "codealpaca",
    "CrossEntropyLoss": "gsm8k",
    "MeanSquaredError": "competition_math",
    "Overfitting": "math_strong",
    "Regularization": "codealpaca",
    "L1Regularization": "gsm8k",
    "L2Regularization": "competition_math",

    # Advanced (10)
    "FourierTransform": "math_strong",
    "Convolution": "codealpaca",
    "InformationTheory": "competition_math",
    "KLDiveregence": "gsm8k",
    "MutualInformation": "math_strong",
    "GraphTheory": "competition_math",
    "TensorOperations": "codealpaca",
    "ManifoldLearning": "math_strong",
    "TopologicalDataAnalysis": "competition_math",
    "LieGroups": "gsm8k",
}

# ---------------------------
# Dataset plans (known-good IDs)
# ---------------------------
DATASET_PLANS = {
    # GSM8K: openai/gsm8k with config "main" [web:58]
    "gsm8k": [
        ("openai/gsm8k", "main", "train"),
        ("openai/gsm8k", "main", "test"),
    ],
    "competition_math": [
        ("hendrycks/competition_math", None, "train"),
        ("hendrycks/competition_math", None, "test"),
    ],
    "codealpaca": [
        ("sahil2801/CodeAlpaca-20k", None, "train"),
    ],
    # Real math fallbacks (avoid ambiguous "math_dataset") [web:48][web:54][web:51]
    "math_strong": [
        ("HuggingFaceH4/MATH-500", None, "test"),
        ("openbmb/UltraData-Math", None, "train"),
        ("tokyotech-llm/swallow-math", None, "train"),
    ],
}

# ---------------------------
# SINGLE-SHOT per-alias caching
# ---------------------------
_DATASET_RESULT_CACHE = {}
_DATASET_FAILED_ONCE = set()

def _normalize_qa(row: dict):
    if "question" in row and "answer" in row:
        return {"question": str(row["question"]), "answer": str(row["answer"])}

    if "problem" in row and ("solution" in row or "answer" in row):
        return {"question": str(row["problem"]), "answer": str(row.get("solution", row.get("answer", "")))}

    if "instruction" in row and ("output" in row or "response" in row):
        return {"question": str(row["instruction"]), "answer": str(row.get("output", row.get("response", "")))}

    if "prompt" in row and ("completion" in row or "target" in row):
        return {"question": str(row["prompt"]), "answer": str(row.get("completion", row.get("target", "")))}

    keys = list(row.keys())
    q = row.get(keys[0], "Problem?") if keys else "Problem?"
    a = row.get(keys[1], "Answer.") if len(keys) > 1 else "Answer."
    return {"question": str(q), "answer": str(a)}

def _synthetic(alias, n, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        x = i + 1
        out.append(
            {
                "question": f"{alias} sample {i}: compute x^2 for x={x}. Provide a short explanation.",
                "answer": f"{x*x}. Explanation: synthetic ({rnd.random():.4f}).",
            }
        )
    return out

def safe_load_dataset_once(alias, *, n=220, seed=0, cache_dir=None):
    """
    Exactly ONE "network attempt pass" per alias per run:
    - If alias already loaded: return cached
    - If alias already failed: return cached synthetic immediately
    - Otherwise: try each candidate dataset ONCE (no retries, no streaming), then synthetic
    HF cache dirs still allow reuse across runs. [web:92]
    """
    alias_norm = (alias or "").strip().lower()

    if alias_norm in _DATASET_RESULT_CACHE:
        return _DATASET_RESULT_CACHE[alias_norm]

    if alias_norm in _DATASET_FAILED_ONCE:
        out = _synthetic(alias_norm, n, seed)
        _DATASET_RESULT_CACHE[alias_norm] = out
        return out

    _DATASET_FAILED_ONCE.add(alias_norm)

    plan = DATASET_PLANS.get(alias_norm)
    if plan is None:
        plan = [(alias, None, "train")] + DATASET_PLANS["math_strong"]

    for (ds_id, cfg, split) in plan:
        try:
            if cfg is None:
                ds = load_dataset(ds_id, split=split, streaming=False, cache_dir=cache_dir)
            else:
                ds = load_dataset(ds_id, cfg, split=split, streaming=False, cache_dir=cache_dir)

            ds = ds.shuffle(seed=seed)
            rows = ds.select(range(min(n, len(ds))))
            rows = [dict(r) for r in rows]
            out = [_normalize_qa(r) for r in rows]
            out = [x for x in out if x["question"].strip() and x["answer"].strip()]

            if len(out) >= 10:
                _DATASET_RESULT_CACHE[alias_norm] = out
                print(f"✅ dataset loaded: alias={alias_norm} -> {ds_id}" + (f"/{cfg}" if cfg else "") + f" split={split} n={len(out)}")
                return out
            else:
                print(f"❌ dataset too-small: {ds_id}" + (f"/{cfg}" if cfg else "") + f" split={split} n={len(out)}")
        except Exception as e:
            print(f"❌ dataset failed (single-shot): {ds_id}" + (f"/{cfg}" if cfg else "") + f" split={split} ({str(e)[:160]})")

    out = _synthetic(alias_norm, n, seed)
    _DATASET_RESULT_CACHE[alias_norm] = out
    print(f"🔄 synthetic used: alias={alias_norm} n={len(out)}")
    return out

# ---------------------------
# Perplexity code generator
# ---------------------------
def perplexity_generate_llm_code(concept: str) -> str:
    if not (PPLX_AVAILABLE and API_KEY):
        return f"""import torch
import torch.nn as nn

class {concept}LLM(nn.Module):
    \"\"\"Template model (API disabled). Concept hook: {concept}.\"\"\"
    def __init__(self, vocab_size=50257, d_model=256, nhead=8, nlayers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead) for _ in range(nlayers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.lm_head(x)
"""
    try:
        prompt = f"build an llm with the concept {concept}, have it output text and {concept} dataset, do the whole code, fix all the errors before writing."
        resp = pplx.generate(
            model="sonar-small-online",
            prompt=prompt,
            max_tokens=900,
            api_key=API_KEY,
        )
        text = getattr(resp.choices[0], "text", None) if getattr(resp, "choices", None) else None
        return (text or "").strip() or f"# Empty API response; concept={concept}\n"
    except Exception as e:
        return f"# API error for concept={concept}: {e}\n"

# ---------------------------
# Local toy trainer / eval
# ---------------------------
def collate_fn(batch):
    qs = [torch.tensor([ord(c) for c in item["question"][:256]], dtype=torch.long) for item in batch]
    ans = [torch.tensor([ord(c) for c in item["answer"][:256]], dtype=torch.long) for item in batch]
    qs = pad_sequence(qs, batch_first=True, padding_value=0)
    ans = pad_sequence(ans, batch_first=True, padding_value=0)
    return {"input_ids": qs, "labels": ans}

class QADataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

class SimpleGenerator(nn.Module):
    def __init__(self, vocab_size=256, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, max_new=64):
        x = self.emb(input_ids)
        out, _ = self.lstm(x)
        last = out[:, -1:, :].repeat(1, max_new, 1)
        return self.fc(last)

def train_model(model, loader, *, epochs=1, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["labels"][:, 1:].to(device)
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"   epoch {ep+1}/{epochs} loss={total/max(1,len(loader)):.3f}")
    return model

def evaluate_rouge(model, samples, *, device="cpu"):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()
    model.to(device)

    scores = []
    with torch.no_grad():
        for item in samples[:10]:
            inp = torch.tensor([[ord(c) for c in item["question"][:160]]], dtype=torch.long).to(device)
            logits = model(inp)
            ids = torch.argmax(logits[0], dim=-1).tolist()
            gen = "".join(chr(max(32, min(122, i))) for i in ids)
            ref = item["answer"]
            scores.append(scorer.score(ref, gen)["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0

# ---------------------------
# Concept pipeline
# ---------------------------
def process_concept(concept, dataset_alias, *, out_dir="out", epochs=1, n=220, seed=0, device="cpu", cache_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🧮 {concept:<25} dataset={dataset_alias}")

    code = perplexity_generate_llm_code(concept)
    code_path = os.path.join(out_dir, f"{concept}_llm.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"   📄 wrote {code_path}")

    # SINGLE-SHOT LOAD (no repeated attempts for this alias)
    items = safe_load_dataset_once(dataset_alias, n=n, seed=seed, cache_dir=cache_dir)

    # Train/eval split
    rnd = random.Random(seed)
    rnd.shuffle(items)
    cut = int(0.6 * len(items))
    train_items = items[:cut]
    test_items = items[cut:]

    loader = DataLoader(QADataset(train_items), batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = SimpleGenerator()
    model = train_model(model, loader, epochs=epochs, device=device)

    rouge = evaluate_rouge(model, test_items, device=device)
    print(f"   ✅ rougeL={rouge:.3f} n={len(items)}")

    return {"concept": concept, "dataset": dataset_alias, "rougeL": rouge, "n": len(items), "code_file": code_path}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 V4.1 run | device={device}")
    print("Dataset loading is single-shot per alias per run; failures become synthetic immediately.")
    print("HF cache dirs set (HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE). [web:92]")

    out_dir = "out"
    cache_dir = os.path.abspath("./hf_load_cache_dir")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, (concept, ds_alias) in enumerate(MATH_CONCEPTS_100.items()):
        try:
            r = process_concept(
                concept,
                ds_alias,
                out_dir=out_dir,
                epochs=1,
                n=220,
                seed=1337 + i,
                device=device,
                cache_dir=cache_dir,
            )
            results.append(r)
        except Exception as e:
            print(f"💥 {concept} hard fail: {str(e)[:220]}")
            print(traceback.format_exc()[:1200])

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "dataset", "rougeL", "n", "code_file"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n📦 wrote {csv_path} rows={len(results)}")

    top = sorted(results, key=lambda x: x["rougeL"], reverse=True)[:10]
    print("\n🏁 TOP 10")
    for r in top:
        print(f"  {r['concept']:<25} rougeL={r['rougeL']:.3f} dataset={r['dataset']} n={r['n']}")

if __name__ == "__main__":
    main()
