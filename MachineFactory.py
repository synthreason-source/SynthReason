#!/usr/bin/env python3
# MATH CONCEPT LLM BUILDER V4.1 - FIXED
# Mass-produces concept-infused LLM code for 100 math concepts, trains tiny models, scores with ROUGE-L.
#
# Install:
#   pip install torch datasets rouge-score numpy openai
#   pip install pplx  (optional, for Perplexity API)
#
# Env:
#   export PPLX_API_KEY=...   (Linux/macOS)
#   set    PPLX_API_KEY=...   (Windows)
#
# Recommended HF cache control:
#   export HF_HOME=./hf_home
#   export HF_HUB_CACHE=./hf_hub_cache
#   export HF_DATASETS_CACHE=./hf_datasets_cache

import os
import sys
import time
import random
import warnings
import traceback
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

# ─── Optional Perplexity API ────────────────────────────────────────────────
PPLX_AVAILABLE = False
pplx_client = None
try:
    # pplx library wraps the OpenAI-compatible Perplexity endpoint
    from openai import OpenAI
    API_KEY = os.getenv("PPLX_API_KEY", "").strip()
    if API_KEY:
        pplx_client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.perplexity.ai",
        )
        PPLX_AVAILABLE = True
except Exception:
    API_KEY = ""

# ─── Stable HF cache dirs ───────────────────────────────────────────────────
os.environ.setdefault("HF_HOME",            os.path.abspath("./hf_home"))
os.environ.setdefault("HF_HUB_CACHE",       os.path.abspath("./hf_hub_cache"))
os.environ.setdefault("HF_DATASETS_CACHE",  os.path.abspath("./hf_datasets_cache"))

# ─── 100 Concepts → dataset alias ───────────────────────────────────────────
MATH_CONCEPTS_100 = {
    # Linear Algebra (20)
    "Vectors":                   "gsm8k",
    "Matrices":                  "competition_math",
    "MatrixMultiplication":      "codealpaca",
    "DotProduct":                "gsm8k",
    "Eigenvalues":               "competition_math",
    "Eigenvectors":              "math_strong",
    "SingularValueDecomposition":"codealpaca",
    "PrincipalComponentAnalysis":"competition_math",
    "VectorNorms":               "gsm8k",
    "MatrixRank":                "math_strong",
    "LinearTransformations":     "codealpaca",
    "Determinant":               "competition_math",
    "InverseMatrix":             "math_strong",
    "OrthogonalMatrices":        "codealpaca",
    "SymmetricMatrices":         "competition_math",
    "LUDecomposition":           "math_strong",
    "QRDecomposition":           "codealpaca",
    "CholeskyDecomposition":     "competition_math",
    "TraceMatrix":               "gsm8k",
    "Transpose":                 "math_strong",
    # Calculus (15)
    "PartialDerivatives":        "gsm8k",
    "Gradient":                  "competition_math",
    "ChainRule":                 "codealpaca",
    "HessianMatrix":             "math_strong",
    "TaylorSeries":              "competition_math",
    "Integration":               "gsm8k",
    "MultivariateCalculus":      "math_strong",
    "JacobianMatrix":            "codealpaca",
    "Laplacian":                 "competition_math",
    "DirectionalDerivative":     "gsm8k",
    "VectorCalculus":            "math_strong",
    "Divergence":                "competition_math",
    "Curl":                      "codealpaca",
    "GreenTheorem":              "gsm8k",
    "StokesTheorem":             "math_strong",
    # Probability & Stats (20)
    "Probability":               "gsm8k",
    "BayesTheorem":              "competition_math",
    "ConditionalProbability":    "math_strong",
    "RandomVariables":           "codealpaca",
    "ExpectedValue":             "gsm8k",
    "Variance":                  "competition_math",
    "StandardDeviation":         "math_strong",
    "Covariance":                "codealpaca",
    "Correlation":               "gsm8k",
    "GaussianDistribution":      "competition_math",
    "BinomialDistribution":      "math_strong",
    "PoissonDistribution":       "codealpaca",
    "BernoulliDistribution":     "gsm8k",
    "CentralLimitTheorem":       "competition_math",
    "LawLargeNumbers":           "math_strong",
    "ConfidenceIntervals":       "codealpaca",
    "HypothesisTesting":         "gsm8k",
    "PValue":                    "competition_math",
    "MarkovChains":              "math_strong",
    "Entropy":                   "codealpaca",
    # Optimization (15)
    "GradientDescent":           "gsm8k",
    "StochasticGradientDescent": "competition_math",
    "Momentum":                  "codealpaca",
    "AdamOptimizer":             "math_strong",
    "RMSprop":                   "gsm8k",
    "LearningRate":              "competition_math",
    "ConvexOptimization":        "codealpaca",
    "LagrangeMultipliers":       "math_strong",
    "KKTConditions":             "competition_math",
    "LineSearch":                "gsm8k",
    "TrustRegion":               "codealpaca",
    "ConjugateGradient":         "math_strong",
    "NewtonsMethod":             "competition_math",
    "QuasiNewton":               "gsm8k",
    "BatchNormalization":        "codealpaca",
    # Neural Networks (20)
    "Backpropagation":           "gsm8k",
    "ForwardPropagation":        "competition_math",
    "ActivationFunctions":       "codealpaca",
    "ReLU":                      "math_strong",
    "SigmoidActivation":         "gsm8k",
    "TanhActivation":            "competition_math",
    "SoftmaxFunction":           "codealpaca",
    "AttentionMechanism":        "competition_math",
    "SelfAttention":             "math_strong",
    "MultiHeadAttention":        "codealpaca",
    "LayerNormalization":        "gsm8k",
    "Dropout":                   "competition_math",
    "WeightInitialization":      "math_strong",
    "LossFunctions":             "codealpaca",
    "CrossEntropyLoss":          "gsm8k",
    "MeanSquaredError":          "competition_math",
    "Overfitting":               "math_strong",
    "Regularization":            "codealpaca",
    "L1Regularization":          "gsm8k",
    "L2Regularization":          "competition_math",
    # Advanced (10)
    "FourierTransform":          "math_strong",
    "Convolution":               "codealpaca",
    "InformationTheory":         "competition_math",
    "KLDivergence":              "gsm8k",          # fixed typo: KLDiveregence → KLDivergence
    "MutualInformation":         "math_strong",
    "GraphTheory":               "competition_math",
    "TensorOperations":          "codealpaca",
    "ManifoldLearning":          "math_strong",
    "TopologicalDataAnalysis":   "competition_math",
    "LieGroups":                 "gsm8k",
}

# ─── Dataset plans (lower-cased keys to match normalisation) ────────────────
DATASET_PLANS = {
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
    "math_strong": [
        ("HuggingFaceH4/MATH-500",        None, "test"),
        ("openbmb/UltraData-Math",         None, "train"),
        ("tokyotech-llm/swallow-math",     None, "train"),
    ],
}

# ─── Single-shot dataset cache ───────────────────────────────────────────────
_DATASET_RESULT_CACHE: dict = {}
_DATASET_FAILED_ONCE:  set  = set()


def _normalize_qa(row: dict) -> dict:
    if "question" in row and "answer" in row:
        return {"question": str(row["question"]), "answer": str(row["answer"])}
    if "problem" in row and ("solution" in row or "answer" in row):
        return {"question": str(row["problem"]),
                "answer":   str(row.get("solution", row.get("answer", "")))}
    if "instruction" in row and ("output" in row or "response" in row):
        return {"question": str(row["instruction"]),
                "answer":   str(row.get("output", row.get("response", "")))}
    if "prompt" in row and ("completion" in row or "target" in row):
        return {"question": str(row["prompt"]),
                "answer":   str(row.get("completion", row.get("target", "")))}
    keys = list(row.keys())
    q = str(row.get(keys[0], "Problem?")) if keys else "Problem?"
    a = str(row.get(keys[1], "Answer.")) if len(keys) > 1 else "Answer."
    return {"question": q, "answer": a}


def _synthetic(alias: str, n: int, seed: int) -> list:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        x = i + 1
        out.append({
            "question": f"{alias} sample {i}: compute x^2 for x={x}. Provide a short explanation.",
            "answer":   f"{x*x}. Explanation: synthetic ({rnd.random():.4f}).",
        })
    return out


def safe_load_dataset_once(alias: str, *, n: int = 220, seed: int = 0,
                            cache_dir: str | None = None) -> list:
    """
    Exactly ONE network-attempt pass per alias per run.
    BUG FIX: do NOT add to _DATASET_FAILED_ONCE before trying —
    only mark failed after all candidates are exhausted.
    """
    key = (alias or "").strip().lower()

    if key in _DATASET_RESULT_CACHE:
        return _DATASET_RESULT_CACHE[key]

    if key in _DATASET_FAILED_ONCE:
        out = _synthetic(key, n, seed)
        _DATASET_RESULT_CACHE[key] = out
        return out

    # BUG FIX (original): alias was added to failed set *before* trying,
    # so the first real attempt was skipped. Now we only mark failure after
    # all candidates are exhausted.
    plan = DATASET_PLANS.get(key)
    if plan is None:
        plan = [(alias, None, "train")] + DATASET_PLANS["math_strong"]

    for (ds_id, cfg, split) in plan:
        try:
            if cfg is None:
                ds = load_dataset(ds_id, split=split, streaming=False, cache_dir=cache_dir)
            else:
                ds = load_dataset(ds_id, cfg, split=split, streaming=False, cache_dir=cache_dir)

            ds    = ds.shuffle(seed=seed)
            rows  = ds.select(range(min(n, len(ds))))
            rows  = [dict(r) for r in rows]
            out   = [_normalize_qa(r) for r in rows]
            out   = [x for x in out if x["question"].strip() and x["answer"].strip()]

            if len(out) >= 10:
                _DATASET_RESULT_CACHE[key] = out
                tag = f"/{cfg}" if cfg else ""
                print(f"✅ dataset loaded: alias={key} -> {ds_id}{tag} split={split} n={len(out)}")
                return out
            else:
                tag = f"/{cfg}" if cfg else ""
                print(f"❌ dataset too-small: {ds_id}{tag} split={split} n={len(out)}")

        except Exception as e:
            tag = f"/{cfg}" if cfg else ""
            print(f"❌ dataset failed (single-shot): {ds_id}{tag} split={split} ({str(e)[:160]})")

    # All candidates exhausted → mark failed and use synthetic
    _DATASET_FAILED_ONCE.add(key)
    out = _synthetic(key, n, seed)
    _DATASET_RESULT_CACHE[key] = out
    print(f"🔄 synthetic used: alias={key} n={len(out)}")
    return out


# ─── Perplexity code generator ───────────────────────────────────────────────
def perplexity_generate_llm_code(concept: str) -> str:
    """
    BUG FIX: original code called pplx.generate() which doesn't exist.
    Now uses the OpenAI-compatible client pointed at api.perplexity.ai.
    Falls back to a clean template if API is unavailable.
    """
    template = f'''import torch
import torch.nn as nn


class {concept}LLM(nn.Module):
    """
    Concept-infused LLM stub: {concept}.
    Replace / extend the architecture as needed.
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 256,
                 nhead: int = 8, nlayers: int = 4, max_seq: int = 512):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        enc_layer      = nn.TransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward=d_model * 4,
                                                    batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.ln       = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos   = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x     = self.embed(input_ids) + self.pos_embed(pos)
        x     = self.encoder(x)
        x     = self.ln(x)
        return self.lm_head(x)          # (B, T, vocab_size)


if __name__ == "__main__":
    model  = {concept}LLM()
    tokens = torch.randint(0, 50257, (2, 32))
    logits = model(tokens)
    print(f"logits shape: {{logits.shape}}")
'''

    if not (PPLX_AVAILABLE and pplx_client):
        return template

    try:
        prompt = (
            f"Write a complete, self-contained PyTorch LLM class whose architecture "
            f"is specifically inspired by the mathematical concept '{concept}'. "
            f"Include __init__, forward, and a small __main__ demo. "
            f"Output only valid Python; no markdown fences."
        )
        resp = pplx_client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        text = resp.choices[0].message.content.strip()
        return text if text else template
    except Exception as e:
        print(f"  ⚠️  Perplexity API error for {concept}: {e}")
        return template


# ─── Dataset / DataLoader ────────────────────────────────────────────────────
MAX_NEW = 64   # tokens the model generates — must match train + eval


def collate_fn(batch: list) -> dict:
    qs  = [torch.tensor([ord(c) % 256 for c in item["question"][:256]],
                        dtype=torch.long) for item in batch]
    ans = [torch.tensor([ord(c) % 256 for c in item["answer"][:MAX_NEW + 1]],
                        dtype=torch.long) for item in batch]
    qs  = pad_sequence(qs,  batch_first=True, padding_value=0)
    ans = pad_sequence(ans, batch_first=True, padding_value=0)
    return {"input_ids": qs, "labels": ans}


class QADataset(Dataset):
    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ─── Model ───────────────────────────────────────────────────────────────────
VOCAB = 256   # byte-level


class SimpleGenerator(nn.Module):
    """
    Tiny seq2seq: encodes question with LSTM, decodes MAX_NEW tokens.
    """

    def __init__(self, vocab_size: int = VOCAB, d_model: int = 256):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.enc    = nn.LSTM(d_model, d_model, num_layers=2,
                              batch_first=True, dropout=0.1)
        self.dec    = nn.LSTM(d_model, d_model, num_layers=2,
                              batch_first=True, dropout=0.1)
        self.fc     = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor,
                max_new: int = MAX_NEW) -> torch.Tensor:
        """
        Returns logits of shape (B, max_new, vocab_size).
        BUG FIX (original): model returned (B, max_new, vocab) but train_model
        tried to align it with labels[:, 1:] which could be up to 255 tokens —
        causing shape mismatches in the loss. Now both encoder and training use
        the same MAX_NEW constant.
        """
        x, (h, c)  = self.enc(self.emb(input_ids))
        # Seed decoder with last encoder hidden state
        start      = x[:, -1:, :]                   # (B, 1, d_model)
        dec_in     = start.expand(-1, max_new, -1)  # (B, max_new, d_model)
        out, _     = self.dec(dec_in, (h, c))
        return self.fc(out)                          # (B, max_new, vocab_size)


# ─── Training ────────────────────────────────────────────────────────────────
def train_model(model: nn.Module, loader: DataLoader, *,
                epochs: int = 1, lr: float = 1e-3,
                device: str = "cpu") -> nn.Module:
    model.to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            inp    = batch["input_ids"].to(device)          # (B, Tq)
            labels = batch["labels"].to(device)             # (B, Tl)

            logits = model(inp)                             # (B, MAX_NEW, V)
            B, G, V = logits.shape

            # BUG FIX (original): labels could be shorter or longer than MAX_NEW.
            # Align by taking min(G, labels.size(1)) columns.
            tgt_len = min(G, labels.size(1))
            tgt     = labels[:, :tgt_len]                   # (B, tgt_len)
            log_cut = logits[:, :tgt_len, :]                # (B, tgt_len, V)

            loss = loss_fn(log_cut.reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

        print(f"  epoch {ep+1}/{epochs} loss={total/max(1, len(loader)):.4f}")

    return model


# ─── Evaluation ─────────────────────────────────────────────────────────────
def evaluate_rouge(model: nn.Module, samples: list, *,
                   device: str = "cpu") -> float:
    scorer_ = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()
    model.to(device)
    scores = []

    with torch.no_grad():
        for item in samples[:20]:
            inp    = torch.tensor(
                [[ord(c) % 256 for c in item["question"][:160]]],
                dtype=torch.long, device=device,
            )
            logits = model(inp)                          # (1, MAX_NEW, V)
            ids    = torch.argmax(logits[0], dim=-1).tolist()
            # BUG FIX (original): chr was clamped to [32,122]; valid printable
            # ASCII goes up to 126. Use 32–126 range.
            gen    = "".join(chr(max(32, min(126, i))) for i in ids)
            ref    = item["answer"]
            scores.append(scorer_.score(ref, gen)["rougeL"].fmeasure)

    return float(np.mean(scores)) if scores else 0.0


# ─── Per-concept pipeline ────────────────────────────────────────────────────
def process_concept(concept: str, dataset_alias: str, *,
                    out_dir: str = "out", epochs: int = 1,
                    n: int = 220, seed: int = 0,
                    device: str = "cpu",
                    cache_dir: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🧮 {concept:<30} dataset={dataset_alias}")

    # 1. Generate (or template) model code
    code      = perplexity_generate_llm_code(concept)
    code_path = os.path.join(out_dir, f"{concept}_llm.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  📄 wrote {code_path}")

    # 2. Load dataset — single-shot, never retried
    raw_items = safe_load_dataset_once(dataset_alias, n=n, seed=seed,
                                       cache_dir=cache_dir)

    # BUG FIX (original): items list was shared with the cache; in-place
    # shuffle mutated the cached copy so subsequent aliases got a different
    # ordering. Now we copy first.
    items = list(raw_items)
    rnd   = random.Random(seed)
    rnd.shuffle(items)

    cut         = int(0.6 * len(items))
    train_items = items[:cut]
    test_items  = items[cut:]

    loader = DataLoader(
        QADataset(train_items),
        batch_size=8, shuffle=True,
        collate_fn=collate_fn,
    )

    # 3. Train
    model  = SimpleGenerator()
    model  = train_model(model, loader, epochs=epochs, device=device)

    # 4. Eval
    rouge  = evaluate_rouge(model, test_items, device=device)
    print(f"  ✅ rougeL={rouge:.4f}  n_total={len(items)}")

    return {
        "concept":   concept,
        "dataset":   dataset_alias,
        "rougeL":    rouge,
        "n":         len(items),
        "code_file": code_path,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Math Concept LLM Builder V4.1 (FIXED) | device={device}")
    print("   Single-shot dataset loading; failures → synthetic immediately.")
    print("   HF cache dirs: HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE")

    out_dir   = "out"
    cache_dir = os.path.abspath("./hf_load_cache_dir")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, (concept, ds_alias) in enumerate(MATH_CONCEPTS_100.items()):
        try:
            r = process_concept(
                concept, ds_alias,
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

    # Write CSV
    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = ["concept", "dataset", "rougeL", "n", "code_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n📦 wrote {csv_path}  rows={len(results)}")

    # Top 10
    top = sorted(results, key=lambda x: x["rougeL"], reverse=True)[:10]
    print("\n🏁 TOP 10 by ROUGE-L")
    for r in top:
        print(f"  {r['concept']:<30} rougeL={r['rougeL']:.4f}"
              f"  dataset={r['dataset']}  n={r['n']}")


if __name__ == "__main__":
    main()
