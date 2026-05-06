# LLaMA Model Architecture Explained

🌍 [中文版本](./README.md) | English Version

A modular breakdown of the four major architectural improvements of LLaMA compared to GPT (RMSNorm / RoPE / SwiGLU / GQA), from mathematical formulas to code skeletons, providing a comprehensive understanding of the design motivation behind each improvement.
---

## 📚 Table of Contents
- [LLaMA Architecture Overview](#llama-architecture-overview)
- [Core Architectural Improvements Comparison](#core-architectural-improvements-comparison)
- [Complete Data Flow Trace](#complete-data-flow-trace)
- [Code-level Differences Preview](#code-level-differences-preview)

---

## LLaMA Architecture Overview
Like GPT, LLaMA adopts a Decoder-only architecture, but each component has been carefully optimized:

```
Input token IDs: [x_1, x_2, ..., x_T]
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  E_tok ∈ R^{V × d}
│  (No position encoding added here)│  ← Note: RoPE is applied in Attention!
└──────────────────┘
       │
       ▼  h_0 = E_tok[x]
┌──────────────────┐
│  LLaMA Block × N  │  ← Repeat N times (LLaMA-7B: N=32)
│  ┌──────────────┐ │
│  │  RMSNorm 1   │ │  ← Replaces LayerNorm
│  │  GQA + RoPE  │ │  ← Replaces MHA + learnable position encoding
│  │  + Residual    │ │
│  ├──────────────┤ │
│  │  RMSNorm 2   │ │
│  │  SwiGLU FFN  │ │  ← Replaces GELU FFN
│  │  + Residual    │ │
│  └──────────────┘ │
└──────────────────┘
       │
       ▼  h_N
┌──────────────────┐
│   RMSNorm         │  ← Final normalization
│   LM Head (Linear)│  W ∈ R^{d × V}, not shared with E_tok
└──────────────────┘
       │
       ▼
  logits ∈ R^{T × V}  → softmax → P(x_{t+1} | x_{≤t})
```

**Key Hyperparameter Reference Table**:

| Hyperparameter | Symbol | LLaMA-7B | LLaMA-13B | LLaMA-33B | LLaMA-65B |
|--------|------|----------|-----------|-----------|-----------|
| Number of Layers | $N$ | 32 | 40 | 60 | 80 |
| Hidden Dimension | $d$ | 4096 | 5120 | 6656 | 8192 |
| Number of Attention Heads | $n_h$ | 32 | 40 | 52 | 64 |
| Per-head Dimension | $d_k = d/n_h$ | 128 | 128 | 128 | 128 |
| FFN Intermediate Dimension | $d_{ff}$ | 11008 | 13824 | 17920 | 22016 |
| Context Length | $T_{max}$ | 2048 | 2048 | 2048 | 2048 |
| Vocabulary Size | $V$ | 32000 | 32000 | 32000 | 32000 |
| GQA KV Heads | $n_{kv}$ | 32 (MHA) | 40 (MHA) | 52 (MHA) | 64 (MHA) |
| Total Parameters | — | 6.7B | 13.0B | 32.5B | 65.2B |

Note: LLaMA-1 uses full MHA (each head has independent KV), LLaMA-2 70B introduces GQA (8 KV heads).

---

## Core Architectural Improvements Comparison
LLaMA-1 is based on the Transformer Decoder-only architecture, but has four key improvements compared to GPT:

| Improvement | GPT-2/3 Implementation | LLaMA Implementation | Reference |
|------|---------|-------|------|
| **Normalization** | LayerNorm (Post-Norm) | **RMSNorm (Pre-Norm)** | GPT-3 uses Pre-LN, LLaMA further simplifies to RMSNorm |
| **Activation Function** | GELU | **SwiGLU** | Shazeer (2020), also used in PaLM |
| **Position Encoding** | Learnable absolute position encoding | **RoPE (Rotary Position Embedding)** | Su et al. (2021) |
| **Attention** | MHA (all heads have independent KV) | **MHA** (LLaMA-1), **GQA** (LLaMA-2) | GQA: Ainslie et al. (2023) |
| **Bias** | All linear layers have bias | **No bias in any linear layer** | Reduces parameters, no performance loss in practice |
| **Weight Sharing** | Embedding = LM Head | **No sharing** | LLaMA chooses not to share weights |

### LLaMA vs GPT Technical Differences Summary
| Comparison Dimension | GPT-2/3 | LLaMA-1 | Improvement Motivation |
|------|---------|---------|---------|
| **Normalization** | LayerNorm | RMSNorm | Removes mean centering, faster computation, comparable performance |
| **Norm Position** | Pre-Norm (GPT-2+) | Pre-Norm | Same as GPT |
| **Position Encoding** | Learnable absolute encoding | RoPE | Supports length extrapolation, encodes relative position information |
| **Activation Function** | GELU | SwiGLU | Gating mechanism + Swish, better experimental performance |
| **FFN Structure** | 2 linear layers | 3 linear layers (gated) | SwiGLU requires an additional linear layer |
| **$d_{ff}$** | $4d$ | $\frac{8}{3}d$ (rounded) | Maintains same parameter count as 2-layer FFN |
| **Attention** | MHA | MHA (v1) / GQA (v2) | GQA reduces KV Cache overhead |
| **Bias** | Present | Absent | Reduces parameters, no performance loss |
| **Weight Sharing** | Emb = LM Head | No sharing | Better performance in large models without sharing |
| **Tokenizer** | GPT-2 BPE (50257) | SentencePiece BPE (32000) | Smaller vocabulary + byte fallback |
| **Context Length** | 1024 (GPT-2) / 2048 (GPT-3) | 2048 (v1) / 4096 (v2) | Longer context window |
| **Training Data** | Proprietary data | Public data | Reproducible results |

---

## Complete Data Flow Trace
Take LLaMA-7B as an example, complete data flow for input `"The cat sat"`:

```
Input token IDs: [450, 6635, 3290]      # SentencePiece tokenize
                 shape: (1, 3)

  ┌─── Token Embedding ─────────────────────────┐
  │ E_tok[450], E_tok[6635], E_tok[3290]        │
  │ shape: (1, 3, 4096)                          │
  │ Note: No position encoding added! RoPE is applied in Attention  │
  └─────────────────────────────────────────────┘

  h_0: shape (1, 3, 4096)

  ─── Block 1 ──────────────────────────────────────
  │ RMSNorm_1:    (1, 3, 4096) → (1, 3, 4096)
  │ Q projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 heads
  │ K projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 KV heads (MHA)
  │ V projection: (1, 3, 4096) → (1, 3, 32, 128)
  │
  │ Apply RoPE to Q, K:
  │   Q: (1, 3, 32, 128) → Rotated → (1, 3, 32, 128)
  │   K: (1, 3, 32, 128) → Rotated → (1, 3, 32, 128)
  │
  │ Attention:
  │   Q·K^T/√128: (1, 32, 3, 128)×(1, 32, 128, 3) → (1, 32, 3, 3)
  │   + Causal Mask + Softmax → (1, 32, 3, 3)
  │   × V: (1, 32, 3, 3)×(1, 32, 3, 128) → (1, 32, 3, 128)
  │   Concat: → (1, 3, 4096)
  │   W_O: → (1, 3, 4096)
  │ + Residual → (1, 3, 4096)
  │
  │ RMSNorm_2:    (1, 3, 4096) → (1, 3, 4096)
  │ SwiGLU FFN:
  │   W_gate + SiLU: (1, 3, 4096) → (1, 3, 11008) → SiLU
  │   W_up:          (1, 3, 4096) → (1, 3, 11008)
  │   Element-wise multiply:      (1, 3, 11008)
  │   W_down:        (1, 3, 11008) → (1, 3, 4096)
  │ + Residual → (1, 3, 4096)
  ──────────────────────────────────────────────────

  ... (Repeat 32 times) ...

  h_32: shape (1, 3, 4096)

  Final RMSNorm: (1, 3, 4096) → (1, 3, 4096)
  LM Head:       (1, 3, 4096) × (4096, 32000) → (1, 3, 32000)

  logits[:, -1, :] → softmax → P(next_token | "The cat sat")
```

---

## Code-level Differences Preview
```python
# ========== Normalization ==========
# GPT: LayerNorm
x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + eps).sqrt()
x = gamma * x + beta  # Has beta offset

# LLaMA: RMSNorm (simpler)
x = x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
x = gamma * x          # No beta offset

# ========== FFN ==========
# GPT: standard FFN
h = GELU(x @ W1 + b1) @ W2 + b2

# LLaMA: SwiGLU FFN (3 weight matrices, no bias)
h = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

# ========== Position Encoding ==========
# GPT: Learnable absolute encoding (added to Embedding)
h_0 = tok_emb + pos_emb  # pos_emb is learnable parameter

# LLaMA: RoPE (rotated on Q, K in Attention)
q, k = apply_rotary_emb(q, k, freqs_cis)  # Rotate, no addition
```