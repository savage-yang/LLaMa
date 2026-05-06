# LLaMA 模型架构详解 

逐模块拆解 LLaMA 相比 GPT 的四大架构改进（RMSNorm / RoPE / SwiGLU / GQA），从数学公式到代码骨架，全面理解每一处改进的设计动机。
---

## 📚 目录
- [LLaMA 架构全景图](#llama-架构全景图)
- [核心架构改进对比](#核心架构改进对比)
- [完整数据流追踪](#完整数据流追踪)
- [代码级差异预览](#代码级差异预览)

---

## LLaMA 架构全景图
LLaMA 与 GPT 一样是 Decoder-only 架构，但每个组件都做了精心改进：

```
输入 token IDs: [x_1, x_2, ..., x_T]
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  E_tok ∈ R^{V × d}
│  （无位置编码加法）│  ← 注意：RoPE 不在这里加！
└──────────────────┘
       │
       ▼  h_0 = E_tok[x]
┌──────────────────┐
│  LLaMA Block × N  │  ← 重复 N 次（LLaMA-7B: N=32）
│  ┌──────────────┐ │
│  │  RMSNorm 1   │ │  ← 替代 LayerNorm
│  │  GQA + RoPE  │ │  ← 替代 MHA + 可学习位置编码
│  │  + Residual    │ │
│  ├──────────────┤ │
│  │  RMSNorm 2   │ │
│  │  SwiGLU FFN  │ │  ← 替代 GELU FFN
│  │  + Residual    │ │
│  └──────────────┘ │
└──────────────────┘
       │
       ▼  h_N
┌──────────────────┐
│   RMSNorm         │  ← 最终归一化
│   LM Head (线性层)│  W ∈ R^{d × V}，不与 E_tok 共享
└──────────────────┘
       │
       ▼
  logits ∈ R^{T × V}  → softmax → P(x_{t+1} | x_{≤t})
```

**关键参数对照表**：

| 超参数 | 符号 | LLaMA-7B | LLaMA-13B | LLaMA-33B | LLaMA-65B |
|--------|------|----------|-----------|-----------|-----------|
| 层数 | $N$ | 32 | 40 | 60 | 80 |
| 隐藏维度 | $d$ | 4096 | 5120 | 6656 | 8192 |
| 注意力头数 | $n_h$ | 32 | 40 | 52 | 64 |
| 每头维度 | $d_k = d/n_h$ | 128 | 128 | 128 | 128 |
| FFN 中间维度 | $d_{ff}$ | 11008 | 13824 | 17920 | 22016 |
| 上下文长度 | $T_{max}$ | 2048 | 2048 | 2048 | 2048 |
| 词表大小 | $V$ | 32000 | 32000 | 32000 | 32000 |
| GQA KV 头数 | $n_{kv}$ | 32 (MHA) | 40 (MHA) | 52 (MHA) | 64 (MHA) |
| 总参数量 | — | 6.7B | 13.0B | 32.5B | 65.2B |

注：LLaMA-1 全部使用 MHA（每个头有独立的 KV），LLaMA-2 的 70B 引入了 GQA（8 个 KV 头）。

---

## 核心架构改进对比
LLaMA-1 基于 Transformer Decoder-only 架构，但相比 GPT 做了四处关键改进：

| 改进项 | GPT-2/3 实现 | LLaMA 实现 | 参考来源 |
|------|---------|-------|------|
| **归一化** | LayerNorm (Post-Norm) | **RMSNorm (Pre-Norm)** | GPT-3 用 Pre-LN，LLaMA 进一步简化为 RMSNorm |
| **激活函数** | GELU | **SwiGLU** | Shazeer (2020)，PaLM 也用了 |
| **位置编码** | 可学习绝对位置编码 | **RoPE（旋转位置编码）** | Su et al. (2021) |
| **注意力** | MHA（全部头独立 KV） | **MHA**（LLaMA-1），**GQA**（LLaMA-2） | GQA: Ainslie et al. (2023) |
| **Bias** | 所有线性层有 bias | **去掉所有 bias** | 减少参数，实践中无性能损失 |
| **权重共享** | Embedding = LM Head | **不共享** | LLaMA 选择不共享 |

### LLaMA vs GPT 技术差异汇总
| 对比维度 | GPT-2/3 | LLaMA-1 | 改进动机 |
|------|---------|---------|---------|
| **归一化** | LayerNorm | RMSNorm | 去掉均值中心化，计算更快，效果相当 |
| **Norm 位置** | Pre-Norm (GPT-2+) | Pre-Norm | 相同 |
| **位置编码** | 可学习绝对编码 | RoPE | 支持长度外推，编码相对位置信息 |
| **激活函数** | GELU | SwiGLU | 门控机制 + Swish，实验效果更好 |
| **FFN 结构** | 2 个线性层 | 3 个线性层（门控） | SwiGLU 需要额外一个线性层 |
| **$d_{ff}$** | $4d$ | $\frac{8}{3}d$（取整） | 保持与 2 层 FFN 相同的参数量 |
| **注意力** | MHA | MHA (v1) / GQA (v2) | GQA 减少 KV Cache |
| **Bias** | 有 | 无 | 减少参数，无性能损失 |
| **权重共享** | Emb = LM Head | 不共享 | 大模型中不共享效果更好 |
| **Tokenizer** | GPT-2 BPE (50257) | SentencePiece BPE (32000) | 更小词表 + 字节回退 |
| **上下文长度** | 1024 (GPT-2) / 2048 (GPT-3) | 2048 (v1) / 4096 (v2) | 更长上下文 |
| **训练数据** | 私有数据 | 公开数据 | 可复现 |

---

## 完整数据流追踪
以 LLaMA-7B 为例，输入 `"The cat sat"` 的完整数据流：

```
输入 token IDs: [450, 6635, 3290]      # SentencePiece tokenize
                 shape: (1, 3)

  ┌─── Token Embedding ─────────────────────────┐
  │ E_tok[450], E_tok[6635], E_tok[3290]        │
  │ shape: (1, 3, 4096)                          │
  │ 注意：没有加位置编码！RoPE 在 Attention 中处理  │
  └─────────────────────────────────────────────┘

  h_0: shape (1, 3, 4096)

  ─── Block 1 ──────────────────────────────────────
  │ RMSNorm_1:    (1, 3, 4096) → (1, 3, 4096)
  │ Q projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 heads
  │ K projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 KV heads (MHA)
  │ V projection: (1, 3, 4096) → (1, 3, 32, 128)
  │
  │ Apply RoPE to Q, K:
  │   Q: (1, 3, 32, 128) → 旋转 → (1, 3, 32, 128)
  │   K: (1, 3, 32, 128) → 旋转 → (1, 3, 32, 128)
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
  │   逐元素乘:      (1, 3, 11008)
  │   W_down:        (1, 3, 11008) → (1, 3, 4096)
  │ + Residual → (1, 3, 4096)
  ──────────────────────────────────────────────────

  ... (重复 32 次) ...

  h_32: shape (1, 3, 4096)

  Final RMSNorm: (1, 3, 4096) → (1, 3, 4096)
  LM Head:       (1, 3, 4096) × (4096, 32000) → (1, 3, 32000)

  logits[:, -1, :] → softmax → P(next_token | "The cat sat")
```

---

## 代码级差异预览
```python
# ========== 归一化 ==========
# GPT: LayerNorm
x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + eps).sqrt()
x = gamma * x + beta  # 有 beta（偏移）

# LLaMA: RMSNorm（更简单）
x = x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
x = gamma * x          # 没有 beta

# ========== FFN ==========
# GPT: standard FFN
h = GELU(x @ W1 + b1) @ W2 + b2

# LLaMA: SwiGLU FFN（3 个权重矩阵，无 bias）
h = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

# ========== 位置编码 ==========
# GPT: 可学习绝对编码（加在 Embedding 上）
h_0 = tok_emb + pos_emb  # pos_emb 是可学习参数

# LLaMA: RoPE（在 Attention 的 Q, K 上做旋转）
q, k = apply_rotary_emb(q, k, freqs_cis)  # 旋转，不加
```