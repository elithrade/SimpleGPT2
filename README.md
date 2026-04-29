# SimpleGPT2 — Learning Inference From Scratch

A minimal GPT-2 inference engine built in C# to understand the principles of LLM inference.

---

## What is Inference?

Inference is running a trained model to produce an output (as opposed to training, which adjusts weights).
Every LLM inference follows this loop:

```
1. Tokenize prompt       "Hello"  →  [15496]
2. Embed tokens          [15496]  →  float vectors
3. Forward pass          vectors  →  logits (score for every word in vocab)
4. Sample next token     logits   →  "world"
5. Append & repeat       until done
```

---

## Architecture Overview

```
Input Text
    ↓
Tokenizer          — text → token IDs
    ↓
Token Embedding    — token ID → float vector [dModel]
    +
Positional Embed   — encodes position in sequence
    ↓
TransformerBlock × N
  ├─ LayerNorm
  ├─ Multi-Head Attention
  ├─ LayerNorm
  └─ Feed-Forward Network (FFN)
    ↓
Final LayerNorm
    ↓
Linear projection  — [dModel] → [vocab_size] logits
    ↓
Softmax + Sample   — pick next token
```

GPT-2 (small) config: `N=12` blocks, `dModel=768`, `nHeads=12`, `vocab=50257`

---

## Math Reference

### Tensor
A multi-dimensional float array in **row-major** order.
Element `[i, j]` of a matrix with `C` columns lives at `Data[i * C + j]`.

---

### MatMul — `MathOps.MatMul(A, B)`
Matrix multiplication. `A: [M, K]`, `B: [K, N]` → `C: [M, N]`

```
C[i,j] = Σₖ A[i,k] * B[k,j]
```

The fundamental building block — used everywhere (QKV projections, FFN, output layer).

---

### Softmax — `MathOps.Softmax(x)`
Converts raw scores (logits) into probabilities that sum to 1.

```
softmax(xᵢ) = e^(xᵢ - max) / Σⱼ e^(xⱼ - max)
```

Subtracting `max` is a numerical stability trick to prevent float overflow.
Used in attention (token-to-token scores) and final token sampling.

---

### LayerNorm — `MathOps.LayerNorm(x, w, b)`
Normalizes a vector to mean=0, variance=1, then applies learnable scale (`w`) and bias (`b`).

```
x_norm = (x - mean) / √(variance + ε)
output  = x_norm * w + b
```

`ε = 1e-5` prevents division by zero. Applied before attention and FFN in each block.

---

### GELU — `MathOps.Gelu(x)`
Activation function used in the Feed-Forward Network. Smoother than ReLU.

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

---

### Attention — `Attention.Forward(x)`

The core of the transformer. Each token "looks at" other tokens to gather context and update its own meaning.

**Intuition:** Given "The animal didn't cross the street because it was too tired", attention lets "it" figure out it refers to "animal" by comparing their representations.

Each token gets projected into 3 vectors:
- **Q (Query)** — "What am I looking for?"
- **K (Key)** — "What do I contain?"
- **V (Value)** — "What will I share if attended to?"

**Step 1 — Project into Q, K, V:**
```
Q = x · Wq + bq     shape: [T, dModel]
K = x · Wk + bk     shape: [T, dModel]
V = x · Wv + bv     shape: [T, dModel]
```

**Step 2 — Compute scores:**
```
scores = Q · Kᵀ / √dHead    shape: [T, T]
```
`scores[i,j]` = how much token `i` attends to token `j`.
Dividing by `√dHead` prevents large dot products from saturating softmax.

**Step 3 — Causal mask:**
```
scores[i,j] = -inf   where j > i
```
Token `i` cannot see future tokens. `-inf` becomes `0` after softmax.

**Step 4 — Softmax each row:**
```
weights = Softmax(scores)    each row sums to 1
```

**Step 5 — Weighted sum of V:**
```
output = weights · V         shape: [T, dModel]
```
Each token's output is a blend of all value vectors, weighted by attention.

**Multi-head:** GPT-2 runs this **12 times in parallel** with different Wq/Wk/Wv weights. Each head learns different relationships (syntax, coreference, proximity). Results are concatenated and projected back:
```
output = Concat(head_1, ..., head_12) · Wo + bo
```

---

## Progress

| Stage | Component | Status |
|---|---|---|
| 1 | Tensor | ✅ Done |
| 2 | MathOps (MatMul, Softmax, LayerNorm, GELU) | ✅ Done |
| 3 | Attention | ✅ Done |
| 4 | TransformerBlock | 🔄 In progress |
| 5 | WeightLoader | ⏳ Pending |
| 6 | Tokenizer | ⏳ Pending |
| 7 | Generation loop | ⏳ Pending |
