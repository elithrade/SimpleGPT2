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

### Attention — `Attention.Forward(x)` *(in progress)*
The core of the transformer. Each token "looks at" other tokens to gather context.

```
Q = x · Wq + bq        # What am I looking for?
K = x · Wk + bk        # What do I contain?
V = x · Wv + bv        # What will I share?

scores = Q · Kᵀ / √dHead    # How relevant is each token?
scores = causal_mask(scores) # Can't look at future tokens
weights = Softmax(scores)
output  = weights · V        # Weighted sum of values
```

GPT-2 runs this **12 times in parallel** (multi-head), each head learning different relationships.

---

## Progress

| Stage | Component | Status |
|---|---|---|
| 1 | Tensor | ✅ Done |
| 2 | MathOps (MatMul, Softmax, LayerNorm, GELU) | ✅ Done |
| 3 | Attention | 🔄 In progress |
| 4 | TransformerBlock | ⏳ Pending |
| 5 | WeightLoader | ⏳ Pending |
| 6 | Tokenizer | ⏳ Pending |
| 7 | Generation loop | ⏳ Pending |
