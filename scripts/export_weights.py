"""
Download GPT-2 (small) from HuggingFace and export weights to a binary file.

Binary format per tensor:
  [4 bytes] name length
  [N bytes] name (UTF-8)
  [4 bytes] number of float32 elements
  [N*4 bytes] float32 data (little-endian)
"""

import struct
import numpy as np
from transformers import GPT2Model

OUTPUT = "../weights.bin"

print("Downloading GPT-2 weights from HuggingFace...")
model = GPT2Model.from_pretrained("gpt2")
state = model.state_dict()

print(f"Writing weights to {OUTPUT}...")
with open(OUTPUT, "wb") as f:
    def write_tensor(name, data):
        data = data.float().numpy().flatten()
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", len(data)))
        f.write(data.astype(np.float32).tobytes())
        print(f"  {name}: {list(data.shape) if hasattr(data, 'shape') else len(data)} elements")

    # Token + positional embeddings
    write_tensor("wte", state["wte.weight"])
    write_tensor("wpe", state["wpe.weight"])

    # 12 transformer blocks
    for i in range(12):
        p = f"h.{i}"

        # LayerNorm 1
        write_tensor(f"{p}.ln1.w", state[f"{p}.ln_1.weight"])
        write_tensor(f"{p}.ln1.b", state[f"{p}.ln_1.bias"])

        # Attention — c_attn combines Q, K, V into [768, 2304], split into 3x [768, 768]
        c_attn_w = state[f"{p}.attn.c_attn.weight"]   # [768, 2304]
        c_attn_b = state[f"{p}.attn.c_attn.bias"]     # [2304]
        write_tensor(f"{p}.attn.wq", c_attn_w[:, :768])
        write_tensor(f"{p}.attn.wk", c_attn_w[:, 768:1536])
        write_tensor(f"{p}.attn.wv", c_attn_w[:, 1536:])
        write_tensor(f"{p}.attn.bq", c_attn_b[:768])
        write_tensor(f"{p}.attn.bk", c_attn_b[768:1536])
        write_tensor(f"{p}.attn.bv", c_attn_b[1536:])

        # Attention output projection
        write_tensor(f"{p}.attn.wo", state[f"{p}.attn.c_proj.weight"])
        write_tensor(f"{p}.attn.bo", state[f"{p}.attn.c_proj.bias"])

        # LayerNorm 2
        write_tensor(f"{p}.ln2.w", state[f"{p}.ln_2.weight"])
        write_tensor(f"{p}.ln2.b", state[f"{p}.ln_2.bias"])

        # FFN
        write_tensor(f"{p}.ffn.w1", state[f"{p}.mlp.c_fc.weight"])
        write_tensor(f"{p}.ffn.b1", state[f"{p}.mlp.c_fc.bias"])
        write_tensor(f"{p}.ffn.w2", state[f"{p}.mlp.c_proj.weight"])
        write_tensor(f"{p}.ffn.b2", state[f"{p}.mlp.c_proj.bias"])

    # Final LayerNorm
    write_tensor("ln_f.w", state["ln_f.weight"])
    write_tensor("ln_f.b", state["ln_f.bias"])

print(f"\nDone! weights.bin written.")
