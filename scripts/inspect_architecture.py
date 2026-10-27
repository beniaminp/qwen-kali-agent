#!/usr/bin/env python3
"""Read a Hugging Face config.json and print architecture shapes and budgets.

No torch, no transformers, no model download: only the JSON config. Useful for
checking a checkpoint before downloading tens of GB, and for comparing Qwen and
Gemma configurations on the same terms.

Examples:
  python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B
  python scripts/inspect_architecture.py --config config.json --seq 8192
  python scripts/inspect_architecture.py --preset gemma4-12b --json
  python scripts/inspect_architecture.py --hf google/gemma-4-E4B-it --seq 32768
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


PRESETS = {
    "qwen3-14b": {
        "hidden_size": 5120, "num_hidden_layers": 40, "intermediate_size": 17408,
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": False,
    },
    "qwen25-coder-14b": {
        "hidden_size": 5120, "num_hidden_layers": 48, "intermediate_size": 13824,
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 152064, "tie_word_embeddings": False,
    },
    "gemma4-12b": {
        "hidden_size": 3840, "num_hidden_layers": 48, "intermediate_size": 15360,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
        "vocab_size": 262144, "tie_word_embeddings": True,
        "sliding_window": 1024, "global_head_dim": 512,
        "num_global_key_value_heads": 1, "attention_k_eq_v": True,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 8,
    },
    # Edge models: per-layer embeddings (PLE), KV sharing in the last layers,
    # no K=V. Values from google/gemma-4-E4B-it and -E2B-it config.json.
    "gemma4-e4b": {
        "hidden_size": 2560, "num_hidden_layers": 42, "intermediate_size": 10240,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
        "vocab_size": 262144, "tie_word_embeddings": True,
        "sliding_window": 512, "global_head_dim": 512,
        "attention_k_eq_v": False, "num_kv_shared_layers": 18,
        "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7,
    },
    "gemma4-e2b": {
        "hidden_size": 1536, "num_hidden_layers": 35, "intermediate_size": 6144,
        "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256,
        "vocab_size": 262144, "tie_word_embeddings": True,
        "sliding_window": 512, "global_head_dim": 512,
        "attention_k_eq_v": False, "num_kv_shared_layers": 20,
        "use_double_wide_mlp": True,
        "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
        "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
    },
    "qwen3-30b-a3b": {
        "hidden_size": 2048, "num_hidden_layers": 48, "intermediate_size": 6144,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": False,
        "num_experts": 128, "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    },
}


def load_config(args) -> dict:
    if args.preset:
        return dict(PRESETS[args.preset])
    if args.hf:
        url = f"https://huggingface.co/{args.hf}/raw/main/config.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    else:
        data = json.loads(Path(args.config).read_text())
    if "text_config" in data:
        data = data["text_config"]
    return data


def num(value, default=0):
    return int(value if value is not None else default)


def analyze(c: dict) -> dict:
    d = num(c.get("hidden_size"))
    layers = num(c.get("num_hidden_layers"))
    ffn = num(c.get("intermediate_size"))
    heads = num(c.get("num_attention_heads"))
    kv_heads = num(c.get("num_key_value_heads"), heads)
    head_dim = num(c.get("head_dim"), d // heads if heads else 0)
    vocab = num(c.get("vocab_size"))
    tied = bool(c.get("tie_word_embeddings", False))
    experts = num(c.get("num_experts"))
    active_experts = num(c.get("num_experts_per_tok"))
    moe_ffn = num(c.get("moe_intermediate_size"), ffn)

    if not all([d, layers, ffn, heads, kv_heads, head_dim, vocab]):
        raise SystemExit("config is missing one of hidden_size/layers/intermediate_size/heads/vocab")

    layer_types = c.get("layer_types")
    if layer_types:
        layer_types = list(layer_types)[:layers]
        is_local = [("sliding" in t or "local" in t) for t in layer_types]
    else:
        is_local = [False] * layers
    local_layers = sum(is_local)
    global_layers = layers - local_layers
    hybrid = local_layers > 0 and global_layers > 0

    local_kv = kv_heads
    local_dim = head_dim
    global_kv = num(c.get("num_global_key_value_heads"), kv_heads)
    global_dim = num(c.get("global_head_dim"), head_dim)
    k_eq_v = bool(c.get("attention_k_eq_v", False))

    # Gemma 4 edge models: the last `num_kv_shared_layers` layers reuse the
    # K/V of the last earlier layer of the same type, so they have no k/v
    # projections and add nothing to the cache. With `use_double_wide_mlp`
    # those shared layers also get a 2x wider MLP.
    kv_shared = num(c.get("num_kv_shared_layers"))
    first_shared = layers - kv_shared if kv_shared else layers
    double_wide = bool(c.get("use_double_wide_mlp", False)) and kv_shared > 0
    ple = num(c.get("hidden_size_per_layer_input"))
    ple_vocab = num(c.get("vocab_size_per_layer_input"), vocab)

    def attn_shape(q_heads, q_dim, kv_h, k_dim, shared=False, no_v=False):
        q = d * q_heads * q_dim
        o = q_heads * q_dim * d
        if shared:
            return q + o
        k = d * kv_h * k_dim
        v = 0 if no_v else d * kv_h * k_dim
        return q + k + v + o

    attn_local = attn_shape(heads, local_dim, local_kv, local_dim)
    # K=V: global layers reuse the key projection as the value; there is no v_proj.
    attn_global = (attn_shape(heads, global_dim, global_kv, global_dim, no_v=k_eq_v)
                   if global_layers else attn_local)
    mlp = 3 * d * ffn
    attn_norm = 2 * d
    mlp_norm = 2 * d
    ple_per_layer = (2 * d * ple + d) if ple else 0  # input gate, projection, norm

    router = d * experts if experts else 0
    expert_mlp = 3 * d * moe_ffn if experts else 0
    layer_params = 0
    attn_total = 0
    mlp_total = 0
    active = 0 if experts else None
    for i in range(layers):
        local = is_local[i]
        shared = i >= first_shared
        if local:
            attn_i = attn_shape(heads, local_dim, local_kv, local_dim, shared=shared)
        else:
            attn_i = attn_shape(heads, global_dim, global_kv, global_dim, shared=shared, no_v=k_eq_v)
        if experts:
            mlp_i = router + expert_mlp * experts
            active += attn_i + attn_norm + router + expert_mlp * active_experts
        else:
            mlp_i = mlp * (2 if double_wide and shared else 1)
        attn_total += attn_i
        mlp_total += mlp_i
        layer_params += attn_i + attn_norm + mlp_i + (0 if experts else mlp_norm) + ple_per_layer

    ffn_block = router + expert_mlp * experts if experts else mlp + mlp_norm
    per_layer_local = attn_local + attn_norm + ffn_block
    per_layer_global = attn_global + attn_norm + ffn_block

    embed = vocab * d
    lm_head = 0 if tied else vocab * d
    final_norm = d
    # PLE: a [vocab, layers * ple] lookup table plus a [d -> layers * ple] projection.
    ple_table = ple_vocab * layers * ple if ple else 0
    ple_projection = (layers * ple * d + ple) if ple else 0
    non_embedding = layer_params + ple_projection
    total = non_embedding + embed + lm_head + final_norm + ple_table
    # Lookups are free; compute is everything except the embedding tables
    # (the tied embedding still costs a V x d matmul as the LM head).
    compute = total - ple_table
    active_total = active + embed + lm_head + final_norm if active is not None else compute

    return {
        "hidden_size": d,
        "layers": layers,
        "local_layers": local_layers,
        "global_layers": global_layers,
        "ffn": ffn,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "global_kv_heads": global_kv,
        "global_head_dim": global_dim,
        "vocab": vocab,
        "tied": tied,
        "experts": experts,
        "active_experts": active_experts,
        "moe_ffn": moe_ffn,
        "attention_local": attn_local,
        "attention_global": attn_global,
        "attention": attn_total / layers if hybrid or kv_shared else attn_local,
        "attention_total": attn_total,
        "mlp": mlp,
        "mlp_total": mlp_total,
        "per_layer_local": per_layer_local,
        "per_layer_global": per_layer_global,
        "per_layer": layer_params / layers,
        "embedding": embed,
        "lm_head": lm_head,
        "ple_dim": ple,
        "ple_table": ple_table,
        "ple_projection": ple_projection,
        "kv_shared_layers": kv_shared,
        "double_wide_mlp": double_wide,
        "k_eq_v": k_eq_v,
        "hybrid": hybrid,
        "total": total,
        "compute": compute,
        "active": active_total,
        "non_embedding": non_embedding,
        "gqa_ratio": heads / kv_heads if kv_heads else None,
        "global_gqa_ratio": heads / global_kv if global_kv else None,
        "layer_types": layer_types,
        "is_local": is_local,
        "first_kv_shared_layer": first_shared,
    }


def kv_bytes(a: dict, seq: int, bytes_per_value: int = 2, sliding_window: int = 0) -> int:
    """KV cache: local layers hold at most `sliding_window` tokens, K=V global
    layers store one tensor, and KV-shared layers store nothing."""
    window = sliding_window if a["hybrid"] and sliding_window else seq
    total = 0
    for i, local in enumerate(a["is_local"]):
        if i >= a["first_kv_shared_layer"]:
            continue
        if local:
            total += 2 * a["kv_heads"] * a["head_dim"] * min(seq, window)
        else:
            tensors = 1 if a["k_eq_v"] else 2
            total += tensors * a["global_kv_heads"] * a["global_head_dim"] * seq
    return int(total * bytes_per_value)

def fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--hf", help="Model id, e.g. Qwen/Qwen3-14B")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--bytes-per-value", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not any([args.config, args.hf, args.preset]):
        ap.error("provide --config, --hf or --preset")

    raw = load_config(args)
    a = analyze(raw)
    cache = kv_bytes(
        a,
        args.seq,
        args.bytes_per_value,
        sliding_window=num(raw.get("sliding_window")),
    )
    train_flops = 6 * a["compute"] * args.seq * args.batch
    infer_flops = 2 * a["active"] * args.seq * args.batch

    result = {
        "config": raw,
        "derived": a,
        "seq": args.seq,
        "batch": args.batch,
        "kv_cache_bytes": cache,
        "train_flops_batch": train_flops,
        "infer_flops_batch": infer_flops,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"hidden size          : {a['hidden_size']:,}")
    print(f"layers               : {a['layers']}")
    print(f"FFN / MoE FFN        : {a['ffn']:,} / {a['moe_ffn']:,}")
    print(f"query / KV heads     : {a['heads']} / {a['kv_heads']}  (local GQA {a['gqa_ratio']:.2f}:1)")
    print(f"head dim             : {a['head_dim']}")
    if a["hybrid"]:
        print(f"global KV / head dim : {a['global_kv_heads']} / {a['global_head_dim']}  (GQA {a['global_gqa_ratio']:.2f}:1)"
              + ("  K=V" if a["k_eq_v"] else ""))
    if a["kv_shared_layers"]:
        print(f"KV-shared layers     : last {a['kv_shared_layers']} of {a['layers']} (from layer {a['first_kv_shared_layer']})"
              + ("  double-wide MLP" if a["double_wide_mlp"] else ""))
    print(f"vocab / tied         : {a['vocab']:,} / {a['tied']}")
    if a["experts"]:
        print(f"experts total/active : {a['experts']} / {a['active_experts']}")
    print()
    if a["hybrid"]:
        print(f"attention / local    : {a['attention_local']:,} ({fmt(a['attention_local'])})")
        print(f"attention / global   : {a['attention_global']:,} ({fmt(a['attention_global'])})")
        print(f"local / global layers: {a['local_layers']} / {a['global_layers']}")
    else:
        print(f"attention / layer    : {a['attention']:,} ({fmt(a['attention'])})")
    print(f"MLP / layer          : {a['mlp']:,} ({fmt(a['mlp'])})"
          + (f"  x2 in the {a['kv_shared_layers']} KV-shared layers" if a["double_wide_mlp"] else ""))
    print(f"total / layer (avg)  : {a['per_layer']:,.0f} ({fmt(a['per_layer'])})")
    print(f"embedding            : {a['embedding']:,} ({fmt(a['embedding'])})")
    print(f"LM head              : {a['lm_head']:,} ({fmt(a['lm_head'])})")
    if a["ple_dim"]:
        print(f"per-layer emb. table : {a['ple_table']:,} ({fmt(a['ple_table'])})  [vocab x layers x {a['ple_dim']}]")
    print(f"total params         : {a['total']:,} ({fmt(a['total'])})")
    if a["ple_dim"]:
        effective = a["total"] - a["embedding"] - a["ple_table"]
        print(f"effective (no tables): {effective:,} ({fmt(effective)})  text decoder; the model card also counts vision/audio")
    else:
        print(f"non-embedding params : {a['non_embedding']:,} ({fmt(a['non_embedding'])})")
    if a["experts"]:
        print(f"active params/token  : {a['active']:,} ({fmt(a['active'])})")
    print()
    print(f"KV cache @ {args.seq:,} tokens (bf16, K+V): {cache / 1e9:.2f} GB")
    print(f"training FLOPs for B={args.batch}, T={args.seq:,}: {train_flops:.3e}")
    print(f"inference FLOPs for B={args.batch}, T={args.seq:,}: {infer_flops:.3e}")
    print()
    if a["hybrid"]:
        print("Note: KV estimate models local sliding windows, K=V global layers and")
        print("KV-shared layers (which store nothing).")
    else:
        print("Note: KV estimate is a full-attention upper bound; it ignores any")
        print("sliding windows or K=V sharing.")
    print("Attention score FLOPs are not included in the 6ND estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
