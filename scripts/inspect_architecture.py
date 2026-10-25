#!/usr/bin/env python3
"""Read a Hugging Face config.json and print architecture shapes and budgets.

No torch, no transformers, no model download: only the JSON config. Useful for
checking a checkpoint before downloading tens of GB, and for comparing Qwen and
Gemma configurations on the same terms.

Examples:
  python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B
  python scripts/inspect_architecture.py --config config.json --seq 8192
  python scripts/inspect_architecture.py --preset gemma4-12b --json
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
        local_layers = sum(1 for t in layer_types if "sliding" in t or "local" in t)
        global_layers = layers - local_layers
    else:
        local_layers, global_layers = layers, 0

    local_kv = kv_heads
    local_dim = head_dim
    global_kv = num(c.get("num_global_key_value_heads"), kv_heads)
    global_dim = num(c.get("global_head_dim"), head_dim)

    def attn_shape(q_heads, q_dim, kv_h, k_dim, out_dim):
        q = d * q_heads * q_dim
        k = d * kv_h * k_dim
        v = d * kv_h * k_dim
        o = q_heads * q_dim * d
        return q + k + v + o

    attn_local = attn_shape(heads, local_dim, local_kv, local_dim, local_dim)
    attn_global = attn_shape(heads, global_dim, global_kv, global_dim, global_dim) if global_layers else attn_local
    mlp = 3 * d * ffn
    attn_norm = 2 * d
    mlp_norm = 2 * d

    if experts:
        router = d * experts
        expert_mlp = 3 * d * moe_ffn
        per_layer_local = attn_local + attn_norm + router + expert_mlp * experts
        per_layer_global = attn_global + attn_norm + router + expert_mlp * experts
        active = layers * (attn_local + attn_norm + router + expert_mlp * active_experts)
    else:
        router = 0
        expert_mlp = 0
        per_layer_local = attn_local + attn_norm + mlp + mlp_norm
        per_layer_global = attn_global + attn_norm + mlp + mlp_norm
        active = None

    layer_params = local_layers * per_layer_local + global_layers * per_layer_global
    embed = vocab * d
    lm_head = 0 if tied else vocab * d
    final_norm = d
    total = layer_params + embed + lm_head + final_norm
    active_total = active + embed + lm_head + final_norm if active is not None else total

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
        "attention": attn_local if not global_layers else (local_layers * attn_local + global_layers * attn_global) / layers,
        "attention_total": local_layers * attn_local + global_layers * attn_global,
        "mlp": mlp,
        "per_layer_local": per_layer_local,
        "per_layer_global": per_layer_global,
        "per_layer": layer_params / layers,
        "embedding": embed,
        "lm_head": lm_head,
        "total": total,
        "active": active_total,
        "non_embedding": layer_params,
        "gqa_ratio": heads / kv_heads if kv_heads else None,
        "global_gqa_ratio": heads / global_kv if global_kv else None,
        "layer_types": layer_types,
    }


def kv_bytes(a: dict, seq: int, bytes_per_value: int = 2, sliding_window: int = 0,
             k_eq_v: bool = False) -> int:
    """KV cache, optionally modelling local windows and K=V global layers."""
    if a.get("layer_types") or a["global_layers"]:
        # The caller passes the raw config values through the analyzed dict.
        local = a["local_layers"] * 2 * a["kv_heads"] * a["head_dim"] * min(seq, sliding_window or seq)
        global_term = a["global_layers"] * a["global_kv_heads"] * a["global_head_dim"] * seq
        if not k_eq_v:
            global_term *= 2
        return int((local + global_term) * bytes_per_value)
    return 2 * a["layers"] * a["kv_heads"] * a["head_dim"] * seq * bytes_per_value

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
        k_eq_v=bool(raw.get("attention_k_eq_v", False)),
    )
    train_flops = 6 * a["total"] * args.seq * args.batch
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
    if a["global_layers"]:
        print(f"global KV / head dim : {a['global_kv_heads']} / {a['global_head_dim']}  (GQA {a['global_gqa_ratio']:.2f}:1)")
    print(f"vocab / tied         : {a['vocab']:,} / {a['tied']}")
    if a["experts"]:
        print(f"experts total/active : {a['experts']} / {a['active_experts']}")
    print()
    if a["global_layers"]:
        print(f"attention / local    : {a['attention_local']:,} ({fmt(a['attention_local'])})")
        print(f"attention / global   : {a['attention_global']:,} ({fmt(a['attention_global'])})")
        print(f"local / global layers: {a['local_layers']} / {a['global_layers']}")
    else:
        print(f"attention / layer    : {a['attention']:,} ({fmt(a['attention'])})")
    print(f"MLP / layer          : {a['mlp']:,} ({fmt(a['mlp'])})")
    print(f"total / layer (avg)  : {a['per_layer']:,} ({fmt(a['per_layer'])})")
    print(f"embedding            : {a['embedding']:,} ({fmt(a['embedding'])})")
    print(f"LM head              : {a['lm_head']:,} ({fmt(a['lm_head'])})")
    print(f"total params         : {a['total']:,} ({fmt(a['total'])})")
    print(f"non-embedding params : {a['non_embedding']:,} ({fmt(a['non_embedding'])})")
    if a["experts"]:
        print(f"active params/token  : {a['active']:,} ({fmt(a['active'])})")
    print()
    print(f"KV cache @ {args.seq:,} tokens (bf16, K+V): {cache / 1e9:.2f} GB")
    print(f"training FLOPs for B={args.batch}, T={args.seq:,}: {train_flops:.3e}")
    print(f"inference FLOPs for B={args.batch}, T={args.seq:,}: {infer_flops:.3e}")
    print()
    if a["global_layers"]:
        print("Note: KV estimate models local sliding windows and global K=V sharing.")
    else:
        print("Note: KV estimate is a full-attention upper bound; it ignores any")
        print("sliding windows or K=V sharing.")
    print("Attention score FLOPs are not included in the 6ND estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
