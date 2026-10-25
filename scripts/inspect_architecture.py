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
        "sliding_window": 1024,
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

    q = d * heads * head_dim
    k = d * kv_heads * head_dim
    v = d * kv_heads * head_dim
    o = heads * head_dim * d
    attn = q + k + v + o
    attn_norm = 2 * d
    mlp = 3 * d * ffn
    mlp_norm = 2 * d
    router = d * experts if experts else 0
    expert_mlp = 3 * d * moe_ffn if experts else 0
    per_layer = attn + attn_norm + mlp + mlp_norm
    if experts:
        per_layer = attn + attn_norm + router + expert_mlp * experts
    embed = vocab * d
    lm_head = 0 if tied else vocab * d
    final_norm = d
    total = layers * per_layer + embed + lm_head + final_norm
    active = layers * (attn + attn_norm + router + expert_mlp * active_experts) + embed + lm_head + final_norm if experts else total

    return {
        "hidden_size": d,
        "layers": layers,
        "ffn": ffn,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "tied": tied,
        "experts": experts,
        "active_experts": active_experts,
        "moe_ffn": moe_ffn,
        "attention": attn,
        "mlp": mlp,
        "per_layer": per_layer,
        "embedding": embed,
        "lm_head": lm_head,
        "total": total,
        "active": active,
        "non_embedding": layers * per_layer,
        "gqa_ratio": heads / kv_heads if kv_heads else None,
    }


def kv_bytes(a: dict, seq: int, bytes_per_value: int = 2) -> int:
    # 2 for K and V; conservative (does not model K=V or local windows).
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
    cache = kv_bytes(a, args.seq, args.bytes_per_value)
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
    print(f"query / KV heads     : {a['heads']} / {a['kv_heads']}  (GQA {a['gqa_ratio']:.2f}:1)")
    print(f"head dim             : {a['head_dim']}")
    print(f"vocab / tied         : {a['vocab']:,} / {a['tied']}")
    if a["experts"]:
        print(f"experts total/active : {a['experts']} / {a['active_experts']}")
    print()
    print(f"attention / layer    : {a['attention']:,} ({fmt(a['attention'])})")
    print(f"MLP / layer          : {a['mlp']:,} ({fmt(a['mlp'])})")
    print(f"total / layer        : {a['per_layer']:,} ({fmt(a['per_layer'])})")
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
    print("Note: KV estimate ignores sliding windows and K=V sharing; it is an")
    print("upper bound for a full-attention model. Attention score FLOPs are not")
    print("included in the 6ND estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
