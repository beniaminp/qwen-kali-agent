#!/usr/bin/env python3
"""Back-of-the-envelope parameter, FLOP and cost estimates for pretraining.

This is a planning tool, not a benchmark. It uses:
  N  ≈ vocab*d (tied) or 2*vocab*d (untied) + layers*(attention + MLP)
  C  ≈ 6 * N * D                (training FLOPs, forward + backward)
  P  ≈ 2 * N                    (inference FLOPs per generated token)

Real throughput depends on MFU, sequence length, attention kernels, network,
checkpointing and failures. Use measured tokens/sec from a short run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    name: str
    vocab: int
    d_model: int
    layers: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    tie_embeddings: bool
    global_kv_heads: int = 0
    global_head_dim: int = 0
    global_layers: int = 0
    sliding_window: int = 0
    attention_k_eq_v: bool = False


PRESETS = {
    "qwen3-14b": Config("Qwen3-14B", 151936, 5120, 40, 17408, 40, 8, 128, False),
    "qwen25-coder-14b": Config("Qwen2.5-Coder-14B", 152064, 5120, 48, 13824, 40, 8, 128, False),
    "gemma4-12b": Config("Gemma 4 12B", 262144, 3840, 48, 15360, 16, 8, 256, True,
                         global_kv_heads=1, global_head_dim=512, global_layers=8,
                         sliding_window=1024, attention_k_eq_v=True),
    "tiny-100m": Config("tiny-100M", 32000, 768, 12, 2048, 12, 4, 64, True),
    "small-1b": Config("small-1B", 32000, 2048, 24, 5632, 16, 4, 128, True),
}


def _attn(d: int, heads: int, kv_heads: int, head_dim: int, k_eq_v: bool = False) -> int:
    q = d * heads * head_dim
    # With K=V the key projection doubles as the value; there is no v_proj.
    kv = (1 if k_eq_v else 2) * d * kv_heads * head_dim
    o = heads * head_dim * d
    return q + kv + o


def params(c: Config) -> dict:
    local_layers = c.layers - c.global_layers
    attn_local = _attn(c.d_model, c.heads, c.kv_heads, c.head_dim)
    attn_global = _attn(c.d_model, c.heads, c.global_kv_heads or c.kv_heads,
                        c.global_head_dim or c.head_dim,
                        k_eq_v=c.attention_k_eq_v) if c.global_layers else attn_local
    mlp = 3 * c.d_model * c.ffn
    norms = 2 * c.d_model
    per_local = attn_local + mlp + norms
    per_global = attn_global + mlp + norms
    layer_params = local_layers * per_local + c.global_layers * per_global
    embed = c.vocab * c.d_model
    lm_head = 0 if c.tie_embeddings else c.vocab * c.d_model
    total = layer_params + embed + lm_head + c.d_model
    attn_total = local_layers * attn_local + c.global_layers * attn_global
    return {
        "attention_per_layer": attn_total / c.layers,
        "attention_local": attn_local,
        "attention_global": attn_global,
        "attention_total": attn_total,
        "mlp_per_layer": mlp,
        "per_layer": layer_params / c.layers,
        "per_layer_local": per_local,
        "per_layer_global": per_global,
        "embedding": embed,
        "lm_head": lm_head,
        "total": total,
        "active": total,
        "non_embedding": layer_params,
    }


def kv_cache_bytes(c: Config, seq: int, bytes_per_value: float = 2.0) -> int:
    """KV cache, modelling local windows and global K=V when configured."""
    if c.global_layers:
        local_layers = c.layers - c.global_layers
        local = local_layers * 2 * c.kv_heads * c.head_dim * min(seq, c.sliding_window or seq)
        glob = c.global_layers * (c.global_kv_heads or c.kv_heads) * (c.global_head_dim or c.head_dim) * seq
        if not c.attention_k_eq_v:
            glob *= 2
        return int((local + glob) * bytes_per_value)
    return int(2 * c.layers * c.kv_heads * c.head_dim * seq * bytes_per_value)


def fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="qwen3-14b")
    ap.add_argument("--name")
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--ffn", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--tokens", type=float, default=36e12,
                    help="Training tokens (default 36T, Qwen3 scale)")
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--gpus", type=int, default=1024)
    ap.add_argument("--peak-tflops", type=float, default=495.0,
                    help="Per-GPU peak dense BF16 TFLOP/s (default H100 SXM ~495; 989 is sparse)")
    ap.add_argument("--mfu", type=float, default=0.40, help="Model FLOPs utilization 0-1")
    ap.add_argument("--cost-per-gpu-hour", type=float, default=2.5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    c = PRESETS[args.preset]
    overrides = {
        "name": args.name, "vocab": args.vocab, "d_model": args.d_model,
        "layers": args.layers, "ffn": args.ffn, "heads": args.heads,
        "kv_heads": args.kv_heads, "head_dim": args.head_dim,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(c, key, value)
    if args.untied:
        c.tie_embeddings = False

    p = params(c)
    train_flops = 6 * p["total"] * args.tokens
    infer_flops = 2 * p["active"]
    effective = args.peak_tflops * 1e12 * args.mfu
    gpu_seconds = train_flops / effective
    gpu_hours = gpu_seconds / 3600
    wall_days = gpu_hours / args.gpus / 24
    cost = gpu_hours * args.cost_per_gpu_hour
    tokens_per_param = args.tokens / p["total"]

    result = {
        "config": asdict(c),
        "params": {k: int(v) for k, v in p.items()},
        "tokens": args.tokens,
        "tokens_per_param": tokens_per_param,
        "train_flops": train_flops,
        "inference_flops_per_token": infer_flops,
        "gpus": args.gpus,
        "peak_tflops_per_gpu": args.peak_tflops,
        "mfu": args.mfu,
        "gpu_hours": gpu_hours,
        "wall_days": wall_days,
        "cost_usd": cost,
        "kv_cache_gb_seq": kv_cache_bytes(c, args.seq) / 1e9,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"model                : {c.name}")
    print(f"vocab / d_model      : {c.vocab:,} / {c.d_model:,}")
    print(f"layers / ffn         : {c.layers} / {c.ffn:,}")
    print(f"heads / kv_heads     : {c.heads} / {c.kv_heads}")
    print(f"head_dim             : {c.head_dim}")
    print(f"embeddings tied      : {c.tie_embeddings}")
    print()
    print(f"total params         : {p['total']:,} ({fmt(p['total'])})")
    print(f"non-embedding params : {p['non_embedding']:,} ({fmt(p['non_embedding'])})")
    if c.global_layers:
        print(f"attention / local    : {p['attention_local']:,}")
        print(f"attention / global   : {p['attention_global']:,}")
        print(f"local / global layers: {c.layers - c.global_layers} / {c.global_layers}")
    else:
        print(f"attention / layer    : {p['attention_per_layer']:,}")
    print(f"MLP / layer          : {p['mlp_per_layer']:,}")
    print()
    print(f"training tokens      : {args.tokens:,.0f} ({fmt(args.tokens)})")
    print(f"tokens / parameter   : {tokens_per_param:.1f}")
    print(f"training FLOPs       : {train_flops:.3e} ({fmt(train_flops)})")
    print(f"inference FLOPs/token: {infer_flops:.3e} ({fmt(infer_flops)})")
    print()
    print(f"GPUs                 : {args.gpus}")
    print(f"peak TFLOP/s/GPU     : {args.peak_tflops}")
    print(f"MFU                  : {args.mfu:.0%}")
    print(f"GPU-hours            : {gpu_hours:,.0f}")
    print(f"wall time            : {wall_days:.1f} days")
    print(f"compute cost         : ${cost:,.0f}")
    print()
    print(f"KV cache @ {args.seq:,} tokens : {result['kv_cache_gb_seq']:.2f} GB (bf16)")
    print()
    print("Reality check: add 20-50% for failures, checkpointing, data loading and")
    print("evaluation. MFU above ~45% is hard on long-context training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
