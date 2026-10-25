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


PRESETS = {
    "qwen3-14b": Config("Qwen3-14B", 151936, 5120, 40, 17408, 40, 8, 128, False),
    "qwen25-coder-14b": Config("Qwen2.5-Coder-14B", 152064, 5120, 48, 13824, 40, 8, 128, False),
    "gemma4-12b": Config("Gemma 4 12B", 262144, 3840, 48, 15360, 16, 8, 256, True),
    "tiny-100m": Config("tiny-100M", 32000, 768, 12, 2048, 12, 4, 64, True),
    "small-1b": Config("small-1B", 32000, 2048, 24, 5632, 16, 4, 128, True),
}


def params(c: Config) -> dict:
    q_dim = c.heads * c.head_dim
    kv_dim = c.kv_heads * c.head_dim
    attn = c.d_model * q_dim + 2 * c.d_model * kv_dim + q_dim * c.d_model
    mlp = 3 * c.d_model * c.ffn
    per_layer = attn + mlp
    embed = c.vocab * c.d_model
    lm_head = 0 if c.tie_embeddings else c.vocab * c.d_model
    total = c.layers * per_layer + embed + lm_head + c.d_model
    return {
        "attention_per_layer": attn,
        "mlp_per_layer": mlp,
        "per_layer": per_layer,
        "embedding": embed,
        "lm_head": lm_head,
        "total": total,
        "non_embedding": c.layers * per_layer,
    }


def kv_cache_bytes(c: Config, seq: int, bytes_per_value: float = 2.0, kv_shared_global: bool = False) -> int:
    """Rough KV cache: 2 (K,V) * layers * kv_heads * head_dim * seq * bytes.

    `kv_shared_global` halves the global-layer V cache (K=V), a Gemma 4 trick.
    """
    per_layer = 2 * c.kv_heads * c.head_dim * seq * bytes_per_value
    total = c.layers * per_layer
    if kv_shared_global:
        total *= 0.75  # illustrative; real saving depends on local/global ratio
    return int(total)


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
    ap.add_argument("--peak-tflops", type=float, default=989.0,
                    help="Per-GPU peak dense BF16 TFLOP/s (default H100 SXM ~989)")
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
    infer_flops = 2 * p["total"]
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
        "kv_cache_gb_seq_kv_shared": kv_cache_bytes(c, args.seq, kv_shared_global=True) / 1e9,
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
    print(f"KV cache @ {args.seq:,} tokens : {result['kv_cache_gb_seq']:.1f} GB (bf16, no sharing)")
    print(f"KV cache, K=V global : {result['kv_cache_gb_seq_kv_shared']:.1f} GB (illustrative)")
    print()
    print("Reality check: add 20-50% for failures, checkpointing, data loading and")
    print("evaluation. MFU above ~45% is hard on long-context training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
