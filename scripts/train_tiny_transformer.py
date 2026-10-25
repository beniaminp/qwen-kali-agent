#!/usr/bin/env python3
"""Train a tiny Llama-style transformer from scratch on a plain text file.

This is a teaching/reference implementation, not a production trainer. It
contains the same mathematical pieces as a modern Qwen/Gemma-style decoder:

  - byte-level tokens (vocab 256), so no tokenizer training is required;
  - pre-norm RMSNorm;
  - RoPE;
  - grouped-query attention (GQA);
  - SwiGLU MLP;
  - causal next-token prediction.

It is deliberately small enough to run on a CPU for a smoke test, and it can
use a single GPU with bf16 autocast.

Example:
  python scripts/train_tiny_transformer.py --data data/tiny/corpus.txt \
    --out runs/tiny --dim 256 --layers 6 --heads 8 --kv-heads 2 \
    --ffn 768 --seq 256 --batch 16 --steps 2000

Requirements: PyTorch 2.x. For a GPU: --device cuda --dtype bf16.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    """Rotary position embedding with a precomputed cos/sin cache."""

    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, seq, head_dim]
        seq = x.shape[-2]
        cos = self.cos[:seq].view(1, 1, seq, -1)
        sin = self.sin[:seq].view(1, 1, seq, -1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int, rope: RoPE):
        super().__init__()
        if dim != heads * head_dim:
            raise ValueError("dim must equal heads * head_dim")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.n_rep = heads // kv_heads
        self.rope = rope
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: repeat each KV head for its query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn, bias=False)
        self.up = nn.Linear(dim, ffn, bias=False)
        self.down = nn.Linear(ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int, ffn: int, rope: RoPE):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, kv_heads, head_dim, rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, kv_heads: int,
                 ffn: int, max_seq: int, tie: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        head_dim = dim // heads
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab, dim)
        self.rope = RoPE(head_dim, max_seq)
        self.blocks = nn.ModuleList([
            Block(dim, heads, kv_heads, head_dim, ffn, self.rope) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def get_batch(data: torch.Tensor, batch: int, seq: int, device: torch.device):
    starts = torch.randint(0, len(data) - seq - 1, (batch,))
    x = torch.stack([data[s:s + seq] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + seq + 1] for s in starts]).to(device)
    return x, y


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Plain UTF-8 text file")
    ap.add_argument("--out", default="runs/tiny")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=768)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float32", "bf16"], default="float32")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sample-tokens", type=int, default=200)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_bf16 = args.dtype == "bf16" and device.type == "cuda"

    raw = Path(args.data).read_bytes()
    if len(raw) < args.seq * 10:
        raise SystemExit("data is too small; provide at least 10 * seq bytes")
    data = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab = 256

    model = TinyLM(vocab, args.dim, args.layers, args.heads, args.kv_heads,
                   args.ffn, args.seq).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} tokens={len(data):,}")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"resumed from {args.resume}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    start = time.time()
    tokens_seen = 0
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch, args.seq, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - start
            tok_s = tokens_seen / max(elapsed, 1e-9)
            print(f"step {step:6d} loss {total_loss:.4f} lr {lr:.2e} tok/s {tok_s:8.0f}")

        if args.eval_every and (step % args.eval_every == 0 or step == args.steps - 1):
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    x, y = get_batch(val_data, args.batch, args.seq, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        _, loss = model(x, y)
                    losses.append(loss.item())
            model.train()
            mean = sum(losses) / len(losses)
            print(f"  val loss {mean:.4f}  perplexity {math.exp(min(mean, 20)):.2f}")
            torch.save({
                "model": model.state_dict(),
                "step": step,
                "config": vars(args),
                "val_loss": mean,
            }, out / "ckpt.pt")

    model.eval()
    prompt = b"The " if len(raw) >= 4 else raw[:1]
    idx = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    generated = model.generate(idx, args.sample_tokens)[0].tolist()
    print("\n--- sample ---")
    print(bytes([b % 256 for b in generated]).decode("utf-8", errors="replace"))
    torch.save({"model": model.state_dict(), "config": vars(args)}, out / "final.pt")
    print(f"\nsaved {out}/final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
