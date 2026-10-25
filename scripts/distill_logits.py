#!/usr/bin/env python3
"""Reference logit knowledge-distillation trainer for same-tokenizer models.

This is a GPU-oriented reference, not the main Mac path. MLX-LM 0.32.x has no
teacher-logit loss; use sequence-level distillation with `mlx_lm.lora` for a
32 GB Mac, or run this script on CUDA with enough memory for both models.

Default recipe:
  loss = (1 - alpha) * CE(hard labels)
       + alpha * T^2 * KL(teacher_topk / T || student_topk / T)

The top-k option is important: a Gemma 4 vocabulary has 262,144 entries, so a
full 2K-token logits tensor is ~1 GB per model in fp16 before softmax.

Examples:
  # 12B teacher -> E4B student on one 80 GB GPU (short sequences)
  python scripts/distill_logits.py \
    --teacher google/gemma-4-12B-it \
    --student google/gemma-4-E4B-it \
    --data data/distill/e4b/train.jsonl \
    --out adapters/gemma4-e4b-distill \
    --device cuda:0 --dtype bf16 --max-seq-length 1024 \
    --top-k 64 --temperature 2.0 --alpha 0.6 \
    --lora-r 16 --grad-checkpoint

  # 31B teacher -> 12B student on two GPUs
  python scripts/distill_logits.py \
    --teacher google/gemma-4-31B-it --teacher-device cuda:1 \
    --student mlx-community/gemma-4-12b-it-4bit --device cuda:0 \
    --load-in-4bit --data data/distill/12b/train.jsonl \
    --out adapters/gemma4-12b-distill --top-k 64

Requires: torch, transformers, peft, and optionally bitsandbytes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def render_pair(tokenizer, row: dict, max_seq: int):
    """Return (input_ids, labels) with prompt tokens masked to -100."""
    tools = row.get("tools")
    if row.get("messages"):
        messages = row["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            return None
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
    elif row.get("prompt") and row.get("response"):
        prompt_text = str(row["prompt"])
        full_text = prompt_text + str(row["response"])
    else:
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # The full render should start with the prompt render. If a template inserts
    # an extra separator, fall back to the longest common prefix.
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if common == 0:
        return None

    if len(full_ids) > max_seq:
        return None
    labels = [-100] * common + full_ids[common:]
    if all(label == -100 for label in labels):
        return None
    return full_ids, labels


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    import torch

    max_len = max(len(ids) for ids, _ in batch)
    input_ids, labels, masks = [], [], []
    for ids, labs in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        masks.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def load_model(path: str, dtype, device: str, load_4bit: bool, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"trust_remote_code": trust_remote_code}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not load_4bit:
        model.to(device)
    model.eval()
    return model


def kd_loss(student_logits, teacher_logits, temperature: float, top_k: int):
    import torch
    import torch.nn.functional as F

    s = student_logits / temperature
    if top_k and top_k > 0:
        with torch.no_grad():
            top_values, top_index = teacher_logits.topk(min(top_k, teacher_logits.size(-1)), dim=-1)
            q = F.softmax(top_values / temperature, dim=-1)
        q = q.to(s.device)
        index = top_index.to(s.device)
        s_top = s.gather(-1, index)
        log_p = F.log_softmax(s_top, dim=-1)
        return -(q * log_p).sum(-1).mean()
    with torch.no_grad():
        p = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(F.log_softmax(s, dim=-1), p.to(s.device), reduction="batchmean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True, help="JSONL with messages or prompt/response")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher-device", default=None, help="default: same as --device")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--load-in-4bit", action="store_true", help="CUDA + bitsandbytes only")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="KD weight; 0 = CE only")
    ap.add_argument("--top-k", type=int, default=64, help="0 = full vocabulary")
    ap.add_argument("--lora-r", type=int, default=16, help="0 = full fine-tune")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_rows(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]
    pairs = []
    skipped = 0
    for row in rows:
        pair = render_pair(tokenizer, row, args.max_seq_length)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)
    if not pairs:
        print("no usable rows; check the data format or --max-seq-length", file=sys.stderr)
        return 2
    print(f"rows: usable={len(pairs)} skipped={skipped} max_seq={args.max_seq_length}")

    if args.dry_run:
        lengths = [len(ids) for ids, _ in pairs]
        target = [sum(1 for x in labels if x != -100) for _, labels in pairs]
        print(f"tokens: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.0f}")
        print(f"target tokens: min={min(target)} max={max(target)} mean={sum(target)/len(target):.0f}")
        print("no model was loaded (--dry-run)")
        return 0

    import torch
    from transformers import AutoTokenizer as _AutoTokenizer  # noqa: F401

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=args.trust_remote_code)
    if teacher_tok.get_vocab() != tokenizer.get_vocab():
        print("teacher and student vocabularies differ; logit KD needs the same tokenizer", file=sys.stderr)
        return 2
    if teacher_tok.pad_token_id is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.device.startswith("cuda") and args.dtype == "fp32":
        print("warning: fp32 on CUDA is slow; use bf16", file=sys.stderr)

    teacher_device = args.teacher_device or args.device
    print(f"loading teacher {args.teacher} on {teacher_device}")
    teacher = load_model(args.teacher, dtype, teacher_device, args.load_in_4bit, args.trust_remote_code)
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"loading student {args.student} on {args.device}")
    student = load_model(args.student, dtype, args.device, args.load_in_4bit, args.trust_remote_code)
    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit:
            student = prepare_model_for_kbit_training(student)
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        student = get_peft_model(
            student,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        student.print_trainable_parameters()
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches_per_epoch = max(1, math.ceil(len(pairs) / args.batch_size))
    total_steps = args.max_steps or (args.epochs * batches_per_epoch)
    if args.max_steps_per_epoch:
        total_steps = min(total_steps, args.epochs * args.max_steps_per_epoch)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    import torch.nn.functional as F

    def evaluate() -> dict:
        student.eval()
        total_ce, total_kd, n = 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(pairs), args.batch_size):
                batch = pairs[start:start + args.batch_size]
                input_ids, labels, mask = collate(batch, tokenizer.pad_token_id)
                input_ids, labels, mask = input_ids.to(args.device), labels.to(args.device), mask.to(args.device)
                out = student(input_ids=input_ids, attention_mask=mask).logits[:, :-1, :]
                target = labels[:, 1:]
                selected = target != -100
                ce = F.cross_entropy(out[selected], target[selected]).item() if selected.any() else 0.0
                total_ce += ce
                if args.alpha > 0:
                    t_ids = input_ids.to(teacher_device)
                    t_mask = mask.to(teacher_device)
                    t_logits = teacher(input_ids=t_ids, attention_mask=t_mask).logits[:, :-1, :]
                    t_selected = selected.to(t_logits.device)
                    kd = kd_loss(out[selected], t_logits[t_selected], args.temperature, args.top_k).item()
                    total_kd += kd
                n += 1
        student.train()
        return {"ce": total_ce / max(n, 1), "kd": total_kd / max(n, 1)}

    started = time.time()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    order = list(range(len(pairs)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        for batch_start in range(0, len(order), args.batch_size):
            if args.max_steps and step >= args.max_steps:
                break
            if args.max_steps_per_epoch and batch_start >= args.max_steps_per_epoch * args.batch_size:
                break
            batch = [pairs[i] for i in order[batch_start:batch_start + args.batch_size]]
            input_ids, labels, mask = collate(batch, tokenizer.pad_token_id)
            input_ids, labels, mask = input_ids.to(args.device), labels.to(args.device), mask.to(args.device)
            student_logits = student(input_ids=input_ids, attention_mask=mask).logits[:, :-1, :]
            target = labels[:, 1:]
            selected = target != -100
            ce = F.cross_entropy(student_logits[selected], target[selected])
            if args.alpha > 0:
                t_logits = teacher(
                    input_ids=input_ids.to(teacher_device),
                    attention_mask=mask.to(teacher_device),
                ).logits[:, :-1, :]
                t_selected = selected.to(t_logits.device)
                kd = kd_loss(student_logits[selected], t_logits[t_selected], args.temperature, args.top_k)
                loss = (1 - args.alpha) * ce + args.alpha * (args.temperature ** 2) * kd
            else:
                kd = torch.zeros((), device=args.device)
                loss = ce
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.report_every == 0:
                elapsed = time.time() - started
                print(
                    f"step {step:6d} ce {ce.item():.4f} kd {kd.item():.4f} "
                    f"lr {lr_at(step):.2e} {elapsed:.0f}s"
                )
            if args.eval_every and step > 0 and step % args.eval_every == 0:
                print("eval", json.dumps(evaluate()))
            if args.save_every and step > 0 and step % args.save_every == 0:
                save(student, tokenizer, args)
            step += 1
        if args.max_steps and step >= args.max_steps:
            break

    save(student, tokenizer, args)
    print(f"done: {step} steps, {time.time() - started:.0f}s, saved {args.out}")
    return 0


def save(student, tokenizer, args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "distill_config.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "student": args.student,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "top_k": args.top_k,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "target_modules": args.target_modules,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
