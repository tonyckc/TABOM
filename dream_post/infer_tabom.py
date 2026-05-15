#!/usr/bin/env python3
"""
infer_tabom.py – Run inference with a TABOM-finetuned Dream checkpoint.

Loads the Dream base model, merges the LoRA adapter saved by train_tabom.py,
and generates responses using Dream's diffusion_generate.

Usage:
    # Interactive mode (reads prompts from stdin)
    python infer_tabom.py --checkpoint ./checkpoints/dream_gt_tabom/step_1000 --temperature 1.0 --verbose

    # Single prompt
    python infer_tabom.py --checkpoint ./checkpoints/tabom/final \
        --prompt "Explain the theory of relativity in simple terms."

    # From a file (one prompt per line)
    python infer_tabom.py --checkpoint ./checkpoints/tabom/final \
        --prompt_file prompts.txt
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference with a TABOM-finetuned Dream checkpoint"
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a LoRA checkpoint directory saved by train_tabom.py "
             "(e.g. ./checkpoints/tabom/final). Omit to run the base model directly."
    )
    p.add_argument(
        "--base_model", default="Dream-org/Dream-v0-Instruct-7B",
        help="Base Dream model. Used directly when --checkpoint is not provided, "
             "or as the backbone when loading a LoRA adapter."
    )
    p.add_argument("--prompt", default=None, help="Single prompt string")
    p.add_argument("--prompt_file", default=None,
                   help="Text file with one prompt per line")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--steps", type=int, default=256,
                   help="Number of diffusion denoising steps")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--alg", default="entropy",
                   choices=["entropy", "random", "topk_margin", "maskgit"],
                   help="Token unmasking algorithm")
    p.add_argument("--alg_temp", type=float, default=0.)
    p.add_argument("--merge_lora", action="store_true",
                   help="Merge LoRA weights into base model before inference "
                        "(faster generation, slightly more VRAM)")
    p.add_argument("--verbose", action="store_true",
                   help="Print verbose output, including diffusion history")
    p.add_argument("--padding_side", default="none", choices=["none", "left", "right"])
    return p.parse_args()


# ──────────────────────────────── Model ───────────────────────────────────────

def load_model(args, device: torch.device):
    # Tokenizer: prefer the checkpoint dir (has any added tokens / configs saved
    # during training), fall back to the base model if no checkpoint is given.
    tok_source = args.checkpoint if args.checkpoint else args.base_model
    print(f"Loading tokenizer from: {tok_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        trust_remote_code=True,
        padding_side="left",
    )

    print(f"Loading base model: {args.base_model}")
    base = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if args.checkpoint:
        print(f"Loading LoRA adapter from: {args.checkpoint}")
        model = PeftModel.from_pretrained(base, args.checkpoint)
        if args.merge_lora:
            print("Merging LoRA weights into base model …")
            model = model.merge_and_unload()
    else:
        print("No checkpoint provided — running base model directly.")
        model = base

    model.eval()
    return model, tokenizer


# ──────────────────────────── Generation ──────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, args, device: torch.device) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        # padding=True if args.padding_side != "none" else False,
        # padding_side=args.padding_side
    )
    # test for padding: left pad
    pad = torch.full(
        (inputs["input_ids"].shape[0], 10),
        tokenizer.pad_token_id,
        dtype=inputs["input_ids"].dtype
    )

    inputs["input_ids"] = torch.cat([pad, inputs["input_ids"]], dim=1)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # diffusion_generate lives on the base model; PeftModel forwards calls
    # through the adapter, so this works whether or not the LoRA is merged.
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=args.verbose,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )

    response_ids = output.sequences[0][input_ids.shape[1]:]
    text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=False)
    # Truncate at the first end-of-turn or EOS marker.
    # <|im_end|> is the Qwen2 chat-template turn-end token (different from pad_token).
    for stop in ["<|im_end|>", tokenizer.eos_token]:
        if stop and stop in text:
            text = text[: text.index(stop)]
            break
    
    return text.strip(), output.history


# ──────────────────────────────── Main ────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(args, device)

    # ── Collect prompts ───────────────────────────────────────────────────────
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\nEnter prompts interactively (Ctrl-D / Ctrl-C to quit):\n")
        prompts = None

    # ── Run ───────────────────────────────────────────────────────────────────
    if prompts is not None:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'─'*60}")
            print(f"[{i}] Prompt: {prompt}")
            print("─" * 60)
            response = generate(model, tokenizer, prompt, args, device)
            print(f"Response: {response}")
    else:
        try:
            while True:
                try:
                    prompt = input("\nYou: ").strip()
                except EOFError:
                    break
                if not prompt:
                    continue
                if prompt.lower() in {"exit", "quit"}:
                    break
                response, history = generate(model, tokenizer, prompt, args, device)
                print(f"Model: {response}")
                if args.verbose:
                    print("\nDiffusion history:")
                    # print(history) # 2-dimensional tensor, each row is a token sequence
                    for i, step in enumerate(history):
                        # print(step)
                        if i % 10 != 0:
                            continue
                        print(f"Step {i}: {tokenizer.decode(step[0].tolist())}")
                        print("─" * 60)
        except KeyboardInterrupt:
            pass
        print("\nDone.")


if __name__ == "__main__":
    main()
