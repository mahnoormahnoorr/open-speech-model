"""
Compare inference between the original (base) Qwen3-ASR model and a
fine-tuned checkpoint on the same set of Finnish validation audio files.

For each example, prints the gold transcript, the base model's prediction,
and the fine-tuned model's prediction, so you can judge the delta by eye/ear.

Usage:
    python3 compare_qwen3_asr.py \
        --base_model Qwen/Qwen3-ASR-1.7B \
        --checkpoint outputs/qwen3-asr-1.7b-ft/checkpoint-33 \
        --num_examples 5

    # Or just run one model at a time (same behavior as the original script):
    python3 compare_qwen3_asr.py --checkpoint outputs/qwen3-asr-1.7b-ft/checkpoint-33
    python3 compare_qwen3_asr.py --base_model Qwen/Qwen3-ASR-1.7B
"""
import argparse
import gc
import json

import torch
from qwen_asr import Qwen3ASRModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B",
                   help="Base/original model id or path. Set to '' to skip base-model inference.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a fine-tuned checkpoint dir. Set to '' or omit to skip fine-tuned inference.")
    p.add_argument("--eval_file", type=str, default="data/validation.jsonl")
    p.add_argument("--num_examples", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_model(path, device):
    print(f"Loading model from: {path}")
    return Qwen3ASRModel.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map=device,
    )


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_examples(eval_file, num_examples):
    examples = []
    with open(eval_file) as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            examples.append(json.loads(line))
    return examples


def run_inference(model, examples):
    """Returns a list of (predicted_text, language) tuples, one per example."""
    outputs = []
    for ex in examples:
        results = model.transcribe(audio=ex["audio"])
        outputs.append((results[0].text, results[0].language))
    return outputs


def main():
    args = parse_args()

    if not args.base_model and not args.checkpoint:
        raise SystemExit("Provide at least one of --base_model or --checkpoint.")

    examples = load_examples(args.eval_file, args.num_examples)
    print(f"\nLoaded {len(examples)} examples from {args.eval_file}\n")

    base_outputs = None
    ft_outputs = None

    # Load, run, and unload one model at a time to avoid holding two
    # copies of the model in GPU memory simultaneously.
    if args.base_model:
        base_model = load_model(args.base_model, args.device)
        base_outputs = run_inference(base_model, examples)
        unload_model(base_model)

    if args.checkpoint:
        ft_model = load_model(args.checkpoint, args.device)
        ft_outputs = run_inference(ft_model, examples)
        unload_model(ft_model)

    print("\n" + "=" * 80)
    print(f"{'BASE MODEL: ' + args.base_model if base_outputs else ''}")
    print(f"{'FINE-TUNED: ' + args.checkpoint if ft_outputs else ''}")
    print("=" * 80)

    for i, ex in enumerate(examples):
        print(f"[{i}] audio: {ex['audio']}")
        print(f"    gold:       {ex['text']}")
        if base_outputs:
            pred, lang = base_outputs[i]
            print(f"    base:       {pred!r}  (lang: {lang})")
        if ft_outputs:
            pred, lang = ft_outputs[i]
            print(f"    fine-tuned: {pred!r}  (lang: {lang})")
        print("-" * 80)


if __name__ == "__main__":
    main()
