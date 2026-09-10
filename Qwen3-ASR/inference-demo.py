"""
Quick inference test for a fine-tuned Qwen3-ASR checkpoint.

Transcribes a handful of real Finnish validation audio files and prints
the model's output next to the gold transcript, so you can judge quality
by ear/eye -- not just by loss numbers.

Usage:
    python3 infer_qwen3_asr.py --checkpoint outputs/qwen3-asr-1.7b-ft/checkpoint-33 --num_examples 5
"""
import argparse
import json

import torch
from qwen_asr import Qwen3ASRModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a fine-tuned checkpoint dir, or a base model id "
                        "(e.g. Qwen/Qwen3-ASR-1.7B) to test the UN-fine-tuned model for comparison.")
    p.add_argument("--eval_file", type=str, default="data/validation.jsonl")
    p.add_argument("--num_examples", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = Qwen3ASRModel.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    examples = []
    with open(args.eval_file) as f:
        for i, line in enumerate(f):
            if i >= args.num_examples:
                break
            examples.append(json.loads(line))

    print(f"\nTranscribing {len(examples)} examples from {args.eval_file}\n")
    print("=" * 80)

    for i, ex in enumerate(examples):
        results = model.transcribe(audio=ex["audio"])
        predicted = results[0].text
        gold = ex["text"]

        print(f"[{i}] audio: {ex['audio']}")
        print(f"    gold:      {gold}")
        print(f"    predicted: {predicted!r}")
        print(f"    language:  {results[0].language}")
        print("-" * 80)


if __name__ == "__main__":
    main()
