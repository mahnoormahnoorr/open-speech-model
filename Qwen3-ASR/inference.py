"""
Transcribe one utterance from the test set with a fine-tuned Qwen3-ASR checkpoint
and show it next to the reference transcript.

Usage:
    python infer_one.py                      # first test utterance, checkpoint-250
    python infer_one.py --index 12           # the 13th line of test.jsonl
    python infer_one.py --model Qwen/Qwen3-ASR-1.7B --index 12   # compare with the base model
"""
import argparse
import json

import torch
from qwen_asr import Qwen3ASRModel

p = argparse.ArgumentParser()
p.add_argument("--model", default="runs/qwen33-asr-fi/checkpoint-250")
p.add_argument("--test_file", default="data/test.jsonl")
p.add_argument("--index", type=int, default=0, help="Which line of the test file to use (0 = first)")
args = p.parse_args()

# Pick one line from the test set
with open(args.test_file, encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]
row = json.loads(lines[args.index])

# Reference text without the "language Finnish<asr_text>" prefix
reference = row["text"].split("<asr_text>", 1)[-1]

model = Qwen3ASRModel.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(audio=row["audio"])

print(f"Model:      {args.model}")
print(f"Audio:      {row['audio']}")
print(f"Language:   {results[0].language}")
print(f"Prediction: {results[0].text}")
print(f"Reference:  {reference}")
