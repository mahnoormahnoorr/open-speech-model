"""
Inspect and evaluate a Qwen3-ASR model (base or fine-tuned) on a jsonl test set.

The jsonl must have lines like:
    {"audio": "/path/to/0.wav", "text": "language Finnish<asr_text>Transcript here."}

Two modes:

  one   Transcribe a single utterance and show it next to the reference.
        python asr_eval.py one  --model MODEL [--test_file FILE] [--index N]

  eval  Transcribe the whole test set and report WER and CER.
        python asr_eval.py eval --model MODEL [--test_file FILE] [--out FILE] [--limit N]

MODEL is a Hugging Face id (e.g. Qwen/Qwen3-ASR-1.7B) or a local model folder
(e.g. qwen33-asr-finetuning-out or qwen3-asr-finetuning-out/checkpoint-129).

Examples:
    python asr_eval.py one  --model qwen33-asr-finetuning-out --index 12
    python asr_eval.py eval --model Qwen/Qwen3-ASR-1.7B       --out results_base.jsonl
    python asr_eval.py eval --model qwen33-asr-finetuning-out --out results_ft.jsonl
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import jiwer
import torch
from qwen_asr import Qwen3ASRModel


# ---------------------------------------------------------------- helpers

def strip_prefix(text: str) -> str:
    """Remove the 'language X<asr_text>' prefix used in the training data."""
    return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Keeps ä, ö, å."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def read_rows(path: str):
    if not os.path.isfile(path):
        sys.exit(f"Error: test file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_model_path(model: str):
    """Fail early with a clear message instead of a confusing Hub error.

    A Hugging Face id has exactly one '/' (namespace/name). Anything that
    starts with '/', '.' or '~', or has more slashes, is treated as a local
    folder and must exist and contain the processor config.
    """
    looks_local = model.startswith(("/", ".", "~")) or model.count("/") != 1 or os.path.isdir(model)
    if not looks_local:
        return
    if not os.path.isdir(model):
        sys.exit(f"Error: model folder not found: {model}")
    if not os.path.isfile(os.path.join(model, "preprocessor_config.json")):
        sys.exit(
            f"Error: {model} has no preprocessor_config.json.\n"
            "Copy the processor files from the base model or a complete output folder."
        )


def load_model(model: str):
    check_model_path(model)
    return Qwen3ASRModel.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda:0")


def transcribe(model, audio_path: str):
    result = model.transcribe(audio=audio_path)[0]
    return result.text or "", result.language


# ---------------------------------------------------------------- modes

def run_one(args):
    rows = read_rows(args.test_file)
    if not 0 <= args.index < len(rows):
        sys.exit(f"Error: --index {args.index} is out of range (test file has {len(rows)} lines)")
    row = rows[args.index]

    model = load_model(args.model)
    hyp, language = transcribe(model, row["audio"])
    ref = strip_prefix(row["text"])

    print(f"Model:      {args.model}")
    print(f"Audio:      {row['audio']}")
    print(f"Language:   {language}")
    print(f"Prediction: {hyp}")
    print(f"Reference:  {ref}")

    ref_n, hyp_n = normalize(ref), normalize(hyp)
    if ref_n:
        print(f"WER:        {jiwer.wer(ref_n, hyp_n) * 100:.2f}%")
        print(f"CER:        {jiwer.cer(ref_n, hyp_n) * 100:.2f}%")


def run_eval(args):
    rows = read_rows(args.test_file)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or f"results_{os.path.basename(os.path.normpath(args.model))}.jsonl"
    print(f"Evaluating {args.model} on {len(rows)} utterances")

    model = load_model(args.model)

    refs, hyps = [], []
    langs = Counter()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            hyp, language = transcribe(model, row["audio"])
            ref = strip_prefix(row["text"])
            langs[language] += 1

            ref_n, hyp_n = normalize(ref), normalize(hyp)
            if ref_n:  # jiwer cannot score empty references
                refs.append(ref_n)
                hyps.append(hyp_n)

            out.write(json.dumps({
                "audio": row["audio"],
                "ref": ref,
                "hyp": hyp,
                "language": language,
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done", flush=True)

    if not refs:
        sys.exit("Error: no non-empty references to score.")

    print(f"\nModel: {args.model}")
    print(f"Utterances scored: {len(refs)}")
    print(f"WER: {jiwer.wer(refs, hyps) * 100:.2f}%")
    print(f"CER: {jiwer.cer(refs, hyps) * 100:.2f}%")
    print(f"Detected languages: {dict(langs)}")
    print(f"Per-utterance results written to {out_path}")


# ---------------------------------------------------------------- CLI

def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True,
                        help="Hugging Face id or local model folder")
    common.add_argument("--test_file", default="data/test.jsonl",
                        help="jsonl file with 'audio' and 'text' fields")

    p = argparse.ArgumentParser(description="Inspect and evaluate a Qwen3-ASR model.")
    sub = p.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("one", parents=[common],
                         help="Transcribe one utterance and compare with the reference")
    one.add_argument("--index", type=int, default=0,
                     help="Which line of the test file to use (0 = first)")
    one.set_defaults(func=run_one)

    ev = sub.add_parser("eval", parents=[common],
                        help="Evaluate the whole test set (WER and CER)")
    ev.add_argument("--out", default=None,
                    help="Per-utterance results file (default: results_<model>.jsonl)")
    ev.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N lines")
    ev.set_defaults(func=run_eval)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
