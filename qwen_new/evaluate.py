"""
Transcribe the test set with one or more Qwen3-ASR models and score them.

    python qwen/evaluate.py --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B
    python qwen/evaluate.py --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B qwen3-asr-1.7b-fi
    python qwen/evaluate.py --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B qwen3-asr-1.7b-fi --quick

--models: Hugging Face ids and/or names of your fine-tuned models.
--quick: only the first 32 sentences, and your '-quick' fine-tuned models.

Results go to $WORK/results/<name>/:
    summary.md          WER, CER and speed per model (also printed in the log)
    side_by_side.tsv    one row per sentence: correct text, each model's transcript
                        and WER. Opens in Excel / LibreOffice.
    transcripts.jsonl   all transcripts, one line per sentence and model
"""
import argparse
import csv
import io
import json
import os
import time

import jiwer
import librosa
import torch
from datasets import load_from_disk
from qwen_asr import Qwen3ASRModel

WORK = os.environ.get("WORK", "/scratch/project_462001302/asr-data")
SAMPLE_RATE = 16000
LANGUAGE_NAMES = {"fi": "Finnish", "en": "English"}   # data language code -> Qwen's name


def normalize(text: str) -> str:
    """For WER: lowercase, punctuation removed, spaces collapsed (same in every script)."""
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return " ".join(text.split())


def load_audio(wav_bytes: bytes):
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
    return audio


def find_model(model: str, quick: bool) -> str:
    """Your fine-tuned model ($WORK/models/<name>/final) or a Hugging Face id."""
    for name in ([model + "-quick"] if quick else []) + [model]:
        path = os.path.join(WORK, "models", name, "final")
        if os.path.isdir(path):
            return path
    return model


def load_transcriber(model_path: str, language: str, batch_size: int):
    """Return a function: list of 16 kHz audio arrays -> list of transcripts."""
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=batch_size, max_new_tokens=256)

    def transcribe(audios):
        out = model.transcribe(audio=[(a, SAMPLE_RATE) for a in audios], language=LANGUAGE_NAMES[language])
        return [(o.text or "").strip() for o in out]

    return transcribe


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Prepared data name, e.g. fleurs-fi")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--quick", action="store_true", help="First 32 sentences and your '-quick' models")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    ds = load_from_disk(os.path.join(WORK, "data", args.data, args.split))
    if args.quick:
        ds = ds.select(range(min(32, len(ds))))
    language = ds[0]["language"]
    paths = [find_model(m, args.quick) for m in args.models]
    labels = [(os.path.basename(os.path.dirname(pth)) if pth.endswith("/final") else pth.split("/")[-1]).lower()
              for pth in paths]
    name = f"{args.data}-{args.split}-" + "-vs-".join(labels)
    if args.quick and not name.endswith("-quick"):
        name += "-quick"   # quick results never overwrite full ones
    out_dir = os.path.join(WORK, "results", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Data   : {args.data} / {args.split}, {len(ds)} sentences, language {language}")
    for label, pth in zip(labels, paths):
        print(f"Model  : {label}  ({pth})")
    print(f"Output : {out_dir}\n", flush=True)

    refs = list(ds["text"])
    audio_seconds = sum(ds["duration"])
    hyps, summary = {}, []
    for label, pth in zip(labels, paths):
        transcribe = load_transcriber(pth, language, args.batch_size)
        t0, out = time.time(), []
        for start in range(0, len(ds), 64):   # 64 sentences at a time to limit memory
            out += transcribe([load_audio(b) for b in ds[start:start + 64]["audio"]])
            print(f"  {label}: {len(out)}/{len(ds)}", flush=True)
        hyps[label] = out
        r, h = [normalize(t) for t in refs], [normalize(t) for t in out]
        summary.append({"model": label, "path": pth, "wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h),
                        "rtf": (time.time() - t0) / audio_seconds})
        print(f"  {label}: WER {summary[-1]['wer']:.2%}  CER {summary[-1]['cer']:.2%}\n")
        del transcribe
        torch.cuda.empty_cache()

    write_results(out_dir, ds, refs, hyps, summary)


def write_results(out_dir, ds, refs, hyps, summary):
    """The same three files for Whisper, Qwen and AITTA results."""
    labels = list(hyps)
    sentence_wer = {lab: [jiwer.wer(normalize(r) or "-", normalize(h)) for r, h in zip(refs, hyps[lab])]
                    for lab in labels}
    with open(os.path.join(out_dir, "transcripts.jsonl"), "w", encoding="utf-8") as f:
        for lab in labels:
            for uid, ref, hyp in zip(ds["id"], refs, hyps[lab]):
                f.write(json.dumps({"model": lab, "id": uid, "reference": ref, "transcript": hyp},
                                   ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "side_by_side.tsv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")   # utf-8-sig: Excel shows ä and ö correctly
        w.writerow(["id", "reference"] + labels + [f"WER {lab}" for lab in labels])
        for i, uid in enumerate(ds["id"]):
            w.writerow([uid, refs[i]] + [hyps[lab][i] for lab in labels]
                       + [f"{sentence_wer[lab][i]:.0%}" for lab in labels])
    lines = ["| Model | WER | CER | Speed (RTF) |", "|---|---|---|---|"]
    lines += [f"| {s['model']} | {s['wer']:.2%} | {s['cer']:.2%} | {s['rtf']:.3f} |" for s in summary]
    lines += ["", f"{len(refs)} sentences. WER/CER after lowercasing and removing punctuation. "
              "RTF = processing time / audio length (lower is faster)."]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
