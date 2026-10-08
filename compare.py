"""
Run one or more models on the same prepared data and compare them.

    python compare.py --data fleurs-en --models openai/whisper-large-v3-turbo Qwen/Qwen3-ASR-1.7B
    python compare.py --data fleurs-fi --models openai/whisper-small whisper-small-fi

--models accepts Hugging Face ids and names of your fine-tuned models.
With a single model this is simply an evaluation.

Try it quickly with --quick: only the first 32 sentences, and your '-quick'
fine-tuned models (from finetune --quick) if they exist.

Writes a folder (default $OUTPUT_DIR/comparisons/<name>) with:
    summary.md          WER, CER and speed per model (also printed)
    side_by_side.tsv    one row per sentence: reference, each model's
                        transcript and its WER. Opens in Excel/LibreOffice.
    summary.json        the same numbers for scripts
"""
import argparse
import csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_common import (LANGUAGES, data_language, decode_audio,  # noqa: E402
                        load_prepared_split, load_transcriber, model_label,
                        output_root, print_setup, quick_model,
                        resolve_model, wer_cer)


def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clean(text: str) -> str:
    """Keep the TSV one line per sentence."""
    return " ".join(str(text).split())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data", "--data_dir", dest="data", required=True,
                   help="Prepared data name (e.g. fleurs-fi) or path")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Column names for the models (default: derived from the model names)")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Only the first N sentences (32 with --quick)")
    p.add_argument("--quick", action="store_true", help="Try it quickly (see above)")
    p.add_argument("--language", default=None, choices=sorted(LANGUAGES),
                   help="Default: the language stored in the data")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=225)
    p.add_argument("--name", default=None, help="Name of the output folder")
    p.add_argument("--output_dir", default=None, help="Default: $OUTPUT_DIR/comparisons/<name>")
    args = p.parse_args()

    if args.quick:
        args.models = [quick_model(m) for m in args.models]
        args.max_samples = args.max_samples or 32
    language = args.language or data_language(args.data)
    labels = args.labels or [model_label(resolve_model(m)) for m in args.models]
    if len(labels) != len(args.models):
        p.error("Give as many --labels as --models")
    if len(set(labels)) != len(labels):  # make duplicate names unique
        labels = [f"{lab}-{i + 1}" for i, lab in enumerate(labels)]

    ds = load_prepared_split(args.data, args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    data_name = os.path.basename(args.data.rstrip("/"))
    name = args.name or f"{data_name}-{args.split}-" + "-vs-".join(labels)
    if args.quick and not args.name and not name.endswith("-quick"):
        name += "-quick"
    out_dir = args.output_dir or os.path.join(output_root(), "comparisons", name)
    os.makedirs(out_dir, exist_ok=True)

    settings = {"Data": f"{args.data}, split '{args.split}'",
                "Sentences": f"{len(ds)} ({sum(ds['duration']) / 60:.1f} min of audio)",
                "Language": language}
    for i, (lab, m) in enumerate(zip(labels, args.models), 1):
        settings[f"Model {i}"] = f"{lab}  ({resolve_model(m)})"
    settings["Output"] = out_dir
    print_setup("Comparison setup", settings)
    print()

    references = [clean(t) for t in ds["text"]]
    durations = ds["duration"]
    audio_seconds = sum(durations)
    predictions, results = {}, []

    for model, label in zip(args.models, labels):
        print(f"=== {label}")
        t0 = time.time()
        transcribe, family = load_transcriber(model, language, args.batch_size, args.max_new_tokens)
        print(f"  {family} model loaded in {time.time() - t0:.0f}s")
        hyps, t_run = [], 0.0
        chunk = args.batch_size * 4  # decode a few batches at a time to limit memory
        for start in range(0, len(ds), chunk):
            audios = [decode_audio(data=b) for b in ds[start:start + chunk]["audio"]]
            t1 = time.time()
            hyps += [clean(h) for h in transcribe(audios)]
            t_run += time.time() - t1
            print(f"  {min(start + chunk, len(ds))}/{len(ds)} sentences")
        predictions[label] = hyps
        metrics = wer_cer(references, hyps)
        results.append({"label": label, "model": resolve_model(model), "family": family,
                        "wer": metrics["wer"], "cer": metrics["cer"],
                        "real_time_factor": t_run / max(audio_seconds, 1e-9)})
        print(f"  WER {metrics['wer']:.2%}  CER {metrics['cer']:.2%}\n")
        del transcribe
        free_gpu_memory()

    # ---- side_by_side.tsv: one row per sentence ------------------------------
    sentence_wer = {lab: [wer_cer([r], [h])["wer"] for r, h in zip(references, predictions[lab])]
                    for lab in labels}
    tsv = os.path.join(out_dir, "side_by_side.tsv")
    # utf-8-sig so Excel shows ä/ö correctly
    with open(tsv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "duration_s", "reference"] + labels + [f"WER {lab}" for lab in labels])
        for i, uid in enumerate(ds["id"]):
            w.writerow([uid, f"{durations[i]:.1f}", references[i]]
                       + [predictions[lab][i] for lab in labels]
                       + [f"{sentence_wer[lab][i]:.0%}" for lab in labels])

    # ---- summary ---------------------------------------------------------------
    lines = [f"# Comparison: {name}", "",
             f"Data: `{args.data}`, split `{args.split}`, {len(ds)} sentences, "
             f"{audio_seconds / 60:.1f} min of audio, language `{language}`", "",
             "| Model | WER | CER | Speed (RTF) |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['wer']:.2%} | {r['cer']:.2%} | {r['real_time_factor']:.3f} |")
    lines += ["", "RTF = processing time / audio duration (lower is faster). "
              "WER/CER after lowercasing and removing punctuation.", "",
              "| Model | Path or id |", "|---|---|"]
    lines += [f"| {r['label']} | `{r['model']}` |" for r in results]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "data": args.data, "split": args.split, "language": language,
                   "num_sentences": len(ds), "results": results}, f, indent=2, ensure_ascii=False)

    # ---- printout: examples where the models differ most -------------------------
    if len(labels) > 1:
        spread = [max(sentence_wer[lab][i] for lab in labels) - min(sentence_wer[lab][i] for lab in labels)
                  for i in range(len(ds))]
        title = "Sentences where the models differ most"
    else:
        spread = sentence_wer[labels[0]]
        title = "Sentences with the most errors"
    order = sorted(range(len(ds)), key=lambda i: spread[i], reverse=True)[:5]
    width = max(len("reference"), *(len(lab) for lab in labels))
    print(f"--- {title}:")
    for i in order:
        print(f"\n  {'reference':<{width}} : {references[i]}")
        for lab in labels:
            print(f"  {lab:<{width}} : {predictions[lab][i]}   (WER {sentence_wer[lab][i]:.0%})")

    print("\n" + summary)
    print(f"Side-by-side transcripts: {tsv}")
    print(f"All files: {out_dir}")


if __name__ == "__main__":
    main()