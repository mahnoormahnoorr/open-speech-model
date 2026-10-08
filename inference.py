"""
Transcribe audio with a ready or fine-tuned model.

    # Your own files or folders (any format ffmpeg reads; long files are chunked)
    python inference.py --model whisper-small-fi --language fi --audio interview.mp3 my_audio/

    # A few sentences from prepared data, shown next to the correct text
    python inference.py --model whisper-small-fi --data fleurs-fi --num_samples 3

--model accepts Hugging Face ids and names of your fine-tuned models.
Try it quickly with --quick: 3 sentences or files only, and your '-quick'
fine-tuned model (from finetune --quick) if it exists.

The transcripts are always shown in the log (the first 10 in full if there are
many); --output_file additionally saves all of them.
Without --language the model detects the language itself (with --data the
data's language is used).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_common import (LANGUAGES, SAMPLE_RATE, data_language, decode_audio,  # noqa: E402
                        load_prepared_split, load_transcriber, print_setup, quick_model,
                        resolve_model)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".mp4", ".aac"}


def collect_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, n) for n in sorted(names)
                          if os.path.splitext(n)[1].lower() in AUDIO_EXTENSIONS]
        else:
            files.append(path)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face id or fine-tuned model name")
    p.add_argument("--language", default=None, choices=sorted(LANGUAGES))
    p.add_argument("--audio", nargs="*", default=[], help="Audio files and/or folders")
    p.add_argument("--data", "--data_dir", dest="data", default=None,
                   help="Prepared data name or path (alternative to --audio)")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=225)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_file", default=None, help="Also save all transcripts to this JSONL file")
    p.add_argument("--quick", action="store_true", help="Try it quickly (see above)")
    args = p.parse_args()

    if not args.audio and not args.data:
        p.error("Give --audio and/or --data")

    if args.quick:
        args.model = quick_model(args.model)
    files = collect_files(args.audio)
    if args.quick:
        files = files[:3]
    items = []  # (name, audio array, reference or None)
    for path in files:
        items.append((path, decode_audio(path=path), None))
    if args.data:
        ds = load_prepared_split(args.data, args.split)
        ds = ds.select(range(min(args.num_samples, len(ds))))
        for ex in ds:
            items.append((ex["id"], decode_audio(data=ex["audio"]), ex["text"]))

    model = resolve_model(args.model)
    language = args.language or (data_language(args.data) if args.data else None)
    n_files = len(items) - (min(args.num_samples, len(ds)) if args.data else 0)
    print_setup("Transcription setup", {
        "Model": model,
        "Language": language or "detected by the model",
        "Audio files": n_files,
        "Sentences from data": f"{len(items) - n_files} from {args.data} ({args.split})" if args.data else "-",
        "Save transcripts to": args.output_file or "(only printed in this log)",
    })
    transcribe, family = load_transcriber(model, language, args.batch_size, args.max_new_tokens)
    print(f"Model family: {family}")

    t0 = time.time()
    texts = transcribe([audio for _, audio, _ in items])
    elapsed = time.time() - t0
    total_audio = sum(len(a) for _, a, _ in items) / SAMPLE_RATE

    # Always show transcripts in the log. With --output_file (which gets all of
    # them), show the first SHOW_IN_LOG in full and shorten very long ones.
    SHOW_IN_LOG, MAX_CHARS = 10, 1000
    limit = SHOW_IN_LOG if args.output_file else len(items)
    for k, ((name, audio, ref), text) in enumerate(zip(items, texts)):
        if k >= limit:
            break
        shown = text if not args.output_file or len(text) <= MAX_CHARS else \
            text[:MAX_CHARS] + f" ... [{len(text) - MAX_CHARS} more characters in the output file]"
        print(f"\n=== {name} ({len(audio) / SAMPLE_RATE:.1f}s)")
        if ref is not None:
            print(f"Reference: {ref}")
        print(f"Model Output: {shown}")
    if len(items) > limit:
        print(f"\n... and {len(items) - limit} more transcripts in {args.output_file}")

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as out:
            for (name, audio, ref), text in zip(items, texts):
                out.write(json.dumps({"source": name, "duration_s": round(len(audio) / SAMPLE_RATE, 1),
                                      "reference": ref, "transcript": text},
                                     ensure_ascii=False) + "\n")
        print(f"\nAll {len(items)} transcripts saved to {args.output_file}")
    print(f"\nTranscribed {total_audio:.1f}s of audio in {elapsed:.1f}s")


if __name__ == "__main__":
    main()