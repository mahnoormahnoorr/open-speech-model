"""
Download a dataset and convert it to the format every other script reads
(run once per dataset + language).

The result is a Hugging Face dataset saved with save_to_disk(), one directory
per split, with these columns:

    id        string   unique utterance id
    audio     binary   16 kHz mono 16-bit WAV bytes
    text      string   transcript, original casing and punctuation kept
    language  string   "fi" or "en"
    duration  float32  seconds

Sources:
    --dataset fleurs     google/fleurs from the Hugging Face Hub (default)
    --dataset parquet    your own parquet files (e.g. a FLEURS-like layout)
    --dataset manifest   your own JSONL files: {"audio_path": ..., "text": ...}

Examples:
    # Full FLEURS Finnish (takes a few minutes). The other scripts' --quick
    # option then uses only part of it, so there is no separate quick dataset.
    python prepare_data.py --language fi                  # -> data name fleurs-fi

    # Your own data
    python prepare_data.py --language fi --dataset manifest --output_dir mydata-fi \\
        --split_files train=my_train.jsonl test=my_test.jsonl

How it stays fast: FLEURS files are downloaded in parallel with huggingface_hub
(using the fast hf-xet transfer that is in the container) into the shared
download cache, so they are only downloaded once. They are then read directly
with pyarrow, and audio that is already 16 kHz mono 16-bit WAV (all of FLEURS)
is stored as-is instead of being decoded and re-encoded.
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_common import (LANGUAGES, SAMPLE_RATE, data_root, decode_audio, encode_wav,  # noqa: E402
                        print_setup)

TEXT_COLUMN_CANDIDATES = ["raw_transcription", "sentence", "transcription", "text"]
FLEURS_REPO = "google/fleurs"
# google/fleurs is a script-based dataset; the Hub's automatic parquet version
# lives on this branch, under <config>/<split>/NNNN.parquet
FLEURS_REVISIONS = ("refs/convert/parquet", "main")


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
def download_fleurs(lang_config: str, splits: list[str], first_file_only: bool) -> dict:
    """Download the parquet files for the given splits. Returns {split: [local paths]}."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for revision in FLEURS_REVISIONS:
        repo_files = api.list_repo_files(FLEURS_REPO, repo_type="dataset", revision=revision)
        wanted = {}
        for split in splits:
            files = sorted(f for f in repo_files
                           if f.startswith(f"{lang_config}/{split}/") and f.endswith(".parquet"))
            if files:
                wanted[split] = files[:1] if first_file_only else files
        if wanted:
            break
    else:
        raise RuntimeError(f"No parquet files for {FLEURS_REPO} {lang_config}")
    missing = [s for s in splits if s not in wanted]
    if missing:
        raise RuntimeError(f"{FLEURS_REPO} {lang_config} has no parquet files for {missing}")

    all_files = [f for files in wanted.values() for f in files]
    print(f"Downloading {len(all_files)} file(s) from {FLEURS_REPO} ({revision}) ...", flush=True)
    local_root = snapshot_download(FLEURS_REPO, repo_type="dataset", revision=revision,
                                   allow_patterns=all_files, max_workers=8,
                                   token=os.environ.get("HF_TOKEN"))
    return {split: [os.path.join(local_root, f) for f in files] for split, files in wanted.items()}


# --------------------------------------------------------------------------
# Readers: each yields dicts {"id", "audio_bytes" | "audio_path", "text"}
# --------------------------------------------------------------------------
def iter_parquet(files: list[str], text_column: str | None, audio_column: str):
    """Read local parquet files directly with pyarrow, a batch at a time."""
    import pyarrow.parquet as pq

    for path in files:
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names
        if text_column is None:
            text_column = next((c for c in TEXT_COLUMN_CANDIDATES if c in columns), None)
            if text_column is None:
                raise RuntimeError(f"No transcript column among {TEXT_COLUMN_CANDIDATES}; "
                                   f"columns are {columns}. Use --text_column.")
            print(f"  transcript column: '{text_column}'")
        read = [c for c in ("id", audio_column, text_column) if c in columns]
        for batch in pf.iter_batches(batch_size=64, columns=read):
            rows = batch.to_pylist()
            for i, row in enumerate(rows):
                audio = row[audio_column]
                if isinstance(audio, dict):        # {"bytes": ..., "path": ...}
                    audio_bytes, audio_path = audio.get("bytes"), audio.get("path")
                else:                              # plain binary column
                    audio_bytes, audio_path = audio, None
                yield {"id": str(row.get("id", i)), "audio_bytes": audio_bytes,
                       "audio_path": None if audio_bytes else audio_path,
                       "text": row.get(text_column) or ""}


def iter_manifest(path: str, text_column: str | None, audio_column: str):
    base = os.path.dirname(os.path.abspath(path))
    text_column = text_column or "text"
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            audio_path = ex[audio_column]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base, audio_path)
            yield {"id": str(ex.get("id", i)), "audio_bytes": None,
                   "audio_path": audio_path, "text": ex.get(text_column) or ""}


# --------------------------------------------------------------------------
# Conversion to the canonical format
# --------------------------------------------------------------------------
def wav_passthrough(data: bytes | None):
    """
    If `data` already is a 16 kHz mono 16-bit PCM WAV, return (data, duration)
    so it can be stored unchanged. Otherwise return None (needs converting).
    """
    if not data or data[:4] != b"RIFF":
        return None
    try:
        with wave.open(io.BytesIO(data)) as w:
            ok = (w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
                  and w.getsampwidth() == 2)
            frames = w.getnframes()
    except Exception:
        return None
    # Some writers put a wrong length in the header; only trust a plausible one.
    if not ok or frames <= 0 or frames * 2 > len(data):
        return None
    return data, frames / SAMPLE_RATE


def convert_split(source, split: str, language: str, limit: int | None,
                  min_dur: float, max_dur: float, out_dir: str) -> dict:
    from datasets import Dataset, Features, Value
    from datasets.arrow_writer import ArrowWriter

    features = Features({"id": Value("string"), "audio": Value("binary"),
                         "text": Value("string"), "language": Value("string"),
                         "duration": Value("float32")})
    stats = {"kept": 0, "too_short": 0, "too_long": 0, "empty_text": 0,
             "decode_error": 0, "hours": 0.0, "converted_audio": 0}

    def generate():
        for n, ex in enumerate(source):
            if limit is not None and stats["kept"] >= limit:
                break
            text = " ".join(ex["text"].split())
            if not text:
                stats["empty_text"] += 1
                continue
            ready = wav_passthrough(ex["audio_bytes"])
            if ready:
                audio_bytes, duration = ready
            else:
                try:
                    audio = decode_audio(data=ex["audio_bytes"], path=ex["audio_path"])
                except Exception as e:
                    stats["decode_error"] += 1
                    if stats["decode_error"] <= 3:
                        print(f"  WARNING: could not decode {ex['id']}: {e}")
                    continue
                audio_bytes, duration = encode_wav(audio), len(audio) / SAMPLE_RATE
                stats["converted_audio"] += 1
            if duration < min_dur:
                stats["too_short"] += 1
                continue
            if duration > max_dur:
                stats["too_long"] += 1
                continue
            stats["kept"] += 1
            stats["hours"] += duration / 3600
            if stats["kept"] % 1000 == 0:
                print(f"  {stats['kept']} sentences ...", flush=True)
            yield {"id": f"{split}-{n:06d}-{ex['id']}", "audio": audio_bytes,
                   "text": text, "language": language, "duration": duration}

    # Write examples to an Arrow file as they come (low memory even for big
    # corpora), then save it as a regular dataset directory.
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".tmp-") as tmp:
        arrow_path = os.path.join(tmp, "data.arrow")
        with ArrowWriter(features=features, path=arrow_path) as writer:
            for example in generate():
                writer.write(example)
            writer.finalize()
        ds = Dataset.from_file(arrow_path)
        ds.save_to_disk(os.path.join(out_dir, split))
        del ds
    stats["hours"] = round(stats["hours"], 3)
    return stats


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(root, f))
               for root, _, files in os.walk(path) for f in files)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def print_report(summary: dict, out_dir: str) -> None:
    rows = []
    for split, st in summary["splits"].items():
        skipped = {k: st[k] for k in ("too_short", "too_long", "empty_text", "decode_error") if st[k]}
        rows.append((split, st["kept"], st["hours"], st["size_bytes"], st["seconds"],
                     ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in skipped.items()) or "-"))
    total = ("total", sum(r[1] for r in rows), sum(r[2] for r in rows),
             sum(r[3] for r in rows), sum(r[4] for r in rows), "")

    print("\n========== Data preparation finished")
    print(f"{'Split':<11}{'Sentences':>10}{'Audio':>10}{'Size':>10}{'Time':>12}   Skipped")
    for split, n, hours, size, secs, skipped in rows + [total]:
        if split == "total":
            print("-" * 56)
        print(f"{split:<11}{n:>10}{hours:>9.2f}h{human_size(size):>10}{human_time(secs):>12}   {skipped}")
    if summary.get("download_seconds") is not None:
        print(f"\nDownload   : {human_size(summary['download_bytes'])} in "
              f"{human_time(summary['download_seconds'])} (cached for later runs)")
    print(f"Total time : {human_time(summary['total_seconds'])}")
    print(f"Saved in   : {out_dir}")
    print(f"Data name  : {os.path.basename(out_dir.rstrip('/'))}   "
          f"(use it with --data {os.path.basename(out_dir.rstrip('/'))})")
    print("=" * 39, flush=True)


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=sorted(LANGUAGES))
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "parquet", "manifest"])
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--split_files", nargs="+", default=[], metavar="SPLIT=PATH[,PATH]",
                   help="For --dataset parquet/manifest, e.g. train=a.parquet,b.parquet test=t.parquet")
    p.add_argument("--text_column", default=None, help="Transcript column (default: auto-detect)")
    p.add_argument("--audio_column", default=None,
                   help="Audio column (default: 'audio' for parquet, 'audio_path' for manifest)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max sentences for the train split (saved as '<name>-subset')")
    p.add_argument("--eval_limit", type=int, default=None, help="Max sentences for other splits")
    p.add_argument("--min_duration", type=float, default=0.5)
    p.add_argument("--max_duration", type=float, default=30.0)
    p.add_argument("--output_dir", default=None,
                   help="Name under $DATA_DIR or a path. Default: <dataset>-<language>[-subset]")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    subset = args.limit is not None or args.eval_limit is not None
    if args.output_dir and not os.path.isabs(args.output_dir) and os.sep not in args.output_dir:
        args.output_dir = os.path.join(data_root(), args.output_dir)  # short name -> under $DATA_DIR
    out_dir = args.output_dir or os.path.join(
        data_root(), f"{args.dataset}-{args.language}" + ("-subset" if subset else ""))
    if os.path.exists(os.path.join(out_dir, "summary.json")) and not args.overwrite:
        print(f"Prepared data already exists at {out_dir} (use --overwrite to redo).")
        print(f"Use it with: --data {os.path.basename(out_dir.rstrip('/'))}")
        return
    if args.overwrite:
        for split in args.splits:
            shutil.rmtree(os.path.join(out_dir, split), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    split_files = {}
    for item in args.split_files:
        split, _, paths = item.partition("=")
        split_files[split] = paths.split(",")
    if args.dataset != "fleurs":
        missing = [s for s in args.splits if s not in split_files]
        if missing:
            print(f"No --split_files for {missing}; skipping those splits.")
            args.splits = [s for s in args.splits if s in split_files]

    print_setup("Data preparation setup", {
        "Dataset": args.dataset + (f" ({LANGUAGES[args.language]['fleurs']})" if args.dataset == "fleurs" else ""),
        "Language": args.language,
        "Splits": ", ".join(args.splits),
        "Limit": (f"{args.limit or 'all'} training / {args.eval_limit or 'all'} other sentences"
                  if subset else "none (full dataset)"),
        "Clip length": f"{args.min_duration}-{args.max_duration} s",
        "Output": out_dir,
        "Data name": os.path.basename(out_dir.rstrip("/")),
    })

    summary = {"dataset": args.dataset, "language": args.language, "splits": {},
               "download_seconds": None, "download_bytes": 0}

    # ---- 1. Download (FLEURS only; a subset needs just the first file per split)
    if args.dataset == "fleurs":
        t0 = time.time()
        split_files = download_fleurs(LANGUAGES[args.language]["fleurs"], args.splits,
                                      first_file_only=subset)
        summary["download_seconds"] = round(time.time() - t0, 1)
        summary["download_bytes"] = sum(os.path.getsize(f) for fs in split_files.values() for f in fs)
        print(f"Downloaded {human_size(summary['download_bytes'])} in "
              f"{human_time(summary['download_seconds'])}", flush=True)

    # ---- 2. Convert each split
    for split in args.splits:
        limit = args.limit if split == "train" else args.eval_limit
        print(f"\n[{split}] converting (limit={limit or 'none'})", flush=True)
        t0 = time.time()
        if args.dataset in ("fleurs", "parquet"):
            source = iter_parquet(split_files[split], args.text_column, args.audio_column or "audio")
        else:
            assert len(split_files[split]) == 1, "Give one JSONL file per split"
            source = iter_manifest(split_files[split][0], args.text_column,
                                   args.audio_column or "audio_path")
        stats = convert_split(source, split, args.language, limit,
                              args.min_duration, args.max_duration, out_dir)
        stats["seconds"] = round(time.time() - t0, 1)
        stats["size_bytes"] = dir_size(os.path.join(out_dir, split))
        summary["splits"][split] = stats
        print(f"[{split}] {stats['kept']} sentences, {stats['hours']:.2f} h, "
              f"{human_time(stats['seconds'])}", flush=True)

    summary["total_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_report(summary, out_dir)


if __name__ == "__main__":
    main()