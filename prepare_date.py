"""
Download speech data and save it in the format that the Whisper, Qwen and
AITTA scripts all read. Run once per language.

    python prepare_data.py --language fi                  # -> data/fleurs-fi
    python prepare_data.py --language en                  # -> data/fleurs-en

    # Your own data: one JSONL file per split, lines like
    #   {"audio": "clips/0001.mp3", "text": "Transcript here."}
    # (audio paths are relative to the JSONL file, any format)
    python prepare_data.py --language fi --name mydata-fi \\
        --own_data train=my_train.jsonl validation=my_val.jsonl test=my_test.jsonl

Output: $WORK/data/<name>/{train,validation,test}, each a Hugging Face dataset
(save_to_disk) with these columns:

    id        unique id of the recording
    audio     the recording as a complete 16 kHz mono 16-bit WAV file (bytes)
    text      the correct transcript, original casing and punctuation
    language  "fi" or "en"
    duration  length in seconds
"""
import argparse
import io
import json
import os
import shutil
import tempfile
import time

import librosa
import numpy as np
import soundfile as sf

WORK = os.environ.get("WORK", "/scratch/project_462001302/asr-data")
SAMPLE_RATE = 16000
FLEURS_CONFIG = {"fi": "fi_fi", "en": "en_us"}
MIN_SECONDS, MAX_SECONDS = 0.5, 30.0   # Whisper cannot take more than 30 s in training


# ---------------------------------------------------------------- sources
def read_fleurs(language: str, split: str):
    """Yield (id, audio bytes, text) from google/fleurs on the Hugging Face Hub."""
    from huggingface_hub import HfApi, snapshot_download
    import pyarrow.parquet as pq

    # google/fleurs is a script-based dataset; the Hub keeps a parquet copy on this branch
    revision = "refs/convert/parquet"
    config = FLEURS_CONFIG[language]
    files = sorted(f for f in HfApi().list_repo_files("google/fleurs", repo_type="dataset", revision=revision)
                   if f.startswith(f"{config}/{split}/") and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"No FLEURS files found for {config}/{split}")
    local = snapshot_download("google/fleurs", repo_type="dataset", revision=revision,
                              allow_patterns=files, max_workers=8)
    for path in files:
        for batch in pq.ParquetFile(os.path.join(local, path)).iter_batches(
                batch_size=64, columns=["id", "audio", "transcription", "raw_transcription"]):
            for row in batch.to_pylist():
                yield str(row["id"]), row["audio"]["bytes"], row["raw_transcription"] or row["transcription"]


def read_own_jsonl(path: str):
    """Yield (id, audio bytes, text) from a JSONL file with "audio" and "text"."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                audio_path = os.path.join(base, row["audio"])   # absolute paths stay as they are
                with open(audio_path, "rb") as a:
                    yield str(row.get("id", i)), a.read(), row["text"]


# ---------------------------------------------------------------- audio
def to_wav_16k(audio_bytes: bytes):
    """Return (16 kHz mono 16-bit WAV bytes, duration in seconds)."""
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate == SAMPLE_RATE and info.channels == 1 and info.subtype == "PCM_16" \
                and info.format == "WAV":
            return audio_bytes, info.frames / SAMPLE_RATE      # already right: keep as is
    except Exception:
        pass
    # Anything else (other sample rate, stereo, mp3, flac, ...): decode and convert
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue(), len(audio) / SAMPLE_RATE


# ---------------------------------------------------------------- conversion
def convert(source, split: str, language: str, out_dir: str) -> dict:
    from datasets import Dataset, Features, Value
    from datasets.arrow_writer import ArrowWriter

    counts = {"kept": 0, "skipped_empty_text": 0, "skipped_length": 0, "skipped_unreadable": 0, "hours": 0.0}

    def rows():
        for n, (uid, audio_bytes, text) in enumerate(source):
            text = " ".join((text or "").split())
            if not text:
                counts["skipped_empty_text"] += 1
                continue
            try:
                wav, duration = to_wav_16k(audio_bytes)
            except Exception as e:
                counts["skipped_unreadable"] += 1
                print(f"  could not read audio {uid}: {e}")
                continue
            if not MIN_SECONDS <= duration <= MAX_SECONDS:
                counts["skipped_length"] += 1
                continue
            counts["kept"] += 1
            counts["hours"] += duration / 3600
            yield {"id": f"{split}-{n:06d}-{uid}", "audio": wav, "text": text,
                   "language": language, "duration": duration}

    features = Features({"id": Value("string"), "audio": Value("binary"), "text": Value("string"),
                         "language": Value("string"), "duration": Value("float32")})
    # Write rows to an Arrow file one by one (little memory), then save as a dataset folder
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        arrow_file = os.path.join(tmp, "data.arrow")
        with ArrowWriter(features=features, path=arrow_file) as writer:
            for row in rows():
                writer.write(row)
            writer.finalize()
        Dataset.from_file(arrow_file).save_to_disk(os.path.join(out_dir, split))
    counts["hours"] = round(counts["hours"], 2)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=sorted(FLEURS_CONFIG))
    p.add_argument("--name", default=None, help="Name of the data folder (default: fleurs-<language>)")
    p.add_argument("--own_data", nargs="+", default=None, metavar="SPLIT=FILE.jsonl",
                   help="Use your own data instead of FLEURS")
    p.add_argument("--overwrite", action="store_true", help="Prepare again even if the data exists")
    args = p.parse_args()

    if args.own_data and not args.name:
        p.error("Give --name for your own data, e.g. --name mydata-fi")
    name = args.name or f"fleurs-{args.language}"
    out_dir = os.path.join(WORK, "data", name)
    if os.path.exists(os.path.join(out_dir, "summary.json")) and not args.overwrite:
        print(f"Data already prepared in {out_dir}. Use --overwrite to prepare it again.")
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    own = dict(item.split("=", 1) for item in args.own_data) if args.own_data else None
    splits = list(own) if own else ["train", "validation", "test"]
    print(f"Preparing '{name}' ({args.language}), splits: {', '.join(splits)}")
    print(f"Output: {out_dir}\n", flush=True)

    summary = {"name": name, "language": args.language,
               "source": "own data" if own else "google/fleurs", "splits": {}}
    t_start = time.time()
    for split in splits:
        t0 = time.time()
        source = read_own_jsonl(own[split]) if own else read_fleurs(args.language, split)
        counts = convert(source, split, args.language, out_dir)
        summary["splits"][split] = counts
        print(f"[{split}] {counts['kept']} recordings, {counts['hours']} h "
              f"({time.time() - t0:.0f} s)  skipped: empty text {counts['skipped_empty_text']}, "
              f"length {counts['skipped_length']}, unreadable {counts['skipped_unreadable']}", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {time.time() - t_start:.0f} s. Data is in {out_dir}")
    print(f"Use it in the other steps with:  --data {name}")


if __name__ == "__main__":
    main()
