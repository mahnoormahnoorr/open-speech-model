"""
Load Finnish ASR data (google/fleurs, config 'fi_fi') and convert it into
the audio+text jsonl format the fine-tuning script expects.

NOTE: Mozilla Common Voice is no longer distributed via Hugging Face as of
October 2025 -- it moved exclusively to Mozilla Data Collective
(https://datacollective.mozillafoundation.org). This script now targets
FLEURS instead, which remains natively hosted on HF. If you specifically
need real Common Voice data later, download it from Mozilla Data Collective
directly and point this script's REPO/LANG/loading logic at the local files
instead of a HF repo id.

Requires: HF_TOKEN env var set (harmless if the target dataset isn't gated,
but needed if you swap in a gated repo later).

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    python prepare_common_voice_fi.py --split train --out_dir data
"""
import argparse
import io
import json
import os

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import HfApi

REPO = "google/fleurs"
LANG = "fi_fi"  # FLEURS uses locale-style config names, not bare ISO codes


def find_parquet_files(repo: str, lang: str, split: str) -> list[str]:
    """List every parquet shard for a given language/split in the repo."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(repo, repo_type="dataset")
    matches = [
        f for f in files
        if f.startswith(f"{lang}/{split}/") and f.endswith(".parquet")
    ]
    if not matches:
        available = sorted({f.split("/")[1] for f in files if f.startswith(f"{lang}/")})
        raise RuntimeError(
            f"No parquet files found for {lang}/{split} in {repo}. "
            f"Available splits for '{lang}': {available}"
        )
    return [f"https://huggingface.co/datasets/{repo}/resolve/main/{f}" for f in matches]


TEXT_FIELD_CANDIDATES = ["sentence", "raw_transcription", "transcription", "text"]


def resolve_text_field(ds) -> str:
    for cand in TEXT_FIELD_CANDIDATES:
        if cand in ds.column_names:
            return cand
    raise RuntimeError(
        f"Could not find a transcript column among {TEXT_FIELD_CANDIDATES}. "
        f"Available columns: {ds.column_names}"
    )


def build_jsonl(split: str, out_dir: str, prompt: str, limit: int | None = None):
    hf_token = os.environ.get("HF_TOKEN")

    try:
        # Works out of the box for most datasets (e.g. google/fleurs) that
        # haven't been affected by any script-loading removal or gating changes.
        ds = load_dataset(REPO, LANG, split=split, token=hf_token)
        print(f"[{split}] loaded via standard load_dataset()")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        print(f"[{split}] standard loader blocked, falling back to direct Parquet discovery")
        parquet_urls = find_parquet_files(REPO, LANG, split)
        print(f"[{split}] found {len(parquet_urls)} parquet shard(s):")
        for u in parquet_urls:
            print(f"  {u}")
        ds = load_dataset("parquet", data_files={split: parquet_urls}, split=split, token=hf_token)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[{split}] loaded {len(ds)} examples")

    # Avoid the new torchcodec requirement for audio auto-decoding entirely --
    # get raw bytes instead and decode them ourselves with soundfile (pure
    # CPU, no GPU/CUDA/ROCm-specific wheel involved).
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    text_field = resolve_text_field(ds)
    print(f"[{split}] using '{text_field}' as the transcript column")

    audio_dir = os.path.join(out_dir, "audio", split)
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{split}.jsonl")

    n_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            text = (ex.get(text_field) or "").strip()
            if not text:
                continue  # skip empty transcripts

            audio = ex["audio"]  # {"bytes": b"...", "path": "..."} since decode=False
            audio_bytes = audio.get("bytes")
            if audio_bytes is not None:
                array, sr = sf.read(io.BytesIO(audio_bytes))
            else:
                array, sr = sf.read(audio["path"])

            wav_path = os.path.join(audio_dir, f"{i}.wav")
            sf.write(wav_path, array, sr)

            f.write(json.dumps({
                "audio": wav_path,
                "text": text,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[{split}] wrote {n_written} examples to {jsonl_path}")
    return jsonl_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--prompt", type=str, default="Transcribe the following Finnish audio.")
    p.add_argument("--limit", type=int, default=None, help="Optional cap for a quick smoke test")
    args = p.parse_args()

    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN is not set. If the dataset is gated, this will fail with a 401.")

    os.makedirs(args.out_dir, exist_ok=True)
    build_jsonl(args.split, args.out_dir, args.prompt, args.limit)


if __name__ == "__main__":
    main()
