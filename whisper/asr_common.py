"""
Shared helpers used by every script in this repository.

Everything here uses only packages that exist in the LUMI AI Factory
container (numpy, scipy, datasets, the ffmpeg binary), so no venv is needed.
In particular we avoid librosa, soundfile, torchcodec and jiwer, which are
not in the container.
"""
import io
import os
import re
import subprocess
import unicodedata
import wave

import numpy as np

SAMPLE_RATE = 16000

# One place that maps our language codes to what each dataset/model expects.
LANGUAGES = {
    "fi": {"fleurs": "fi_fi", "whisper": "finnish", "qwen": "Finnish"},
    "en": {"fleurs": "en_us", "whisper": "english", "qwen": "English"},
}


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def _to_float_mono(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        arr = arr.astype(np.float32) / 2147483648.0
    elif arr.dtype == np.uint8:
        arr = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr


def _ffmpeg_decode(data: bytes | None, path: str | None, sr: int) -> np.ndarray:
    """Decode any format ffmpeg understands to mono float32 at `sr`."""
    src = path if path is not None else "pipe:0"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src,
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    proc = subprocess.run(cmd, input=None if path else data,
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def decode_audio(data: bytes | None = None, path: str | None = None,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Return mono float32 audio at `sr` Hz from raw file bytes or a file path.

    Fast path: WAV that is already at the target rate is read with scipy.
    Everything else (other rates, mp3, flac, ...) goes through ffmpeg.
    """
    if data is None and path is None:
        raise ValueError("Give either data or path")
    try:
        from scipy.io import wavfile
        file_sr, arr = wavfile.read(io.BytesIO(data) if data is not None else path)
        if file_sr == sr:
            return _to_float_mono(arr)
    except Exception:
        pass  # not a plain WAV, or a WAV variant scipy can't read
    return _ffmpeg_decode(data, path, sr)


def encode_wav(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode mono float32 audio as 16-bit PCM WAV bytes (our storage format)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------
# Log output
# --------------------------------------------------------------------------
def print_setup(title: str, settings: dict) -> None:
    """Print the run's settings as one block at the top of the log."""
    if int(os.environ.get("RANK", "0")) != 0:   # multi-GPU: only the first process prints
        return
    width = max(len(k) for k in settings)
    print(f"========== {title}")
    for key, value in settings.items():
        print(f"{key:<{width}} : {value}")
    print("=" * (11 + len(title)), flush=True)


# --------------------------------------------------------------------------
# Text normalization and metrics
# --------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^\w\s']", flags=re.UNICODE)
_EDGE_APOSTROPHE = re.compile(r"(^|\s)'+|'+(\s|$)")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Scoring normalization, applied to BOTH reference and hypothesis:
    Unicode NFKC, lowercase, punctuation removed (letters incl. å/ä/ö, digits
    and in-word apostrophes kept), whitespace collapsed.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("_", " ")
    text = _NON_WORD.sub(" ", text)
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two sequences."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer_cer(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus-level WER and CER (total edits / total reference length)."""
    assert len(references) == len(hypotheses)
    w_err = w_tot = c_err = c_tot = 0
    for ref, hyp in zip(references, hypotheses):
        ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
        rw, hw = ref_n.split(), hyp_n.split()
        w_err += edit_distance(rw, hw)
        w_tot += len(rw)
        c_err += edit_distance(list(ref_n), list(hyp_n))
        c_tot += len(ref_n)
    return {
        "wer": w_err / max(w_tot, 1),
        "cer": c_err / max(c_tot, 1),
        "num_utterances": len(references),
        "num_ref_words": w_tot,
    }


# --------------------------------------------------------------------------
# Paths. The Slurm scripts set DATA_DIR and OUTPUT_DIR, so short names work:
#   --data fleurs-fi                    -> $DATA_DIR/fleurs-fi
#   --model whisper-large-v3-turbo-fi   -> $OUTPUT_DIR/models/whisper-large-v3-turbo-fi/final
# --------------------------------------------------------------------------
def data_root() -> str:
    return os.environ.get("DATA_DIR") or "data"


def output_root() -> str:
    return os.environ.get("OUTPUT_DIR") or "runs"


def models_root() -> str:
    return os.path.join(output_root(), "models")


def resolve_data_dir(data: str) -> str:
    """Accept a full path or a dataset name under $DATA_DIR."""
    for candidate in (data, os.path.join(data_root(), data)):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    available = sorted(os.listdir(data_root())) if os.path.isdir(data_root()) else []
    raise FileNotFoundError(f"Prepared data '{data}' not found (also looked in {data_root()}). "
                            f"Available: {available or 'none, run prepare_data.py first'}")


def resolve_model(model: str) -> str:
    """
    Accept a fine-tuned model name (under $OUTPUT_DIR/models), a local path, or
    a Hugging Face model id. A folder that contains "final" means its final model.
    """
    for candidate in (os.path.join(models_root(), model), model):
        if os.path.isdir(candidate):
            final = os.path.join(candidate, "final")
            return os.path.abspath(final if os.path.isdir(final) else candidate)
    return model  # Hugging Face id, e.g. openai/whisper-small


QUICK_SUFFIX = "-quick"


def quick_data(data: str) -> str:
    """--quick: use the '-quick' version of the data if it exists ('fleurs-fi' -> 'fleurs-fi-quick')."""
    if data.rstrip("/").endswith(QUICK_SUFFIX):
        return data
    candidate = data.rstrip("/") + QUICK_SUFFIX
    try:
        resolve_data_dir(candidate)
        return candidate
    except FileNotFoundError:
        print(f"Note: no '{candidate}' found (prepare it with --quick); using '{data}' "
              f"with quick limits instead.")
        return data


def quick_model(model: str) -> str:
    """--quick: use your '-quick' fine-tuned model if it exists ('whisper-small-fi' -> 'whisper-small-fi-quick')."""
    candidate = model.rstrip("/") + QUICK_SUFFIX
    if not model.rstrip("/").endswith(QUICK_SUFFIX) and os.path.isdir(os.path.join(models_root(), candidate)):
        return candidate
    return model


def model_label(model: str) -> str:
    """Short readable name for a model: 'openai/whisper-small' -> 'whisper-small',
    '.../models/whisper-small-fi/final' -> 'whisper-small-fi'."""
    parts = model.rstrip("/").split("/")
    if len(parts) > 1 and (parts[-1] == "final" or parts[-1].startswith("checkpoint-")):
        return parts[-2] if parts[-1] == "final" else f"{parts[-2]}-{parts[-1]}"
    return parts[-1]


def load_prepared_split(data: str, split: str):
    """Load a split written by prepare_data.py."""
    from datasets import load_from_disk
    root = resolve_data_dir(data)
    path = os.path.join(root, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No '{split}' split in {root} (has: {sorted(os.listdir(root))})")
    return load_from_disk(path)


def data_language(data: str) -> str:
    """The language code ('fi' or 'en') stored in prepared data."""
    ds = load_prepared_split(data, "test" if os.path.isdir(os.path.join(resolve_data_dir(data), "test"))
                             else "train")
    return ds[0]["language"]


# --------------------------------------------------------------------------
# Models. The family (Whisper, Qwen3-ASR, ...) is read from the model's own
# config.json, so users only ever give --model. Each family becomes a
# "transcriber": a function from a list of 16 kHz float32 arrays to a list of
# transcripts. compare.py and inference.py use these, so every model is run
# and scored the same way.
# --------------------------------------------------------------------------
FAMILIES = {"whisper": "whisper"}   # config.json model_type -> family


def detect_family(model: str) -> str:
    """Read model_type from config.json (local folder or Hugging Face Hub)."""
    import json
    config_file = os.path.join(model, "config.json")
    if not os.path.isfile(config_file):
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(model, "config.json")
    model_type = json.load(open(config_file)).get("model_type", "")
    if model_type not in FAMILIES:
        raise ValueError(f"Model '{model}' has model_type '{model_type}', which these scripts "
                         f"don't support (yet). Supported: {sorted(FAMILIES)}")
    return FAMILIES[model_type]


def _device_and_dtype():
    import torch
    if torch.cuda.is_available():  # ROCm GPUs also appear as "cuda" in PyTorch
        return 0, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return -1, torch.float32


def _load_whisper(model_path, language, batch_size, max_new_tokens):
    from transformers import pipeline
    device, dtype = _device_and_dtype()
    asr = pipeline("automatic-speech-recognition", model=model_path,
                   device=device, dtype=dtype, chunk_length_s=30)
    generate_kwargs = {"task": "transcribe", "max_new_tokens": max_new_tokens}
    if language:
        generate_kwargs["language"] = LANGUAGES[language]["whisper"]

    def transcribe(audios):
        inputs = [{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audios]
        outputs = asr(inputs, batch_size=batch_size, generate_kwargs=generate_kwargs)
        return [o["text"].strip() for o in outputs]

    return transcribe


def load_transcriber(model: str, language: str | None = None,
                     batch_size: int = 16, max_new_tokens: int = 225):
    """
    model: anything resolve_model() accepts. language=None lets the model
    detect the language. max_new_tokens caps runaway (hallucinating) outputs.
    Returns (transcribe_function, family).
    """
    model = resolve_model(model)
    family = detect_family(model)
    if family == "whisper":
        return _load_whisper(model, language, batch_size, max_new_tokens), family
    raise ValueError(f"No transcriber for family '{family}'")
