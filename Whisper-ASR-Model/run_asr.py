
from datasets import load_dataset
from transformers import pipeline
import librosa
import numpy as np

MODEL_ID = "openai/whisper-small"

dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="test",
    streaming=False
)

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device="cpu"
)

for i, sample in enumerate(dataset):
    audio_path = sample["file"]

    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    audio_array = np.asarray(audio_array, dtype=np.float32)

    result = asr({
        "array": audio_array,
        "sampling_rate": 16000
    })

    print("Reference:", sample.get("text", "N/A"))
    print("Prediction:", result["text"])
    print("-" * 80)

    if i == 4:
        break
