import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to an audio file, e.g. sample.wav or sample.mp3",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language, e.g. english, finnish, arabic",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="transcribe = same language, translate = translate to English",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Audio file: {args.audio}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"ROCm/CUDA available: {n_gpus} GCD(s)")
        for i in range(n_gpus):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GCD {i}: {torch.cuda.get_device_name(i)} ({total:.1f} GB)")
        device = "cuda:0"
        torch_dtype = torch.bfloat16
    else:
        print("No GPU available, exiting")
        exit(1)

    processor = AutoProcessor.from_pretrained(args.model)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
    )

    model.to(device)
    model.eval()

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0,
    )

    generate_kwargs = {
        "task": args.task,
    }

    if args.language is not None:
        generate_kwargs["language"] = args.language

    result = asr(
        args.audio,
        chunk_length_s=30,
        batch_size=8,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )

    print("Transcription:")
    print(result["text"])

    if "chunks" in result:
        print("\nTimestamps:")
        for chunk in result["chunks"]:
            print(chunk)
