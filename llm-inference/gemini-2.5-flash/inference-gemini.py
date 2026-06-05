import argparse
import os
import sys
from google import genai
from google.genai import types
from pydub import AudioSegment


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file, e.g. tests_data_multilingual.mp3",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model ID",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["summary", "describe", "transcribe", "timestamp", "tokens", "clip"],
    )

    parser.add_argument(
        "--start-ms",
        type=int,
        default=0,
        help="Start time in milliseconds for clip mode",
    )

    parser.add_argument(
        "--duration-ms",
        type=int,
        default=10000,
        help="Clip duration in milliseconds for clip mode",
    )

    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set")
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Using model: {args.model}")
    print(f"Audio file: {args.audio}")
    print(f"Task: {args.task}")

    if args.task == "clip":
        sound = AudioSegment.from_file(args.audio)
        clip = sound[args.start_ms : args.start_ms + args.duration_ms]

        audio_bytes = clip.export(format="mp3").read()

        response = client.models.generate_content(
            model=args.model,
            contents=[
                "Describe this audio clip.",
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/mp3",
                ),
            ],
        )

        print(response.text)
        return

    uploaded_file = client.files.upload(file=args.audio)

    if args.task == "summary":
        prompt = "Listen carefully to the following audio file. Provide a brief summary."

    elif args.task == "describe":
        prompt = "Describe this audio file."

    elif args.task == "transcribe":
        prompt = "Generate a transcript of the speech."

    elif args.task == "timestamp":
        prompt = "Provide a transcript of the speech between the timestamps 02:30 and 03:29."

    elif args.task == "tokens":
        count_tokens_response = client.models.count_tokens(
            model=args.model,
            contents=[uploaded_file],
        )
        print("Audio file tokens:", count_tokens_response.total_tokens)
        return

    response = client.models.generate_content(
        model=args.model,
        contents=[
            prompt,
            uploaded_file,
        ],
    )

    print(response.text)


if __name__ == "__main__":
    main()
