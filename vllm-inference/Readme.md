
# Audio Transcription Using vLLM on LUMI

This example runs the [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) speech-to-text model with vLLM on LUMI.

The vLLM server is launched as a Slurm batch job and exposes an OpenAI-compatible audio transcription endpoint.

## Files

```text
run-vllm-lumi4.sh      # Slurm batch script for starting the vLLM server
run-vllm-process.sh    # Script that runs vllm serve
```

Submit the slurm job with: 

sbatch run-vllm-lumi4.sh

When the job is running, open an interactive shell on compute node and send an audio fle to the vLLM transcription endpoints:

```
srun --overlap --jobid <slurm-job-id> --pty bash

curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer EMPTY" \
  -F "file=@/scratch/project_462001302/mmahnoor/vllm/speech-to-text/input/tests_data_multilingual.wav;type=audio/wav" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=text"
```

The output will be the transcribed text from the audio file.

## Notes:

Make sure the audio dependencies are available in the container. If vLLM returns `Invalid or unsupported audio file`, check whether packages such as `soundfile` and `librosa` are installed.



