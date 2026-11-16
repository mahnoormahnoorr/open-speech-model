# Whisper

Fine-tune and evaluate [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
models. Prepare the data first (see the [main README](../README.md)), and submit
all jobs from the repository root.

## 1. Evaluate a ready model

```bash
sbatch whisper/evaluate.sh --data fleurs-fi --models openai/whisper-small --quick   # 32 sentences
sbatch whisper/evaluate.sh --data fleurs-fi --models openai/whisper-small openai/whisper-large-v3-turbo
```

Every model transcribes the same test sentences. With several models this is a
comparison.

**Look at:** the WER/CER table at the end of the log. The results folder
(`$WORK/results/<name>/`, printed in the log) contains:

- `summary.md`: WER, CER and speed per model
- `side_by_side.tsv`: one row per sentence with the correct text, each model's
  transcript and WER. Open it in Excel or LibreOffice and sort by WER.
- `transcripts.jsonl`: all transcripts

## 2. Fine-tune

```bash
sbatch whisper/finetune_1gpu.sh --data fleurs-fi --quick     # whisper-small, 64 sentences, 30 steps
sbatch whisper/finetune_1gpu.sh --data fleurs-fi             # whisper-large-v3-turbo, 5 epochs
sbatch whisper/finetune_1gpu.sh --data fleurs-fi --model openai/whisper-small
sbatch whisper/finetune_8gpu.sh --data fleurs-fi             # a full node
sbatch whisper/finetune_1gpu.sh --data fleurs-fi --model openai/whisper-large-v3 --quick --batch_size 8 #adjusted batch_size to fit into memory
```

The model is saved as `$WORK/models/<name>/final`, where the name is
`<model>-<language>`, e.g. `whisper-small-fi` (`whisper-small-fi-quick` with
`--quick`). The checkpoint with the best validation WER becomes `final`.

With 8 GPUs, `--batch_size` is per GPU, so the effective batch is 8 times larger.
FLEURS is small (about 2,700 sentences), so keep it low there, e.g. `--batch_size 4`.

**Look at:** lines with `'loss'` (should go down) and `eval_wer` (validation WER,
should go down in a full run), and the last lines, which give your model's name.

Continue a stopped run with `--resume`. More options: see the top of `finetune.py`.

## 3. Compare the ready and fine-tuned model

```bash
sbatch whisper/evaluate.sh --data fleurs-fi --quick --models openai/whisper-small whisper-small-fi
sbatch whisper/evaluate.sh --data fleurs-fi --models openai/whisper-small whisper-small-fi
```

With `--quick`, `whisper-small-fi` automatically means your quick model
`whisper-small-fi-quick`, so the two commands are otherwise identical. Don't
expect a better WER after only 30 quick steps; the point is to see how it works.

## 4. Transcribe your own audio

```bash
sbatch whisper/transcribe.sh --model whisper-small-fi --language fi --audio /scratch/project_462001302/my_audio/
```

Any audio format works, and long recordings are split into 30-second pieces
automatically. Leave out `--language` to let the model detect it. Transcripts
are shown in the log and saved in `$WORK/results/<model>-transcribe/transcripts.jsonl`.

## Files

| File | What it does |
|---|---|
| `finetune.py` | Fine-tuning; `finetune_1gpu.sh` / `finetune_8gpu.sh` run it |
| `evaluate.py` | Scores one or more models on prepared data; `evaluate.sh` runs it |
| `transcribe.py` | Transcribes audio files; `transcribe.sh` runs it |
