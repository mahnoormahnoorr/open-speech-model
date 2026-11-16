# Qwen3-ASR

Fine-tune and evaluate [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR).
Prepare the data first (see the [main README](../README.md)), and submit all
jobs from the repository root. The steps and options are the same as for
[Whisper](../whisper/README.md).

## 1. Evaluate the ready model

```bash
sbatch qwen/evaluate.sh --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B --quick   # 32 sentences
sbatch qwen/evaluate.sh --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B
```

**Look at:** the WER/CER table at the end of the log. The results folder
(`$WORK/results/<name>/`) has `summary.md`, `side_by_side.tsv` and
`transcripts.jsonl`, in the same format as Whisper's.

## 2. Fine-tune

```bash
sbatch qwen/finetune_1gpu.sh --data fleurs-fi --quick     # 64 sentences, 30 steps
sbatch qwen/finetune_1gpu.sh --data fleurs-fi             # 3 epochs
sbatch qwen/finetune_8gpu.sh --data fleurs-fi --batch_size 4 --grad_acc 2
```

The model is saved as `$WORK/models/<name>/final`, e.g. `qwen3-asr-1.7b-fi`
(`qwen3-asr-1.7b-fi-quick` with `--quick`). Checkpoints are saved every
`--save_steps` steps next to it. `final` is the model after the last step.

With 8 GPUs, `--batch_size` is per GPU: the effective batch is
`batch_size x grad_acc x 8`. When you change the number of GPUs, adjust
`--grad_acc` to keep the effective batch the same.

**Look at:** `'loss'` (training) and `eval_loss` (validation), which should both
go down, and the last lines, which give your model's name.

Continue a stopped run with `--resume`. More options: see the top of `finetune.py`.

### The language tag

Qwen3-ASR expects every training transcript to start with a language tag:

```
language Finnish<asr_text>Hyönteiset olivat ensimmäisiä eläimiä.
```

The prepared data has plain transcripts (Whisper needs them that way).
`finetune.py` adds the tag from the data's `language` column, so you don't
need to do anything.

## 3. Compare the ready and fine-tuned model

```bash
sbatch qwen/evaluate.sh --data fleurs-fi --quick --models Qwen/Qwen3-ASR-1.7B qwen3-asr-1.7b-fi
sbatch qwen/evaluate.sh --data fleurs-fi --models Qwen/Qwen3-ASR-1.7B qwen3-asr-1.7b-fi
```

## 4. Transcribe your own audio

```bash
sbatch qwen/transcribe.sh --model qwen3-asr-1.7b-fi --language fi --audio /scratch/project_462001302/my_audio/
```

Leave out `--language` to let the model detect it; the detected language is
shown with each transcript.

## Files

| File | What it does |
|---|---|
| `finetune.py` | Fine-tuning, based on the Qwen team's example; `finetune_1gpu.sh` / `finetune_8gpu.sh` run it |
| `evaluate.py` | Scores one or more models on prepared data; `evaluate.sh` runs it |
| `transcribe.py` | Transcribes audio files; `transcribe.sh` runs it |
