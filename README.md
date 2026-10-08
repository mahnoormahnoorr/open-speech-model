# Speeck-to-text / ASR models on LUMI

**WORK IN PROGRESS!**

# Speech-to-text on LUMI: compare and fine-tune

This repository shows how to

- **compare ready-made speech-to-text models** on English, by WER and by reading
  their transcripts side by side, and
- **fine-tune models on Finnish** and compare each one with its original version.

ATM everything runs as Slurm jobs in the LUMI AI Factory container. Each step is a
single `sbatch` command. Supported models: **Whisper** (e.g. `openai/whisper-small`,
`openai/whisper-large-v3-turbo`).

## Setup (once)

1. Unpack the repository, for example under your project's application directory,
   and go into it:
   ```bash
   cd /projappl/project_462XXXXXX/$USER/<repo>
   ```
2. Put your project number into the slurm scripts. It appears in exactly one line of
   each script (`#SBATCH --account=...`).
   ```

Always submit jobs from this folder. Each job writes its log to `logs/`.

## How the steps work

Every step after data preparation can be run in two ways:

- **Quick** (add `--quick`): uses only part of the data and, for fine-tuning,
  only 30 training steps. Finishes in a few minutes and produces exactly the same
  kind of output as a full run, so you see what the step does and what to look at.
- **Full** (without `--quick`): the real run on all the data.

Do each step quickly first, then run it in full. Quick results never overwrite
full ones: quick models and comparison folders get `-quick` in their names.


Every log starts with the job's setup (node, GPUs, arguments) followed by the
settings the step actually uses (model, data, sizes, output folder), so you can
always check afterwards what a job did.

## Step 1: Prepare the data

```bash
sbatch run-prepare-data-lumi.sh --language en     # -> data name: fleurs-en
sbatch run-prepare-data-lumi.sh --language fi     # -> data name: fleurs-fi
```

Downloads the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset
(read Wikipedia sentences, about 10 hours per language) and converts it to the
format all the other steps use. This is always done for the full dataset, takes
a few minutes, and only needs to be done once; the later steps' `--quick` uses
just part of it.

It runs on a CPU node, debug-partition.

**Look at:** the `Data preparation finished` summary at the end of the log:
sentences, hours of audio, size and time per split, how many sentences were
skipped and why, and the data name to use in the next steps.

## Step 2: Compare ready models on English

**Quick** (first 32 test sentences):
```bash
sbatch run-compare-lumi.sh --data fleurs-en --quick \
    --models openai/whisper-small openai/whisper-large-v3-turbo
```

**Full** (all test sentences):
```bash
sbatch run-compare-lumi.sh --data fleurs-en \
    --models openai/whisper-small openai/whisper-large-v3-turbo
```

Every model transcribes the same test sentences; no training is involved.

**Look at:** the WER table and the sentences where the models disagree most,
both in the log. The results folder (printed at the end of the log) contains:

- `summary.md`: WER, CER and speed per model
- `side_by_side.tsv`: one row per sentence with the correct text, each model's
  transcript and each model's WER for that sentence. Open it in Excel or
  LibreOffice and sort by the WER columns.

## Step 3: Fine-tune on Finnish


### Using 1 GPU
**Quick** (Whisper-small, first 64 training sentences, 30 steps):
```bash
sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi --quick
```

**Full** (all training sentences, 5 epochs):
```bash
sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi --model openai/whisper-small

```

Without `--model`, the full run trains `openai/whisper-large-v3-turbo` (set in
the script), which gives better results but takes longer. The version with the
best validation WER is kept.

### Multi-GPU training

`run-finetune-lumi-gpu8.sh` trains on a full node

```bash
sbatch run-finetune-lumi-gpu8.sh --data fleurs-fi --model openai/whisper-large-v3-turbo
```

**Look at:**
- the `Fine-tuning setup` block: model, amount of training data, training length
- lines with `'loss'` (training progress) and `eval_wer` (validation score,
  which should go down in a full run)
- the last line, which gives your model's name: `whisper-small-fi-quick`
  (quick) or `whisper-small-fi` (full)

**Optional: check that training really learns.** This trains and tests on the
same 32 sentences, so a working setup memorizes them:

```bash
sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi --model openai/whisper-small \
    --overfit 32 --max_steps 100 --eval_steps 25 --learning_rate 1e-4 --name overfit-check
sbatch run-compare-lumi.sh --data fleurs-fi --split train --max_samples 32 \
    --models openai/whisper-small overfit-check
```

In the comparison, `overfit-check` should have a WER far below the ready model,
ideally near 0. If it doesn't, something is wrong; don't start long runs before
this works.

## Step 4: Compare the ready and fine-tuned model

**Quick** (first 32 test sentences, your quick model):
```bash
sbatch run-compare-lumi.sh --data fleurs-fi --quick \
    --models openai/whisper-small whisper-small-fi
```

**Full** (all test sentences, your full model):
```bash
sbatch run-compare-lumi.sh --data fleurs-fi \
    --models openai/whisper-small whisper-small-fi
```

With `--quick`, `whisper-small-fi` automatically means your quick model
`whisper-small-fi-quick`, so the commands are otherwise identical. Don't expect
better WER from the quick model after only 30 steps; the point is to see how the
comparison works.

**Look at:** the same outputs as in step 2. `side_by_side.tsv` now shows, for
every sentence, the correct text, the ready model's transcript and the
fine-tuned model's transcript, so you can see exactly which sentences improved.

## Step 5: Transcribe audio

**Quick** (3 test sentences):
```bash
sbatch run-inference-lumi.sh --model whisper-small-fi --data fleurs-fi --quick
```

**Your own audio:**
```bash
sbatch run-inference-lumi.sh --model whisper-small-fi --language fi \
    --audio /scratch/project_462XXXXXX/my_audio/ --output_file my_transcripts.jsonl
```

Any audio format works, and long recordings are split automatically. With
`--quick` only the first 3 files are transcribed.

**Look at:** the transcripts in the log. With prepared data each one is shown
with the correct text (`REF`) and the model's output (`HYP`). With
`--output_file`, all transcripts are saved to the file and the log shows the
first 10. Run it again with `--model openai/whisper-small` to hear what
fine-tuning changed.

## Following your jobs

```bash
squeue --me                              # your queued and running jobs
tail -f logs/<job-name>-<job-id>.out     # follow a job's log (Ctrl+C to stop)
scancel <job-id>                         # cancel a job
sacct -j <job-id> -X --format=JobID,JobName%25,State,Elapsed,ExitCode   # how long it took
```

## Where things go

| What | Where |
|---|---|
| Scripts and logs | this folder |
| Prepared data | `/scratch/<project>/$USER/asr/data/<data name>` |
| Fine-tuned models | `/scratch/<project>/$USER/asr/runs/models/<model name>/` (`final/` is the best one) |
| Comparisons | `/scratch/<project>/$USER/asr/runs/comparisons/<name>/` |
| Downloaded models and data | `/scratch/<project>/$USER/asr/hf-cache/` |

## Files

| File | What it does |
|---|---|
| `run-prepare-data-lumi.sh` | Step 1: Slurm job for data preparation |
| `run-compare-lumi.sh` | Steps 2 and 4: Slurm job for comparing models |
| `run-finetune-lumi-gpu1.sh` | Step 3: Slurm job for fine-tuning on 1 GPU |
| `run-finetune-lumi-gpu8.sh` | Step 3: Fine-tuning on a full node (8 GPUs)|
| `run-inference-lumi.sh` | Step 5: Slurm job for transcribing audio |
| `prepare_data.py` | Downloads and converts data |
| `compare.py` | Runs and scores one or more models; writes the summary and side-by-side file |
| `finetune_whisper.py` | Whisper fine-tuning |
| `inference.py` | Transcribes audio |
| `asr_common.py` | Shared code: audio, text normalization, WER, model loading |

Every Python file explains its options at the top. Any option can be added to
the `sbatch` command, e.g. `--epochs 3` or `--max_samples 100`.


**How scoring works.** WER (word error rate) and CER (character error rate) are
computed after lowercasing and removing punctuation from both the correct text and
the transcript. CER matters for Finnish because long compound words make WER look
worse.

**Technical notes.** The container lacks `librosa`, `soundfile`, `torchcodec` and
`jiwer`; audio is decoded with scipy/ffmpeg and WER is computed in
`asr_common.py`. Models are loaded in fp32 and trained with bf16 mixed precision.