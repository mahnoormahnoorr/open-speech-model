# ASR/Speech models on LUMI

**WORK IN PROGRESS!**

Example scripts for fine-tuning and evaluating open-source speech-to-text models
on LUMI, for Finnish and English:

| Folder | What it does |
|---|---|
| [`whisper/`](whisper/README.md) | Fine-tune and evaluate OpenAI Whisper |
| [`qwen/`](qwen/README.md) | Fine-tune and evaluate Qwen3-ASR |
| [`aitta/`](aitta/README.md) | Transcribe through CSC's AITTA API (no GPU of your own) |

All three use the same prepared data, and their results have the same format,
so they can be compared directly.

## How it works

- **Everything runs as Slurm jobs** in the LUMI AI Factory container.
- **Submit every job from this folder** (the repository root), e.g.
  `sbatch whisper/evaluate.sh ...`. A job submitted from anywhere else stops
  with an error. Logs go to `logs/<job name>-<job id>.out`.
- **One set of extra Python packages for everything**: `venv/` in this folder,
  created once from the terminal (Step 0). It adds the few packages listed in
  `requirements.txt` on top of the container. Every job script uses it.
- **Everything the jobs produce goes to one folder** on `/scratch`:

```
/scratch/project_462001302/asr-data/
├── data/       prepared datasets, e.g. data/fleurs-fi
├── models/     fine-tuned models, e.g. models/whisper-small-fi/final
├── results/    evaluations and transcripts, e.g. results/fleurs-fi-test-whisper-small/
└── hf-cache/   models and datasets downloaded from Hugging Face
```

The project number (`project_462001302`) is written in each job script: in the
`#SBATCH --account` line and in `WORK=...`.

## Step 0: Set up (once)

**1. Get the repository** into your project's `/scratch` ):

```bash
mkdir -p /scratch/project_462001302/$USER
cd /scratch/project_462001302/$USER
git clone <repository URL> open-speech-model
cd open-speech-model
```


**2. Create the Python environment.** Run these in the repository folder, on
the login node. It takes a couple of minutes.

```bash
cd /projappl/project_462001302/$USER/open-speech-model    # the repository root
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif

# Open an interactive shell within a container,
singularity shell "$SIF" # Now inside the container (prompt: Apptainer>)

# Create a venv that also sees all the container's packages (PyTorch for ROCm etc.)
Apptainer> python -m venv venv --system-site-packages
Apptainer> source venv/bin/activate
# The extra packages
(venv) Apptainer> pip install --no-cache-dir -r requirements.txt
# qwen-asr without its demo dependencies (gradio etc.; see requirements.txt)
(venv) Apptainer> pip install --no-cache-dir --no-deps qwen-asr==0.0.6
# Exit shell
(venv) Apptainer> exit
```

**How the jobs use it:** at the end of every job script you can see it:

```bash
export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PYTHONPATH"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "
python whisper/evaluate.py $*"
```

The jobs run the container's own Python and put the venv's packages first on
`PYTHONPATH`. So the packages installed here are used first, and everything
else (PyTorch for ROCm etc.) comes from the container. `$PWD` is the repository
root, because jobs are submitted from there. `$*` is replaced by the options you
give after the script name in `sbatch`, e.g. `--data fleurs-fi --quick`
(option values can't contain spaces).

`SIF` must be the same container as in the job scripts. If you change the
container in the scripts, create the venv again: delete the `venv/` folder and
repeat this step.


## Step 1: Prepare the data (once)

```bash
sbatch prepare_data.sh --language fi      # -> asr-data/fleurs-fi
sbatch prepare_data.sh --language en      # -> asr-data/fleurs-en
```

This downloads [FLEURS](https://huggingface.co/datasets/google/fleurs) (read
Wikipedia sentences, about 10 hours per language) and saves it in the format
all other steps read. It runs on a CPU node and takes a few minutes.

**Look at:** the end of the log shows how many recordings each split has
(about 2,700 train, 400 validation and 900 test for Finnish) and the data name
to use in the next steps.

**Your own data:** see the top of `prepare_data.py`.

## Step 2 onwards

Continue in [`whisper/`](whisper/README.md), [`qwen/`](qwen/README.md) or
[`aitta/`](aitta/README.md). The Whisper and Qwen folders have the same scripts:

| Script | What it does |
|---|---|
| `finetune_1gpu.sh` | Fine-tune on 1 GPU |
| `finetune_8gpu.sh` | Fine-tune on a full node (8 GPUs) |
| `evaluate.sh` | Score one or more models on the test set (WER, CER) |
| `transcribe.sh` | Transcribe your own audio files |

Each step can first be tried with `--quick` (a small part of the data, finishes
in minutes). Quick results never overwrite full ones: their names end in `-quick`.

## Following your jobs

```bash
squeue --me                              # your queued and running jobs
tail -f logs/<job-name>-<job-id>.out     # follow a log (Ctrl+C to stop)
scancel <job-id>                         # cancel a job
```

## How scoring works

WER (word error rate) and CER (character error rate) are computed with
[jiwer](https://github.com/jitsi/jiwer) after lowercasing and removing
punctuation from both the correct text and the transcript. CER matters for
Finnish, where one wrong letter in a long compound word makes the whole word
wrong in WER.


Developed with the help of Claude (Anthropic). Reviewed and tested by the authors.