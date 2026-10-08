#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=whisper-finetune-gpu1
#SBATCH --output=logs/%x-%j.out
#
# STEP 4: Fine-tune Whisper on prepared data (1 GPU).
#
#   sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi
#   sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi --model openai/whisper-small
#
#   # Try it quickly: Whisper-small, first 64 sentences, 30 steps -> whisper-small-fi-quick
#   sbatch run-finetune-lumi-gpu1.sh --data fleurs-fi --quick
#
# The model is saved under the name <model>-<language>, e.g.
# whisper-large-v3-turbo-fi, which you then use with run-compare-lumi.sh and
# run-inference-lumi.sh. Training progress (loss, validation WER) is in the log.
# More options: see the top of finetune_whisper.py

# ---- Setup (the same in every script; you normally don't need to change it) ----
# The container that provides Python, PyTorch (ROCm) and Transformers:
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif
# Everything the scripts produce goes under one folder on /scratch:
WORK=/scratch/$SLURM_JOB_ACCOUNT/$USER/asr
export DATA_DIR=$WORK/data          # prepared datasets
export OUTPUT_DIR=$WORK/runs        # fine-tuned models, comparisons, quick tests
export HF_HOME=$WORK/hf-cache       # downloaded models and datasets
# Technical settings for LUMI's AMD GPUs and the container:
export MIOPEN_USER_DB_PATH=/tmp/$USER-miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH && mkdir -p $MIOPEN_USER_DB_PATH
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
BINDS=/pfs,/scratch/$SLURM_JOB_ACCOUNT,/projappl/$SLURM_JOB_ACCOUNT
# --------------------------------------------------------------------------------

# Print the job's setup at the top of the log
echo "========== Job"
echo "Job        : $SLURM_JOB_NAME (id $SLURM_JOB_ID)"
echo "Node       : $SLURMD_NODENAME, partition $SLURM_JOB_PARTITION, GPUs ${SLURM_GPUS_ON_NODE:-0}"
echo "Started    : $(date '+%Y-%m-%d %H:%M')"
echo "Container  : $(basename "$SIF")"
echo "Results in : $WORK"
echo "Arguments  : ${*:-(none)}"
echo "=========="

# Which Whisper model to fine-tune. Change it here, or give --model on the
# sbatch line (that wins). Smaller and faster: openai/whisper-small
MODEL=openai/whisper-large-v3-turbo
# With --quick the small model is used instead, so trying it out is fast.
if [[ " $* " == *" --quick "* ]]; then MODEL=openai/whisper-small; fi

singularity exec -B $BINDS $SIF python finetune_whisper.py --model $MODEL --num_workers 6 "$@"