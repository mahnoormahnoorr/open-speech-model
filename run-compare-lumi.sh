#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --job-name=asr-compare
#SBATCH --output=logs/%x-%j.out
#
# STEP 3 and 5: Run one or more models on the same test data and compare them.
# Produces a summary (WER, CER, speed) and side_by_side.tsv with every
# sentence: the correct text and each model's transcript.
#
#   # Two ready models on English (Qwen3-ASR support arrives in the next update)
#   sbatch run-compare-lumi.sh --data fleurs-en \
#       --models openai/whisper-large-v3-turbo Qwen/Qwen3-ASR-1.7B
#
#   # Ready model vs your fine-tuned version on Finnish
#   sbatch run-compare-lumi.sh --data fleurs-fi \
#       --models openai/whisper-large-v3-turbo whisper-large-v3-turbo-fi
#
#   # Try it quickly: first 32 sentences, your '-quick' fine-tuned models
#   sbatch run-compare-lumi.sh --data fleurs-fi \
#       --models openai/whisper-small whisper-small-fi --quick
#
# Results: /scratch/<project>/$USER/asr/runs/comparisons/<name>/
# More options: see the top of compare.py

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

singularity exec -B $BINDS $SIF python compare.py "$@"