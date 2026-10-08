#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --job-name=asr-inference
#SBATCH --output=logs/%x-%j.out
#
# STEP 6: Transcribe your own audio (any format; long recordings are chunked).
#
#   sbatch run-inference-lumi.sh --model whisper-large-v3-turbo-fi --language fi \
#       --audio /scratch/project_462XXXXXX/my_audio/
#
# Transcripts are shown in the log; add --output_file transcripts.jsonl to also
# save all of them. Try it quickly with --quick (3 files or sentences).
# More options: see the top of inference.py

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

singularity exec -B $BINDS $SIF python inference.py "$@"