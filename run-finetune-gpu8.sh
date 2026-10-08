#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --job-name=whisper-finetune-gpu8
#SBATCH --output=logs/%x-%j.out
#
# Fine-tune Whisper on one full node (8 GPUs). Same as run-finetune-lumi-gpu1.sh,
# but faster for large datasets.
#
#   sbatch run-finetune-lumi-gpu8.sh --data fleurs-fi --batch_size 4
#
# The effective batch size is batch_size x grad_acc x 8. For small datasets
# like FLEURS (~2,700 sentences) keep it low (e.g. --batch_size 4 -> 32),
# otherwise there are too few training steps.
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

singularity exec -B $BINDS $SIF python -m torch.distributed.run --standalone --nproc_per_node=8 \
    finetune_whisper.py --model $MODEL --num_workers 6 "$@"