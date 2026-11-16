#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --job-name=whisper-finetune-8gpu
#SBATCH --output=logs/%x-%j.out
#
# Fine-tune Whisper on a full node (8 GPUs). Same options as finetune_1gpu.sh.
#
#     sbatch whisper/finetune_8gpu.sh --data fleurs-fi
#
# --batch_size is per GPU: the effective batch is batch_size x grad_acc x 8.

# ---- Setup: the same in every job script ---------------------------------------
# Submit every job from the repository root folder, e.g. "sbatch whisper/finetune_8gpu.sh"
[ -f requirements.txt ] || { echo "ERROR: submit this job from the repository root folder"; exit 1; }

module purge
module load Local-LAIF lumi-aif-singularity-bindings
# The container: Python, PyTorch (ROCm), Transformers, datasets ...
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif

# Everything the jobs produce goes here: data/, models/, results/, hf-cache/
export WORK=/scratch/$SLURM_JOB_ACCOUNT/asr-data
export HF_HOME=$WORK/hf-cache
mkdir -p $WORK/data $WORK/models $WORK/results $HF_HOME

# Settings for LUMI's AMD GPUs
export MIOPEN_USER_DB_PATH=/tmp/$USER-miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1

# Python: the container's own Python, with the extra packages from venv/
# (README.md, Step 0) first on PYTHONPATH. Jobs run in the repository root.
VENV_PACKAGES=$PWD/venv/lib/python3.12/site-packages
[ -d "$VENV_PACKAGES" ] || { echo "ERROR: $VENV_PACKAGES not found. Create the venv first: README.md, Step 0"; exit 1; }
PYTHON="singularity exec --env PYTHONPATH=$VENV_PACKAGES:\$PYTHONPATH $SIF python"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURMD_NODENAME, $(date '+%Y-%m-%d %H:%M')"
echo "Arguments: ${*:-(none)}"
echo "Output folder: $WORK"
echo "Container: $(basename $SIF)"
# -----------------------------------------------------------------------------------

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 whisper/finetune.py "$@"
