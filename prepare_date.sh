#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=prepare-data
#SBATCH --output=logs/%x-%j.out
#
# Download data and convert it for Whisper, Qwen and AITTA. Run once per language.
# Runs on a CPU node.
#
#     sbatch prepare_data.sh --language fi      # -> data name: fleurs-fi
#     sbatch prepare_data.sh --language en      # -> data name: fleurs-en
#
# More options (e.g. your own data): see the top of prepare_data.py

# ---- Setup: the same in every job script ---------------------------------------
# Submit every job from the repository root folder, e.g. "sbatch prepare_data.sh"
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

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURMD_NODENAME, $(date '+%Y-%m-%d %H:%M')"
echo "Arguments: ${*:-(none)}"
echo "Output folder: $WORK"
echo "Container: $(basename $SIF)"

# ---- Python environment: the venv created in README.md, Step 0 -------------------
# Its packages go first on PYTHONPATH; everything else comes from the container.
[ -d venv/lib/python3.12/site-packages ] || { echo "ERROR: no venv/ in the repository root. Create it first: README.md, Step 0"; exit 1; }
export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PYTHONPATH"

# srun gives the program all the CPUs of the job (otherwise only 1)
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# ---- Run ----------------------------------------------------------------------------
# $* = the options given after the script name in sbatch
srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "
python prepare_data.py $*"
