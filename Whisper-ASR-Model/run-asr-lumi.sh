#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load the module
module purge
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings


export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
mkdir -p /scratch/project_462000131/mmahnoor/tmp
export TMPDIR=/scratch/project_462000131/mmahnoor/tmp


# Activate the virtual environment from your current directory or change to the appropriate path
VENV=/scratch/project_462001302/mmahnoor/espnet2/venv

# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

srun singularity exec "$SIF" bash -c "
    source ${VENV}/bin/activate && \
    python3 run_asr.py
"
