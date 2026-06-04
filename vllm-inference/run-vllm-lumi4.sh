#!/bin/bash
#SBATCH -A project_462001302
#SBATCH -p dev-g
#SBATCH --time 1:00:00
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1
#SBATCH --mem 240G

# Set MIOPEN temp folder to avoid collisions with other users on the same node
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

export PYTHONPATH=/scratch/project_462001302/mmahnoor/vllm/python-packages:$PYTHONPATH



srun singularity exec "$CONTAINER_IMAGE" bash -lc '
export PYTHONPATH=/scratch/project_462001302/mmahnoor/vllm/python-packages:$PYTHONPATH
bash ./run-vllm-process.sh openai/whisper-large-v3
'
