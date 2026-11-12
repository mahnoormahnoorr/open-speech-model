#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=01:30:00

#Load the module
module purge
module load Local-LAIF lumi-aif-singularity-bindings

# export path to used container image
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif


export PYTHONPATH="/scratch/project_462001302/mmahnoor/test-osm/venv/lib/python3.12/site-packages:$PYTHONPATH"
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

#set MIOPEN temp folder
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"

mkdir -p "$TORCH_HOME" "$HF_HOME"

srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH $SIF bash -c 'torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  qwenmodel.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file data/train.jsonl \
  --eval_file data/validation.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 4 \
  --grad_acc 8 \
  --lr 2e-5 \
  --epochs 3 \
  --save_steps 20'

