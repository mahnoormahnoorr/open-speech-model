#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00

#Load the module
module purge
module load Local-LAIF lumi-aif-singularity-bindings

# export path to used container image
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif

export PYTHONPATH="/scratch/project_462001302/mmahnoor/speech_models/venv/lib/python3.12/site-packages:$PYTHONPATH"

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "
  python3 infer_qwen3_asr.py \
    --checkpoint outputs/qwen3-asr-1.7b-ft/checkpoint-33 \
    --eval_file data/validation.jsonl \
    --num_examples 5
"
