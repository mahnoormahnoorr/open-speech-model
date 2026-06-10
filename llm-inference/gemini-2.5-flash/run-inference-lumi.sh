#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=1:45:00
#SBATCH --gpus-per-node=1

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers//lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif


export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

VENV=/scratch/project_462001302/mmahnoor/vllm/openai/venv

OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-data
mkdir -p $OUTPUT_DIR

export TOKENIZERS_PARALLELISM=false
export GOOGLE_API_KEY="YOUR_API_KEY"

set -xv

Audio_file=/scratch/project_462001302/mmahnoor/vllm/openai/input/tests_data_multilingual.mp3

srun singularity exec "$SIF" bash -c "
    source ${VENV}/bin/activate && \
    for TASK in summary transcribe timestamp tokens clip; do
        echo '=== Running task: '$TASK' ===' && \
        python3 run-gemini-audio.py \
            --audio ${Audio_file} \
            --model gemini-2.5-flash \
            --task \$TASK \
            --output-dir ${OUTPUT_DIR}
    done
"
