#!/bin/bash
#SBATCH --account=project_462001302
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif



AUDIO_FILE=//scratch/project_462001302/mmahnoor/vllm/openai/input/tests_data_multilingual.mp3
OUTPUT_JSON=/scratch/project_462001302/mmahnoor/vllm/openai/diarized_output.json

set -xv

srun singularity exec "$SIF" python inference-openai-diarize.py \
  --model gpt-4o-transcribe-diarize \
  --audio "$AUDIO_FILE" \
  --output-json "$OUTPUT_JSON"
