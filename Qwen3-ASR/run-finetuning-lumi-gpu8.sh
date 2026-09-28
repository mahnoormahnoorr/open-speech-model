#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=02:30:00

#Load the module
module purge
module load Local-LAIF lumi-aif-singularity-bindings

# export path to used container image
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif

#export PYTHONPATH="/scratch/project_462001302/mmahnoor/speech_models/venv/lib/python3.12/site-packages:$PYTHONPATH"
export PYTHONPATH="/scratch/project_462000131/hintsala/open-speech-model/Qwen3-ASR/venv/lib/python3.12/site-packages:$PYTHONPATH"


# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

# MIOpen cache must be on node-local /tmp (Lustre lacks the file locking
# MIOpen needs), in a directory of your own so it doesn't clash with other
# users' files. See: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/
# This works because the job runs on a single node.
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH


export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
 

srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "
  torchrun --standalone --nnodes=1 --nproc_per_node=\${SLURM_GPUS_PER_NODE} \
    finetuning.py \
      --model_path Qwen/Qwen3-ASR-1.7B \
      --train_file data/train.jsonl \
      --eval_file data/validation.jsonl \
      --output_dir outputs/qwen3-asr-1.7b-ft \
      --batch_size 8 \
      --grad_acc 4 \
      --lr 2e-5 \
      --epochs 3 \
      --save_steps 200 \
      --save_total_limit 5 \
      --language Finnish
"