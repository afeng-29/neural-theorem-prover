#!/bin/bash
#SBATCH --job-name=finetune_calculus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/project/dachxiu/afeng/prover/logs/finetune_calculus_%j.log
#SBATCH --error=/project/dachxiu/afeng/prover/logs/finetune_calculus_%j.log
#SBATCH --account=pi-dachxiu
#SBATCH --chdir=/project/dachxiu/afeng/prover

module load python/3.11.9
module load cuda/12.1

source /project/dachxiu/afeng/prover/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python training/finetune.py \
    --train-data data/calculus/train.jsonl \
    --val-data   data/calculus/val.jsonl \
    --test-data  data/calculus/test.jsonl \
    --base-model models/pretrained/leandojo-lean4-tacgen-byt5-small \
    --output-dir models/finetuned/calculus/ \
    --epochs 10 --batch-size 4 --grad-accum 4 --max-input-length 512 --resume
