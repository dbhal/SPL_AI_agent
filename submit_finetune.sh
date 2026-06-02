#!/bin/bash
#SBATCH --job-name=spl-llm-finetune
#SBATCH --partition=v100_normal_q          # GPU partition on TinkerCliffs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                       # request 1 GPU
#SBATCH --mem=40G
#SBATCH --time=04:00:00                    # 4 hours max
#SBATCH --output=logs/finetune_%j.out      # %j = job ID
#SBATCH --error=logs/finetune_%j.err
#SBATCH --account=your_allocation_id      # replace with your VT ARC allocation

# ── Environment ───────────────────────────────────────────────────────────────
module load Python/3.11
module load CUDA/12.1

# Activate your virtual environment
source ~/venv/spl_llm/bin/activate

# ── Run fine-tuning script ────────────────────────────────────────────────────
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python rag/finetune_qlora.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "data/spl_finetune_dataset.json" \
    --output_dir "models/spl-llama3-qlora" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4

echo "Job finished at: $(date)"
