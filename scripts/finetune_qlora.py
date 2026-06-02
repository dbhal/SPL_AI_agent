"""
SPL Llama 3 Fine-Tuning with QLoRA
=====================================
Fine-tunes Llama 3 8B on SPL-specific Q&A data using QLoRA.
Designed to run on VT ARC TinkerCliffs GPU (A100 80GB).

Install:
    pip install transformers peft bitsandbytes datasets accelerate trl

Submit on VT ARC:
    sbatch rag/submit_finetune.sh
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_path",   default="data/spl_finetune_dataset.json")
    p.add_argument("--output_dir",  default="models/spl-llama3-qlora")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()


# ── Dataset format ────────────────────────────────────────────────────────────

def load_dataset(path: str) -> Dataset:
    """
    Load fine-tuning dataset.
    Expected format in spl_finetune_dataset.json:
    [
      {
        "instruction": "What research areas does SPL focus on?",
        "response": "SPL focuses on DEA, System Dynamics, Fuzzy Logic..."
      },
      ...
    ]
    """
    with open(path) as f:
        data = json.load(f)

    # Format into Llama 3 chat template
    def format_example(ex):
        return {
            "text": (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"You are the SPL assistant at Virginia Tech.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{ex['instruction']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
                f"{ex['response']}<|eot_id|>"
            )
        }

    formatted = [format_example(ex) for ex in data]
    return Dataset.from_list(formatted)


# ── QLoRA config ──────────────────────────────────────────────────────────────

def get_bnb_config():
    """4-bit quantization — this is what makes it fit on one GPU."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 from QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # extra memory saving
    )


def get_lora_config():
    """LoRA adapter config — only trains ~0.1% of parameters."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                   # rank of the low-rank matrices
        lora_alpha=32,          # scaling factor
        lora_dropout=0.05,
        target_modules=[        # which layers to adapt
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    print(f"Loading model: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
    )
    model.config.use_cache = False

    # Apply LoRA adapters
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    # Output example: trainable params: 6,815,744 || all params: 8,037,584,896
    # Only ~0.08% of params are trained — that's why it's so cheap!

    # Load dataset
    dataset = load_dataset(args.data_path)
    print(f"Training on {len(dataset)} examples")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,              # bfloat16 — better for A100
        report_to="none",       # disable wandb etc.
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter (not the full model — much smaller file)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved fine-tuned adapter to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
