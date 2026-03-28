"""
Resume Summarizer Training Script
Fine-tunes google/gemma-3-270M on resume summarization using LoRA.
Loads tokenized data from data/tokenised/ and trains the model.

Usage:
    python src/train.py
    python src/train.py --output-dir models/resume-summarizer --num-epochs 3
"""

import argparse
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

BASE_MODEL      = "google/gemma-3-270M"
DATA_DIR        = "data/tokenized_gemma"
OUTPUT_DIR      = "models/resume-summarizer-gemma"
NUM_EPOCHS      = 10
BATCH_SIZE      = 4
GRAD_ACCUM_STEPS = 2
LEARNING_RATE   = 2e-4
WARMUP_RATIO    = 0.1
SAVE_STEPS      = 100
LOGGING_STEPS   = 10
MAX_LENGTH      = 1024

# LoRA configuration
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
# Gemma uses a different attention/MLP naming scheme than GPT-2.
# Target modules should match the actual submodules in the base model.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "up_proj",
    "gate_proj",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_datasets(data_dir: str):
    """Load tokenized train and validation datasets."""
    train_dataset = load_from_disk(os.path.join(data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(data_dir, "val"))
    return train_dataset, val_dataset


def setup_model_and_tokenizer(base_model: str):
    """Load model and tokenizer with LoRA configuration."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto",  # Automatically distribute across available GPUs
    )

    # Prepare model for LoRA
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def setup_training_args(output_dir: str, num_epochs: int, batch_size: int, learning_rate: float):
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=learning_rate,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,  # Keep only last 2 checkpoints
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Use mixed precision training
        dataloader_pin_memory=False,  # Avoid memory issues
        report_to="none",  # Disable wandb/tensorboard logging
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    base_model: str = BASE_MODEL,
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Main training function."""
    print(f"🚀 Starting training with {base_model}")
    print(f"📊 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎯 Training for {num_epochs} epochs")

    # Load datasets
    print("📚 Loading datasets...")
    train_dataset, val_dataset = load_datasets(data_dir)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Setup model and tokenizer
    print("🤖 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(base_model)

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )

    # Setup training arguments
    training_args = setup_training_args(output_dir, num_epochs, batch_size, learning_rate)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("🏃 Starting training...")
    trainer.train()

    # Save the final model
    print("💾 Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete!")
    print(f"📁 Model saved to: {output_dir}")

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Resume Summarizer")
    parser.add_argument("--base-model", default=BASE_MODEL,
                       help="Base model to fine-tune")
    parser.add_argument("--data-dir", default=DATA_DIR,
                       help="Directory containing tokenized datasets")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                       help="Directory to save trained model")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help="Learning rate")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
