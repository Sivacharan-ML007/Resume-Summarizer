"""
Data Preparation for Resume Summariser Fine-tuning
Loads train.jsonl / val.jsonl, tokenises them with the model tokeniser,
and saves HuggingFace Dataset shards ready for train.py.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --data-dir data --output-dir data/tokenized --max-length 1024
"""

import argparse
import os
import json
import logging

# Set Hugging Face API token for model/tokenizer downloads
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXXXXXXXXXXX"

from transformers import AutoTokenizer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

BASE_MODEL   = "google/gemma-3-270M"  # conversational model for summarization tasks
DATA_DIR     = "data"
OUTPUT_DIR   = "data/tokenized_gemma"
MAX_LENGTH   = 1024               # truncate prompt+completion to this many tokens
IGNORE_INDEX = -100               # standard label mask value for cross-entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def tokenise_and_mask(record: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenise a prompt+completion record and mask prompt tokens in labels
    so loss is only computed on the completion (summary) tokens.

    Labels for prompt tokens are set to IGNORE_INDEX (-100) so they are
    excluded from the cross-entropy loss - the model only learns to generate
    the summary, not to reproduce the input prompt.
    """
    prompt     = record["prompt"]
    completion = record["completion"]

    # Tokenise prompt alone to find where completion starts
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    # Tokenise full text (prompt + completion)
    full_ids = tokenizer(
        prompt + completion,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    input_ids      = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    # Mask prompt tokens - only train on completion tokens
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]

    # Mask padding tokens
    labels = [
        IGNORE_INDEX if attention_mask[i] == 0 else labels[i]
        for i in range(len(labels))
    ]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def prepare(
    base_model: str  = BASE_MODEL,
    data_dir: str    = DATA_DIR,
    output_dir: str  = OUTPUT_DIR,
    max_length: int  = MAX_LENGTH,
):
    # Load tokeniser
    logger.info(f"Loading tokeniser: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not available

    # Load raw data
    train_records = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(data_dir, "val.jsonl"))

    # Tokenise
    logger.info(f"Tokenising with max_length={max_length} ...")

    def _tokenise(records):
        return [tokenise_and_mask(r, tokenizer, max_length) for r in records]

    train_tokenised = _tokenise(train_records)
    val_tokenised   = _tokenise(val_records)

    # Build HuggingFace Datasets
    train_dataset = Dataset.from_list(train_tokenised)
    val_dataset   = Dataset.from_list(val_tokenised)

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "val"))

    logger.info(f"Saved tokenised train ({len(train_dataset)} samples) -> {output_dir}/train")
    logger.info(f"Saved tokenised val   ({len(val_dataset)} samples) -> {output_dir}/val")

    # Print sample
    sample      = train_records[0]
    sample_tok  = train_tokenised[0]
    n_prompt    = sum(1 for l in sample_tok["labels"] if l == IGNORE_INDEX)
    n_completion = sum(1 for l in sample_tok["labels"] if l != IGNORE_INDEX)
    logger.info(
        f"Sample token counts - prompt (masked): {n_prompt}, "
        f"completion (trained): {n_completion}, total: {len(sample_tok['input_ids'])}"
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tokenise resume summariser training data")
    parser.add_argument("--base-model",  default=BASE_MODEL,   help="HuggingFace model id for tokeniser")
    parser.add_argument("--data-dir",    default=DATA_DIR,     help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output-dir",  default=OUTPUT_DIR,   help="Directory to save tokenised datasets")
    parser.add_argument("--max-length",  type=int, default=MAX_LENGTH, help="Max token sequence length")
    args = parser.parse_args()

    prepare(
        base_model  = args.base_model,
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        max_length  = args.max_length,
    )


if __name__ == "__main__":
    main()
    