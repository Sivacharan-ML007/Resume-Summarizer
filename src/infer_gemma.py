"""
Resume Summarizer Inference Script
Loads a trained model and generates resume summaries.

Usage:
    python src/infer.py --resume-text "your resume text here"
    python src/infer.py --resume-file path/to/resume.txt
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/resume-summarizer-gemma"
BASE_MODEL = "google/gemma-3-270M"
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 150

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, base_model: str):
    """Load the fine-tuned model and tokenizer."""
    print(f"🤖 Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for Gemma models to avoid precision issues
        device_map="auto",
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)

    # Set to evaluation mode
    model.eval()

    return model, tokenizer


def generate_summary(model, tokenizer, resume_text: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """Generate a summary for the given resume text."""
    # Format the input as a prompt
    prompt = f"Summarize this resume:\n\n{resume_text}\n\nSummary:"

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH - max_new_tokens,
    ).to(model.device)

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,  # Add repetition penalty to avoid loops
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the summary part (after "Summary:")
    if "Summary:" in generated_text:
        summary = generated_text.split("Summary:")[-1].strip()
    else:
        summary = generated_text[len(prompt):].strip()

    return summary


def load_resume_from_file(file_path: str) -> str:
    """Load resume text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Resume Summaries")
    parser.add_argument("--resume-text", type=str,
                       help="Resume text to summarize")
    parser.add_argument("--resume-file", type=str,
                       help="Path to resume text file")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                       help="Path to trained model")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                       help="Maximum tokens to generate")

    args = parser.parse_args()

    # Validate input
    if not args.resume_text and not args.resume_file:
        print("❌ Error: Please provide either --resume-text or --resume-file")
        return

    if args.resume_text and args.resume_file:
        print("❌ Error: Please provide only one of --resume-text or --resume-file")
        return

    # Load resume text
    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print(f"❌ Error: File {args.resume_file} not found")
            return
        resume_text = load_resume_from_file(args.resume_file)
    else:
        resume_text = args.resume_text

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("💡 Train the model first using: python src/train.py")
        return

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, BASE_MODEL)

        # Generate summary
        print("📝 Generating summary...")
        summary = generate_summary(model, tokenizer, resume_text, args.max_tokens)

        # Print results
        print("\n" + "="*50)
        print("📄 ORIGINAL RESUME:")
        print("="*50)
        print(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        print("\n" + "="*50)
        print("📋 GENERATED SUMMARY:")
        print("="*50)
        print(summary)
        print("="*50)

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("💡 Make sure the model was trained successfully")


if __name__ == "__main__":
    main()