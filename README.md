# Resume Summarizer (Gemma)

A fine-tuned language model that generates concise summaries of resumes using Gemma-2B and LoRA (Low-Rank Adaptation).

## Features

- **Efficient Fine-tuning**: Uses LoRA to train only ~0.5% of model parameters
- **Resume-focused**: Trained on diverse resume data across multiple industries
- **Easy Inference**: Simple command-line interface for generating summaries
- **Scalable**: Works with different batch sizes and learning rates
- **PDF Support**: Automatically extract and summarize resumes from PDF files
- **Multi-format Input**: Support for text files, plain text, and PDF documents
- **Batch Processing**: Process multiple resumes simultaneously
- **Enhanced Output**: Structured summaries with key information extraction

## Project Structure

```
├── data/
│   ├── train.jsonl          # Training data (resume -> summary pairs)
│   ├── val.jsonl            # Validation data
│   └── tokenized_gemma/     # Tokenized datasets (created by prepare_data_gemma.py)
├── src/
│   ├── prepare_data_gemma.py # Data preparation script for Gemma
│   ├── train_gemma.py       # Training script for Gemma
│   ├── infer_gemma.py       # Inference script for Gemma
│   └── synthetic_data.py    # Generate synthetic training data
├── models/                  # Trained models (created during training)
│   └── resume-summarizer-gemma/
└── requirements.txt         # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Generate synthetic training data (optional)
python src/synthetic_data.py

# Tokenize data for training
python src/prepare_data_gemma.py
```

### 3. Train Model

```bash
# Train with default settings (15 epochs)
python src/train_gemma.py

# Train with custom settings
python src/train_gemma.py --num-epochs 20 --batch-size 2 --learning-rate 1e-4
```

### 4. Generate Summaries

```bash
# Summarize from text
python src/infer_gemma.py --resume-text "John Doe is a software engineer with 5 years experience..."

# Summarize from file
python src/infer_gemma.py --resume-file path/to/resume.txt

# Summarize from PDF (NEW)
python src/infer_gemma.py --resume-file path/to/resume.pdf

# Batch process multiple PDFs (NEW)
python src/infer_gemma.py --batch-dir path/to/resumes/

# Custom output format (NEW)
python src/infer_gemma.py --resume-file resume.pdf --output-format json
```

## Training Details

- **Base Model**: `google/gemma-2b` (2B parameters)
- **Fine-tuning**: LoRA with rank 16, alpha 32
- **Training**: Causal language modeling with masked prompt tokens
- **Hardware**: Works on single GPU or CPU (slower)

## Configuration

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resume-text` | - | Resume text to summarize |
| `resume-file` | - | Path to resume file (txt, pdf) |
| `batch-dir` | - | Directory with multiple resume files |
| `output-format` | text | Output format (text, json) |
| `model-path` | models/resume-summarizer-gemma | Path to trained model |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-epochs` | 15 | Number of training epochs |
| `batch-size` | 4 | Training batch size |
| `learning-rate` | 2e-4 | Learning rate |
| `max-length` | 1024 | Maximum sequence length |

### LoRA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora-r` | 16 | LoRA rank |
| `lora-alpha` | 32 | LoRA alpha |
| `target-modules` | q_proj, k_proj, v_proj, o_proj | Gemma attention modules |

## Example Usage

### Training
```bash
python src/train_gemma.py --num-epochs 15 --output-dir models/my-resume-summarizer-gemma
```

### Inference
```bash
# Text input
python src/infer_gemma.py --model-path models/resume-summarizer-gemma --resume-text "
Senior Software Engineer with 8 years experience. Led development of
scalable microservices at Google, reducing latency by 70%. Expert in
Python, Go, Kubernetes, and distributed systems.
"

# PDF input (NEW)
python src/infer_gemma.py --model-path models/resume-summarizer-gemma --resume-file resume.pdf

# Batch processing (NEW)
python src/infer_gemma.py --model-path models/resume-summarizer-gemma --batch-dir ./resumes/ --output-format json
```

### Expected Output
```
📋 GENERATED SUMMARY:
Senior Software Engineer with 8 years of experience at Google. Expert in
distributed systems, microservices, and Kubernetes. Led performance
optimizations reducing latency by 70%. Proficient in Python and Go.
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **PDF extraction fails**
   ```bash
   pip install pypdf pdfplumber
   ```

3. **CUDA out of memory**
   - Reduce batch size: `--batch-size 2`
   - Use gradient accumulation: increase `GRAD_ACCUM_STEPS`

4. **Model download slow**
   - Model downloads once and caches locally
   - Subsequent runs are faster

5. **Poor summary quality**
   - Train for more epochs
   - Use higher quality training data
   - Adjust generation parameters in `infer_gemma.py`

6. **Batch processing issues**
   - Ensure all files in batch directory are valid (txt or pdf)
   - Check file permissions and disk space

### Performance Tips

- **GPU Training**: Use `device_map="auto"` for multi-GPU
- **Memory**: Use `torch_dtype=torch.float16` for lower memory usage
- **Speed**: Increase `gradient_accumulation_steps` to simulate larger batches

## License

This project uses open-source models and data. Check individual component licenses.