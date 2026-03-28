# Resume Summarizer

A fine-tuned language model that generates concise summaries of resumes using Gemma-3-270M and LoRA (Low-Rank Adaptation).

## Features

- **Efficient Fine-tuning**: Uses LoRA to train only ~0.5% of model parameters
- **Resume-focused**: Trained on diverse resume data across multiple industries
- **Easy Inference**: Simple command-line interface for generating summaries
- **Scalable**: Works with different batch sizes and learning rates

## Project Structure

```
├── data/
│   ├── train.jsonl          # Training data (resume -> summary pairs)
│   ├── val.jsonl            # Validation data
│   └── tokenised/           # Tokenized datasets (created by prepare_data.py)
├── src/
│   ├── prepare_data.py      # Data preparation script
│   ├── train.py            # Training script
│   ├── infer.py            # Inference script
│   └── synthetic_data.py   # Generate synthetic training data
├── models/                 # Trained models (created during training)
└── requirements.txt        # Python dependencies
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
python src/prepare_data.py
```

### 3. Train Model

```bash
# Train with default settings (3 epochs)
python src/train.py

# Train with custom settings
python src/train.py --num-epochs 5 --batch-size 2 --learning-rate 1e-4
```

### 4. Generate Summaries

```bash
# Summarize from text
python src/infer.py --resume-text "John Doe is a software engineer with 5 years experience..."

# Summarize from file
python src/infer.py --resume-file path/to/resume.txt
```

## Training Details

- **Base Model**: `microsoft/DialoGPT-medium` (345M parameters)
- **Fine-tuning**: LoRA with rank 16, alpha 32
- **Training**: Causal language modeling with masked prompt tokens
- **Hardware**: Works on single GPU or CPU (slower)

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-epochs` | 3 | Number of training epochs |
| `batch-size` | 4 | Training batch size |
| `learning-rate` | 2e-4 | Learning rate |
| `max-length` | 1024 | Maximum sequence length |

### LoRA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora-r` | 16 | LoRA rank |
| `lora-alpha` | 32 | LoRA alpha |
| `target-modules` | c_attn, c_proj, c_fc | GPT-2 attention modules |

## Example Usage

### Training
```bash
python src/train.py --num-epochs 2 --output-dir models/my-resume-summarizer
```

### Inference
```bash
python src/infer.py --model-path models/my-resume-summarizer --resume-text "
Senior Software Engineer with 8 years experience. Led development of
scalable microservices at Google, reducing latency by 70%. Expert in
Python, Go, Kubernetes, and distributed systems.
"
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

2. **CUDA out of memory**
   - Reduce batch size: `--batch-size 2`
   - Use gradient accumulation: increase `GRAD_ACCUM_STEPS`

3. **Model download slow**
   - Model downloads once and caches locally
   - Subsequent runs are faster

4. **Poor summary quality**
   - Train for more epochs
   - Use higher quality training data
   - Adjust generation parameters in `infer.py`

### Performance Tips

- **GPU Training**: Use `device_map="auto"` for multi-GPU
- **Memory**: Use `torch_dtype=torch.float16` for lower memory usage
- **Speed**: Increase `gradient_accumulation_steps` to simulate larger batches

## License

This project uses open-source models and data. Check individual component licenses.
