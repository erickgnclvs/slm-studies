# SLM Studies

A project for studying and deploying small language models (SLMs) optimized for resource-constrained environments like Mac M1 with 8GB RAM.

## Project Structure

```
├── data/           # Training and evaluation datasets
├── models/         # Saved model checkpoints and configurations
├── notebooks/      # Jupyter notebooks for experiments and tutorials
├── scripts/        # Python scripts for training, fine-tuning, and deployment
├── docs/           # Documentation and resources
└── app/            # Deployment application
```

## Getting Started

### Setup Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Working with Small Language Models

This project focuses on deploying lightweight language models suitable for Mac M1 with 8GB RAM:

- **DistilBERT/TinyBERT**: Compressed BERT models
- **BERT-mini/BERT-tiny**: Smaller BERT variants
- **GPT-2 small**: Lightweight GPT model
- **MobileBERT**: Mobile-optimized BERT
- **ALBERT**: A Lite BERT architecture

## Model Optimization Techniques

- Quantization (int8/int4)
- Knowledge distillation
- Pruning
- ONNX Runtime optimization

## Deployment

The project includes a Flask-based API for model deployment and inference. The web interface allows users to interact with the optimized models through a simple Q&A interface.

## Features

- **Model Optimization**: Scripts for optimizing models with ONNX and quantization
- **Inference Benchmarking**: Compare performance before and after optimization
- **Web Interface**: Interactive UI for testing models
- **Curated Responses**: Pre-defined high-quality responses for common ML questions

## Current Implementation

The current implementation uses DistilGPT2 (82M parameters) optimized with ONNX Runtime and dynamic quantization, achieving significant speed improvements while maintaining reasonable output quality.

```bash
# Run the deployment server
python scripts/deploy.py
```

## Resources

- [Hugging Face Small Models](https://huggingface.co/models?sort=downloads&search=distil)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
