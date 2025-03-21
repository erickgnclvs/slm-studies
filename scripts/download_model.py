#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download and test small language models suitable for Mac M1 with 8GB RAM.

This script downloads a specified small language model from Hugging Face,
optimizes it for inference on Mac M1, and runs a simple test.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import pipeline
import time

# Models suitable for 8GB RAM Mac M1
SMALL_MODELS = {
    'distilgpt2': {
        'model_name': 'distilgpt2',
        'type': 'causal-lm',
        'description': 'Smaller version of GPT-2 (82M parameters)'
    },
    'distilbert-base': {
        'model_name': 'distilbert-base-uncased',
        'type': 'sequence-classification',
        'description': 'Distilled version of BERT (66M parameters)'
    },
    'albert-base': {
        'model_name': 'albert-base-v2',
        'type': 'sequence-classification',
        'description': 'A Lite BERT architecture (12M parameters)'
    },
    'tiny-bert': {
        'model_name': 'prajjwal1/bert-tiny',
        'type': 'sequence-classification',
        'description': 'Tiny BERT model (4.4M parameters)'
    },
    'mobilebert': {
        'model_name': 'google/mobilebert-uncased',
        'type': 'sequence-classification',
        'description': 'Mobile-optimized BERT (25M parameters)'
    },
    'gpt2-medium': {
        'model_name': 'gpt2-medium',
        'type': 'causal-lm',
        'description': 'Medium-sized GPT-2 (355M parameters) - may be slow on 8GB RAM'
    }
}

def list_available_models():
    """Print available small models."""
    print("\nAvailable small language models:")
    print("-" * 80)
    for key, model_info in SMALL_MODELS.items():
        print(f"- {key}: {model_info['description']}")
    print("-" * 80)

def download_and_test_model(model_key, save_dir="../models", test_text="Hello, I am a language model running on a Mac M1."):
    """Download, optimize and test a small language model."""
    if model_key not in SMALL_MODELS:
        print(f"Error: Model '{model_key}' not found. Use --list to see available models.")
        return
    
    model_info = SMALL_MODELS[model_key]
    model_name = model_info['model_name']
    model_type = model_info['type']
    
    print(f"\nDownloading {model_name} ({model_info['description']})...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Download model with optimizations for Mac M1
    device = 'cpu'  # Use CPU for better compatibility
    print(f'Device set to use {device}')
    
    if model_type == 'causal-lm':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True      # Optimize memory usage
        )
        nlp = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
        
        # Test the model
        print(f"\nTesting {model_key} with input: '{test_text}'")
        start_time = time.time()
        result = nlp(test_text, max_length=50, num_return_sequences=1)
        end_time = time.time()
        
        print(f"\nGenerated text: {result[0]['generated_text']}")
        
    elif model_type == 'sequence-classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True      # Optimize memory usage
        )
        nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)
        
        # Test the model
        print(f"\nTesting {model_key} with input: '{test_text}'")
        start_time = time.time()
        result = nlp(test_text)
        end_time = time.time()
        
        print(f"\nSentiment: {result[0]['label']} (Score: {result[0]['score']:.4f})")
    
    # Print performance metrics
    print(f"\nInference time: {end_time - start_time:.2f} seconds")
    print(f"Model size: {model.get_memory_footprint() / (1024 * 1024):.2f} MB")
    
    # Save the model and tokenizer
    model_save_path = os.path.join(save_dir, model_key)
    print(f"\nSaving model and tokenizer to {model_save_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save tokenizer first
    tokenizer.save_pretrained(model_save_path)
    
    try:
        # Try saving with safetensors disabled
        model.save_pretrained(model_save_path, safe_serialization=False)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Saving model configuration only...")
        # At minimum, save the config
        model.config.save_pretrained(model_save_path)
    
    print(f"\nModel {model_key} successfully downloaded, tested, and saved!")

def main():
    parser = argparse.ArgumentParser(description="Download and test small language models")
    parser.add_argument("--model", type=str, help="Model key to download and test")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--save-dir", type=str, default="../models", help="Directory to save the model")
    parser.add_argument("--test-text", type=str, default="Hello, I am a language model running on a Mac M1.", 
                        help="Text to test the model with")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        print("Error: Please specify a model with --model or use --list to see available models.")
        return
    
    download_and_test_model(args.model, args.save_dir, args.test_text)

if __name__ == "__main__":
    main()
