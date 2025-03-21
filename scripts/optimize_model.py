#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimize language models for Mac M1 with 8GB RAM.

This script applies various optimization techniques to make language models
more efficient for deployment on resource-constrained devices.
"""

import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import time
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def optimize_model(model_path, output_path, quantize=True, onnx_export=True, model_type="causal-lm", model_id=None):
    """Optimize a pre-trained model for Mac M1 deployment."""
    print(f"Optimizing model from {model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Determine if we're using a local path or Hugging Face model ID
    if model_id is None:
        # Try to extract model ID from config file
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.loads(f.read())
                    if '_name_or_path' in config_data:
                        model_id = config_data['_name_or_path']
                        print(f"Found model ID in config: {model_id}")
            except Exception as e:
                print(f"Error reading config file: {e}")
        
        if model_id is None:
            print("Could not determine original model ID, using local path only")
            
    # Load tokenizer
    print(f"Loading model from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load model based on type
    if model_type == "causal-lm":
        model_class = AutoModelForCausalLM
        ort_model_class = ORTModelForCausalLM
    else:  # sequence-classification
        model_class = AutoModelForSequenceClassification
        ort_model_class = ORTModelForSequenceClassification
    
    # Load model with optimizations
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision
        local_files_only=True      # Use local files only
    )
    
    # Measure original model size and performance
    original_size = model.get_memory_footprint() / (1024 * 1024)  # MB
    print(f"Original model size: {original_size:.2f} MB")
    
    # Test inference speed
    test_input = "This is a test sentence to measure inference speed."
    encoded_input = tokenizer(test_input, return_tensors="pt")
    
    # For causal-lm models, we need to set specific parameters for generation
    generation_kwargs = {}
    if model_type == "causal-lm":
        generation_kwargs = {
            "max_length": 20,  # Limit output length for testing
            "num_return_sequences": 1,
            "do_sample": False  # Use greedy decoding for consistent timing
        }
    
    # Measure original model inference time
    start_time = time.time()
    with torch.no_grad():
        if model_type == "causal-lm":
            _ = model.generate(**encoded_input, **generation_kwargs)
        else:
            _ = model(**encoded_input)
    original_inference_time = time.time() - start_time
    print(f"Original inference time: {original_inference_time:.4f} seconds")
    
    # Export to ONNX format for better performance on Mac M1
    if onnx_export:
        print("\nExporting to ONNX format...")
        onnx_path = os.path.join(output_path, "onnx")
        
        # Export the model to ONNX format
        # Use the original model ID if available, otherwise use local path
        export_path = model_id if model_id is not None else model_path
        print(f"Using {export_path} for ONNX export")
        
        # Configure ONNX export options based on model type
        export_kwargs = {
            # Using only export=True as from_transformers is deprecated
            "export": True,
            "local_files_only": (model_id is None)  # Only use local_files_only if we don't have a model_id
        }
        
        # For causal-lm models, add specific configuration
        if model_type == "causal-lm":
            # For causal-lm, we have two options:
            # 1. use_cache=True, use_io_binding=True (default)
            # 2. use_cache=False, use_io_binding=False
            # We'll use option 2 for simplicity
            export_kwargs.update({
                "use_cache": False,  # Disable KV cache for simpler export
                "use_io_binding": False  # Must be False when use_cache is False
            })
        
        ort_model = ort_model_class.from_pretrained(
            export_path, 
            **export_kwargs
        )
        
        # Save the ONNX model
        ort_model.save_pretrained(onnx_path)
        tokenizer.save_pretrained(onnx_path)
        
        print(f"ONNX model saved to {onnx_path}")
        
        # Test ONNX model performance
        # For causal-lm models, we need to specify use_cache and use_io_binding parameters
        load_kwargs = {}
        if model_type == "causal-lm":
            load_kwargs["use_cache"] = False
            load_kwargs["use_io_binding"] = False  # Must be False when use_cache is False
            
        ort_model = ort_model_class.from_pretrained(onnx_path, **load_kwargs)
        
        start_time = time.time()
        try:
            if model_type == "causal-lm":
                _ = ort_model.generate(**encoded_input, **generation_kwargs)
            else:
                _ = ort_model(**encoded_input)
            onnx_inference_time = time.time() - start_time
            print(f"ONNX inference time: {onnx_inference_time:.4f} seconds")
            print(f"Speed improvement: {original_inference_time / onnx_inference_time:.2f}x")
        except Exception as e:
            print(f"Error testing ONNX model: {str(e)}")
            print("Skipping ONNX performance test, but will continue with deployment")
            onnx_inference_time = original_inference_time  # Use original time as fallback
    
    # Apply quantization for further size reduction
    if quantize and onnx_export:
        print("\nApplying quantization...")
        quantized_path = os.path.join(output_path, "quantized")
        
        # Initialize quantizer - ORTQuantizer doesn't accept use_cache or use_io_binding parameters
        quantizer = ORTQuantizer.from_pretrained(onnx_path)
        
        # Define quantization configuration - using dynamic quantization which doesn't require calibration
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False,  # Use dynamic quantization instead of static
            per_channel=False
        )
        
        # Apply quantization
        quantizer.quantize(
            save_dir=quantized_path,
            quantization_config=qconfig
        )
        
        # Save tokenizer with quantized model
        tokenizer.save_pretrained(quantized_path)
        
        print(f"Quantized model saved to {quantized_path}")
        
        # Test quantized model performance
        # Apply the same parameters as when loading the ONNX model
        load_kwargs = {}
        if model_type == "causal-lm":
            load_kwargs["use_cache"] = False
            load_kwargs["use_io_binding"] = False
            
        quantized_model = ort_model_class.from_pretrained(quantized_path, **load_kwargs)
        
        start_time = time.time()
        # Use the appropriate inference method based on model type
        if model_type == "causal-lm":
            _ = quantized_model.generate(**encoded_input, **generation_kwargs)
        else:
            _ = quantized_model(**encoded_input)
        quantized_inference_time = time.time() - start_time
        print(f"Quantized inference time: {quantized_inference_time:.4f} seconds")
        print(f"Speed improvement vs original: {original_inference_time / quantized_inference_time:.2f}x")
        
        # Estimate size reduction
        print(f"Estimated size reduction: ~75% (typical for int8 quantization)")
    
    print("\nOptimization complete!")
    print(f"Optimized models saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Optimize language models for Mac M1")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the pre-trained model")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the optimized model")
    parser.add_argument("--model-type", type=str, default="causal-lm", 
                        choices=["causal-lm", "sequence-classification"],
                        help="Type of model to optimize")
    parser.add_argument("--no-quantize", action="store_true", 
                        help="Skip quantization step")
    parser.add_argument("--no-onnx", action="store_true", 
                        help="Skip ONNX export")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Original Hugging Face model ID (optional)")
    
    args = parser.parse_args()
    
    optimize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        quantize=not args.no_quantize,
        onnx_export=not args.no_onnx,
        model_type=args.model_type,
        model_id=args.model_id
    )

if __name__ == "__main__":
    main()
