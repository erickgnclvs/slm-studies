#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web application for serving small language model predictions.

This Flask application provides a web interface for interacting with
small language models optimized for Mac M1.
"""

import os
import sys
import argparse
import torch
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global variables to hold model and tokenizer
MODEL = None
TOKENIZER = None
MODEL_TYPE = None
MODEL_NAME = None
MAX_LENGTH = 100

def load_model(model_path, model_type):
    """Load the optimized model and tokenizer."""
    global MODEL, TOKENIZER, MODEL_TYPE, MODEL_NAME
    
    print(f"Loading model from {model_path}")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL_NAME = os.path.basename(model_path)
    
    # Load the appropriate model type
    if model_type == "causal-lm":
        try:
            # Try to load ONNX optimized model first
            MODEL = ORTModelForCausalLM.from_pretrained(model_path)
        except:
            # Fall back to regular model
            from transformers import AutoModelForCausalLM
            MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        MODEL_TYPE = "causal-lm"
    else:  # sequence-classification
        try:
            # Try to load ONNX optimized model first
            MODEL = ORTModelForSequenceClassification.from_pretrained(model_path)
        except:
            # Fall back to regular model
            from transformers import AutoModelForSequenceClassification
            MODEL = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        MODEL_TYPE = "sequence-classification"
    
    print(f"Model loaded successfully: {model_type}")

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', 
                          model_name=MODEL_NAME,
                          model_type=MODEL_TYPE)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from input."""
    if MODEL is None or TOKENIZER is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    # Get form data
    input_text = request.form.get('input_text', '')
    max_length = int(request.form.get('max_length', MAX_LENGTH))
    temperature = float(request.form.get('temperature', 0.7))
    do_sample = request.form.get('do_sample', 'true').lower() == 'true'
    
    try:
        # Process based on model type
        if MODEL_TYPE == "causal-lm":
            # Text generation
            inputs = TOKENIZER(input_text, return_tensors="pt")
            
            # Generate text
            outputs = MODEL.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.9,
                do_sample=do_sample
            )
            
            # Decode the generated text
            generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            
            return jsonify({
                "input": input_text,
                "generated_text": generated_text
            })
            
        else:  # sequence-classification
            # Sentiment analysis or classification
            inputs = TOKENIZER(input_text, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                outputs = MODEL(**inputs)
            
            # Get predicted class and score
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get label if available
            if hasattr(MODEL.config, 'id2label'):
                label = MODEL.config.id2label[predicted_class]
            else:
                label = f"Class {predicted_class}"
            
            # Calculate confidence score
            scores = torch.nn.functional.softmax(logits, dim=1)
            confidence = scores[0][predicted_class].item()
            
            return jsonify({
                "input": input_text,
                "prediction": {
                    "label": label,
                    "confidence": confidence
                }
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if MODEL is None or TOKENIZER is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
    # Get request data
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"status": "error", "message": "Missing 'text' field"}), 400
    
    input_text = data['text']
    max_length = data.get('max_length', MAX_LENGTH)
    temperature = data.get('temperature', 0.7)
    do_sample = data.get('do_sample', True)
    
    try:
        # Process based on model type
        if MODEL_TYPE == "causal-lm":
            # Text generation
            inputs = TOKENIZER(input_text, return_tensors="pt")
            
            # Generate text
            outputs = MODEL.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.9,
                do_sample=do_sample
            )
            
            # Decode the generated text
            generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            
            return jsonify({
                "status": "ok",
                "input": input_text,
                "generated_text": generated_text
            })
            
        else:  # sequence-classification
            # Sentiment analysis or classification
            inputs = TOKENIZER(input_text, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                outputs = MODEL(**inputs)
            
            # Get predicted class and score
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get label if available
            if hasattr(MODEL.config, 'id2label'):
                label = MODEL.config.id2label[predicted_class]
            else:
                label = f"Class {predicted_class}"
            
            # Calculate confidence score
            scores = torch.nn.functional.softmax(logits, dim=1)
            confidence = scores[0][predicted_class].item()
            
            return jsonify({
                "status": "ok",
                "input": input_text,
                "prediction": {
                    "label": label,
                    "confidence": confidence
                }
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="Deploy a language model as a web application")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the model")
    parser.add_argument("--model-type", type=str, default="causal-lm", 
                        choices=["causal-lm", "sequence-classification"],
                        help="Type of model to deploy")
    parser.add_argument("--port", type=int, default=5000, 
                        help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Load the model
    load_model(args.model_path, args.model_type)
    
    # Run the server
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
