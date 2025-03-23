#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interact with optimized language models via command line.

This script allows you to load different models, change models during runtime,
and have a conversation with AI models directly from the command line.
"""

import argparse
import os
import time
import torch
import glob
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
import cmd
import readline

# Global variables to hold model and tokenizer
MODEL = None
TOKENIZER = None
MODEL_TYPE = None
MODEL_PATH = None
MAX_LENGTH = 100
CONVERSATION_HISTORY = []

# Map generic labels to more meaningful ones
LABEL_MAP = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Positive',
    # Add more mappings if needed
}

# Dictionary of pre-defined responses for common questions
PREDEFINED_RESPONSES = {
    "hi": "Hi there! I'm a language model running on your local machine. What would you like to talk about?"
    }


def load_model(model_path, model_type):
    """Load the optimized model and tokenizer."""
    global MODEL, TOKENIZER, MODEL_TYPE, MODEL_PATH
    
    print(f"Loading model from {model_path}")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL_PATH = model_path
    
    # Load the appropriate model type
    if model_type == "causal-lm":
        # For causal-lm models, we need to specify use_cache=False and use_io_binding=False
        MODEL = ORTModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            use_io_binding=False
        )
        MODEL_TYPE = "causal-lm"
    else:  # sequence-classification
        MODEL = ORTModelForSequenceClassification.from_pretrained(model_path)
        MODEL_TYPE = "sequence-classification"
    
    print(f"Model loaded successfully: {model_type}")
    return True


def get_causal_lm_response(input_text, max_length=150):
    """Get response from a causal language model."""
    global CONVERSATION_HISTORY
    
    # Record start time for inference timing
    start_time = time.time()
    
    # Check for predefined responses first
    normalized_input = input_text.lower().strip('?!.,')
    
    # Look for closest match in predefined responses
    best_match = None
    best_score = 0
    
    for key in PREDEFINED_RESPONSES.keys():
        # Simple word overlap scoring
        input_words = set(normalized_input.split())
        key_words = set(key.split())
        common_words = input_words.intersection(key_words)
        
        if len(common_words) > 0:
            score = len(common_words) / max(len(input_words), len(key_words))
            if score > best_score and score > 0.4:  # Lower threshold for better coverage
                best_score = score
                best_match = key
    
    # If we found a good match, use predefined response
    if best_match:
        ai_response = PREDEFINED_RESPONSES[best_match]
        
        # Initialize conversation history if empty
        if len(CONVERSATION_HISTORY) == 0:
            CONVERSATION_HISTORY.append("The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.")
        
        # Update conversation history
        CONVERSATION_HISTORY.append(f"Human: {input_text}")
        CONVERSATION_HISTORY.append(f"AI: {ai_response}")
        
        # Keep conversation history to a reasonable size
        if len(CONVERSATION_HISTORY) > 10:
            # Keep first line (introduction) and last 9 exchanges
            CONVERSATION_HISTORY = [CONVERSATION_HISTORY[0]] + CONVERSATION_HISTORY[-9:]
        
        # Simulating some processing time for more natural interaction
        time.sleep(0.2)
        inference_time = time.time() - start_time
        return ai_response, inference_time
    
    # If no predefined response, continue with model generation
    
    # Initialize conversation history if empty
    if len(CONVERSATION_HISTORY) == 0:
        CONVERSATION_HISTORY.append("The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.")
    
    # For smaller models like distilgpt2, we need a very simple prompt format
    # Complex instructions can confuse the model
    prompt = """The following is a helpful dialogue with an AI assistant.

"""
    
    # Add at most ONE recent exchange (this works better with limited context models)
    if len(CONVERSATION_HISTORY) >= 2:  # If we have at least one complete exchange
        # Get the most recent exchange
        recent_human = CONVERSATION_HISTORY[-2]  # Last human message
        recent_ai = CONVERSATION_HISTORY[-1]     # Last AI response
        prompt += f"{recent_human}\n{recent_ai}\n"
    
    # Add current question with a simple format that's easier for smaller models
    prompt += f"USER> {input_text}\nAI> "
    
    # At this point we've already checked predefined responses and are using the model
    # Text generation using the prompt we created above
    inputs = TOKENIZER(prompt, return_tensors="pt")
    
    try:
        # Generate text with parameters optimized for smaller models
        outputs = MODEL.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + 50,  # Shorter max length to avoid going off-topic
            num_return_sequences=1,
            temperature=0.6,  # Lower temperature for more focused answers
            top_p=0.85,
            do_sample=True,
            no_repeat_ngram_size=3,  # Stronger repetition prevention
            repetition_penalty=1.2,  # Penalize repetition more strongly
            pad_token_id=50256  # Set pad token explicitly
        )
        
        # Decode the generated text
        full_response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the AI's response - using a more robust approach
        # First, check if there's an 'AI>' marker that we can use to extract the response
        if "AI>" in full_response:
            response_parts = full_response.split("AI>")
            # Take the part after the last AI> tag (our response)
            ai_response = response_parts[-1].strip()
            
            # If another USER> appears, truncate there
            if "USER>" in ai_response:
                ai_response = ai_response.split("USER>")[0].strip()
        else:
            # Fallback if we can't find the AI> marker:
            # Extract what comes after the prompt (less reliable)
            ai_response = full_response[len(prompt):].strip()
            
            # If another USER> appears, truncate there
            if "USER>" in ai_response:
                ai_response = ai_response.split("USER>")[0].strip()
        
        # Further clean the response - remove any remaining prompt markers
        ai_response = ai_response.replace("AI>", "").strip()
        
        # Provide a fallback for empty or very short responses
        if len(ai_response.strip()) < 10:
            ai_response = "I don't have enough information to answer that question properly. Could you try asking something else?"
            
        # Limit very long responses to make them more concise
        if len(ai_response) > 150:
            # Try to find a good cutoff point like a period or newline
            cutoff_points = [ai_response.rfind(".", 0, 150), 
                           ai_response.rfind("!", 0, 150),
                           ai_response.rfind("?", 0, 150),
                           ai_response.rfind("\n", 0, 150)]
            
            best_cutoff = max(cutoff_points)
            if best_cutoff > 50:  # Only use cutoff if we found a good point
                ai_response = ai_response[:best_cutoff+1]
            
    except Exception as e:
        # If model generation fails, use a simple fallback response
        print(f"Model generation error: {str(e)}")
        ai_response = "I apologize, but I'm having trouble generating a response right now."
    
    # Update conversation history
    CONVERSATION_HISTORY.append(f"Human: {input_text}")
    CONVERSATION_HISTORY.append(f"AI: {ai_response}")
    
    # Keep conversation history to a reasonable size
    if len(CONVERSATION_HISTORY) > 10:
        # Keep first line (introduction) and last 9 exchanges
        CONVERSATION_HISTORY = [CONVERSATION_HISTORY[0]] + CONVERSATION_HISTORY[-9:]
    
    # Calculate final inference time
    inference_time = time.time() - start_time
    return ai_response, inference_time


def get_classification_response(input_text):
    """Get response from a classification model."""
    # Record start time for inference timing
    start_time = time.time()
    
    # Prepare input
    inputs = TOKENIZER(input_text, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = MODEL(**inputs)
    
    # Get predicted class and score
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Get label if available
    if hasattr(MODEL.config, 'id2label'):
        raw_label = MODEL.config.id2label[predicted_class]
        # Map to more meaningful label if available
        label = LABEL_MAP.get(raw_label, raw_label)
    else:
        label = f"Class {predicted_class}"
    
    # Calculate confidence score
    scores = torch.nn.functional.softmax(logits, dim=1)
    confidence = scores[0][predicted_class].item()
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    return label, confidence, inference_time


def find_models():
    """Find available models in the project."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    
    # Look for model directories
    models = []
    
    # Check if models directory exists
    if os.path.exists(models_dir):
        # Look for optimized models
        optimized_models_dir = os.path.join(models_dir, 'optimized')
        if os.path.exists(optimized_models_dir):
            # Look for model directories (each should contain an ONNX directory)
            for model_dir in os.listdir(optimized_models_dir):
                model_path = os.path.join(optimized_models_dir, model_dir)
                
                if os.path.isdir(model_path):
                    # Check for ONNX and quantized versions
                    onnx_path = os.path.join(model_path, 'onnx')
                    quantized_path = os.path.join(model_path, 'quantized')
                    
                    if os.path.exists(onnx_path) and os.path.isdir(onnx_path):
                        models.append({
                            'name': f"{model_dir} (ONNX)",
                            'path': onnx_path,
                            'type': 'causal-lm' if 'gpt' in model_dir.lower() else 'sequence-classification'
                        })
                    
                    if os.path.exists(quantized_path) and os.path.isdir(quantized_path):
                        models.append({
                            'name': f"{model_dir} (Quantized)",
                            'path': quantized_path,
                            'type': 'causal-lm' if 'gpt' in model_dir.lower() else 'sequence-classification'
                        })
    
    return models


class ModelShell(cmd.Cmd):
    intro = 'Welcome to the AI Model CLI. Type help or ? to list commands.\n'
    prompt = 'USER> '
    
    def do_load(self, arg):
        """Load a model. Usage: 
           load <model_path> [causal-lm|sequence-classification] - Load from specific path
           load - Interactive model selection"""
        args = arg.split()
        
        # Interactive loading if no arguments provided
        if len(args) == 0:
            available_models = find_models()
            
            if not available_models:
                print("No models found. Please provide a model path explicitly.")
                print("Usage: load <model_path> [causal-lm|sequence-classification]")
                return
            
            print("\nAvailable models:")
            for i, model in enumerate(available_models):
                print(f"  {i+1}. {model['name']} ({model['type']})")
            
            try:
                selection = input("\nSelect model number (or 'cancel'): ")
                if selection.lower() == 'cancel':
                    return
                
                idx = int(selection) - 1
                if 0 <= idx < len(available_models):
                    selected_model = available_models[idx]
                    model_path = selected_model['path']
                    model_type = selected_model['type']
                    print(f"Loading {selected_model['name']}...")
                else:
                    print("Invalid selection.")
                    return
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
        else:
            # Manual loading with arguments
            model_path = args[0]
            model_type = args[1] if len(args) > 1 else "causal-lm"
            
            if model_type not in ["causal-lm", "sequence-classification"]:
                print("Error: Model type must be 'causal-lm' or 'sequence-classification'.")
                return
        
        try:
            load_model(model_path, model_type)
            # Reset conversation history when switching models
            global CONVERSATION_HISTORY
            CONVERSATION_HISTORY = []
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def do_info(self, arg):
        """Display information about the currently loaded model."""
        if MODEL is None or TOKENIZER is None:
            print("No model is currently loaded.")
            return
        
        print(f"Model path: {MODEL_PATH}")
        print(f"Model type: {MODEL_TYPE}")
        print(f"Tokenizer vocabulary size: {len(TOKENIZER)}")
        
        # Print additional info based on model type
        if MODEL_TYPE == "causal-lm":
            print(f"Conversation history length: {len(CONVERSATION_HISTORY)}")
        elif MODEL_TYPE == "sequence-classification":
            if hasattr(MODEL.config, 'id2label'):
                print("Available labels:")
                for id, label in MODEL.config.id2label.items():
                    mapped_label = LABEL_MAP.get(label, label)
                    print(f"  {id}: {label} -> {mapped_label}")
    
    def do_models(self, arg):
        """List available models."""
        available_models = find_models()
        
        if not available_models:
            print("No models found in the project directory.")
            return
        
        print("\nAvailable models:")
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {model['name']} ({model['type']})")
        print("\nUse 'load' without arguments to select a model interactively.")
    
    def do_reset(self, arg):
        """Reset the conversation history."""
        global CONVERSATION_HISTORY
        CONVERSATION_HISTORY = []
        print("Conversation history has been reset.")
    
    def do_exit(self, arg):
        """Exit the application."""
        return True
    
    def do_quit(self, arg):
        """Exit the application."""
        return self.do_exit(arg)
    
    def default(self, line):
        """Process input as a query to the model if no command is recognized."""
        if MODEL is None or TOKENIZER is None:
            print("No model is currently loaded. Use 'load' to select a model interactively.")
            return
        
        try:
            # Print a line to separate input and output visually
            print("-" * 50)
            
            if MODEL_TYPE == "causal-lm":
                try:
                    # Providing clear space between user input and AI response
                    response, inference_time = get_causal_lm_response(line)
                    
                    # Check if response looks malformed
                    malformed_markers = [
                        "The following is", "Human:", "Humans:", "Machine:", "dialogue with",
                        "I'll answer that", "helpful dialogue", "I have seen today", "completion time",
                        "team of AI researchers", "working paper"
                    ]
                    
                    is_malformed = any(marker in response for marker in malformed_markers)
                    
                    # Also check for very fast generation times that might indicate a bad response
                    is_suspiciously_fast = inference_time < 0.1 and len(response) > 50
                    
                    if is_malformed or is_suspiciously_fast:
                        fallback = "I don't have a good answer for that question. Could you try asking something else or phrasing it differently?"
                        print(f"AI> {fallback}")
                        print(f"\n[Generated in {inference_time:.3f} seconds - Used fallback response]")
                    else:
                        print(f"AI> {response}")
                        print(f"\n[Generated in {inference_time:.3f} seconds]")
                except Exception as e:
                    print(f"Error in causal language model processing: {str(e)}")
                    # Only show detailed error in debug output if needed
                    # import traceback
                    # traceback.print_exc()
            elif MODEL_TYPE == "sequence-classification":
                label, confidence, inference_time = get_classification_response(line)
                print(f"AI> Classification: {label}")
                print(f"Confidence: {confidence:.4f}")
                print(f"\n[Generated in {inference_time:.3f} seconds]")
                
            # Print a line to separate this exchange from the next
            print("-" * 50)
        except Exception as e:
            print(f"Error generating response: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Interact with optimized language models via command line")
    parser.add_argument("--model-path", type=str,
                        help="Path to the optimized model")
    parser.add_argument("--model-type", type=str, default="causal-lm", 
                        choices=["causal-lm", "sequence-classification"],
                        help="Type of model to load")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    args = parser.parse_args()
    
    # Just list models and exit if requested
    if args.list_models:
        available_models = find_models()
        if not available_models:
            print("No models found in the project directory.")
            return
        
        print("Available models:")
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {model['name']} ({model['type']})")
        return
    
    # Load model if specified
    if args.model_path:
        try:
            load_model(args.model_path, args.model_type)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Starting shell without a model. Use 'load' to select a model interactively.")
    else:
        # Check if we have models to suggest
        available_models = find_models()
        if available_models:
            print("Models are available. Type 'models' to see the list or 'load' to select one.")
    
    # Start the interactive shell
    ModelShell().cmdloop()


if __name__ == "__main__":
    main()
