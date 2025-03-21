#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune a small language model on custom data.

This script fine-tunes a pre-trained small language model on custom data
while optimizing for Mac M1 with 8GB RAM.
"""

import argparse
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """Compute metrics for sequence classification."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def tokenize_function(examples, tokenizer, max_length, text_column="text"):
    """Tokenize the examples."""
    return tokenizer(examples[text_column], truncation=True, max_length=max_length)

def prepare_dataset(dataset_path, tokenizer, max_length=128, text_column="text", label_column=None):
    """Prepare the dataset for training."""
    # Load dataset
    if dataset_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=dataset_path)
    elif dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path)
    elif os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    # Split dataset if it doesn't have a validation split
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset = {
            "train": dataset["train"],
            "validation": dataset["test"]
        }
    
    # Tokenize the dataset
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            lambda examples: tokenize_function(examples, tokenizer, max_length, text_column),
            batched=True,
            remove_columns=[col for col in dataset[split].column_names if col != label_column]
        )
        
        # Format for the model
        if label_column:
            tokenized_dataset[split] = tokenized_dataset[split].rename_column(label_column, "labels")
    
    return tokenized_dataset

def finetune_model(model_path, dataset_path, output_dir, model_type="causal-lm", 
                  epochs=3, batch_size=8, learning_rate=5e-5, max_length=128,
                  text_column="text", label_column=None):
    """Fine-tune a pre-trained model on custom data."""
    print(f"Fine-tuning model from {model_path} on {dataset_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model based on type with optimizations for Mac M1
    if model_type == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True      # Optimize memory usage
        )
        # Ensure the model knows about the padding token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling, not masked
        )
    else:  # sequence-classification
        if label_column is None:
            raise ValueError("label_column must be specified for sequence classification")
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True      # Optimize memory usage
        )
        
        # Prepare data collator for sequence classification
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(
        dataset_path, tokenizer, max_length, text_column, label_column
    )
    
    # Define training arguments optimized for Mac M1 with 8GB RAM
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,  # Only keep the 2 best checkpoints to save disk space
        fp16=True,  # Use mixed precision training
        gradient_accumulation_steps=4,  # Reduce memory usage by accumulating gradients
        optim="adamw_torch",  # Use AdamW optimizer
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none"  # Don't report to any tracking service
    )
    
    # Initialize Trainer
    if model_type == "causal-lm":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    else:  # sequence-classification
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    
    # Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the final model
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning complete!")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small language model on custom data")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the pre-trained model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset for fine-tuning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model-type", type=str, default="causal-lm", 
                        choices=["causal-lm", "sequence-classification"],
                        help="Type of model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name for text data")
    parser.add_argument("--label-column", type=str, default=None,
                        help="Column name for labels (required for sequence-classification)")
    
    args = parser.parse_args()
    
    finetune_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        text_column=args.text_column,
        label_column=args.label_column
    )

if __name__ == "__main__":
    main()
