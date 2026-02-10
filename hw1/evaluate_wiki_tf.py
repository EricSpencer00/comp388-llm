#!/usr/bin/env python3
"""
Evaluate LLM on True/False Wikipedia Statements

Loads data/wiki_tf.jsonl and evaluates the model's ability to classify statements as true or false.
"""

import argparse
import json
import os
import sys
import torch
from llm_prompt import load_model_and_tokenizer, generate

def normalize_answer(text):
    """Extract true/false from model output."""
    text = text.lower().strip()
    
    if "true" in text:
        # Find first occurrence of "true"
        idx = text.find("true")
        return "true"
    elif "false" in text:
        # Find first occurrence of "false"
        idx = text.find("false")
        return "false"
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Wikipedia true/false statements")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name from Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--data_path", type=str, default="hw1/data/wiki_tf.jsonl", help="Path to data file")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    
    # Load data
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} examples\n")
    
    # Evaluate
    correct = 0
    incorrect_examples = []
    
    for example in data:
        statement = example["statement"]
        expected = example["label"].lower()
        
        # Create prompt - use a more direct format to encourage true/false response
        prompt = f"""Q: Is this statement true or false?
Statement: {statement}

A: true

Q: Is this statement true or false?
Statement: Mars has two moons.

A: true

Q: Is this statement true or false?
Statement: {statement}

A:"""
        
        # Generate response
        generated_ids = generate(model, tokenizer, prompt, device, max_new_tokens=10)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Normalize answer
        model_answer = normalize_answer(generated_text)
        
        if model_answer == expected:
            correct += 1
        else:
            if len(incorrect_examples) < 5:
                incorrect_examples.append({
                    "id": example["id"],
                    "statement": statement,
                    "expected": expected,
                    "model_output": model_answer if model_answer else "(no valid answer)"
                })
    
    # Print results
    accuracy = correct / len(data) * 100
    print("="*60)
    print(f"Results on Wikipedia True/False Task")
    print("="*60)
    print(f"Accuracy: {correct}/{len(data)} = {accuracy:.1f}%\n")
    
    if incorrect_examples:
        print("Incorrect Examples (up to 5):")
        print("-"*60)
        for example in incorrect_examples:
            print(f"ID: {example['id']}")
            print(f"Statement: {example['statement']}")
            print(f"Expected: {example['expected']}")
            print(f"Model Output: {example['model_output']}")
            print()


if __name__ == "__main__":
    main()
