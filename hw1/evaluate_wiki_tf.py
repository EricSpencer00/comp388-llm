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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prevent multiprocessing issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Disable MPS acceleration; use CPU fallback

# Set multiprocessing start method early on macOS
if sys.platform == 'darwin':
    try:
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
    except Exception:
        pass
    # Force CPU-only PyTorch on macOS to avoid MPS/Metal issues
    try:
        torch.set_default_device('cpu')
    except Exception:
        pass


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


def generate_manual(model, tokenizer, input_ids, attention_mask, device, max_new_tokens=10):
    """Manual token generation loop to avoid .generate() crashes."""
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
        
        # Get the last logit and pick the token with highest probability
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append and continue
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Stop if we hit EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Wikipedia true/false statements")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name from Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--data_path", type=str, default="hw1/data/wiki_tf.jsonl", help="Path to data file")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {args.data_path}")
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
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]  # Track prompt length
        attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        output = generate_manual(model, tokenizer, input_ids, attention_mask, device, max_new_tokens=10)
        
        # Extract only the generated tokens (after the prompt)
        generated_ids = output[0, prompt_len:]
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
