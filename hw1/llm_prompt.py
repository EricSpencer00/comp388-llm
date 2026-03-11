#!/usr/bin/env python3
"""
LLM Prompting Script

Loads an LLM from Hugging Face and generates responses to user prompts.
"""

import argparse
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

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


def generate_manual(model, tokenizer, input_ids, attention_mask, device, max_new_tokens=128, temperature=0.7):
    """Manual token generation with optional sampling to avoid .generate() crashes."""
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
        
        # Get the last logit
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append and continue
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Stop if we hit EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text using an LLM")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name from Hugging Face Hub")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Basic environment diagnostics
    if sys.version_info >= (3, 14):
        print("Warning: Python 3.14+ may be incompatible with some PyTorch builds.\nConsider using Python 3.11 or 3.10 if you see crashes.")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Reduce parallelism to avoid resource_tracker / semaphore issues
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
    # Load model and tokenizer with safer, low-memory options
    print(f"Loading model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Failed to load tokenizer for {args.model}: {e}")
        raise

    try:
        # Use low_cpu_mem_usage to reduce peak memory during loading
        model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    except Exception as e:
        print(f"Model load failed with error: {e}")
        print("Retrying with CPU-only map and smaller model suggestion...")
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map={'': 'cpu'}, low_cpu_mem_usage=True)
        except Exception as e2:
            print(f"Fallback load also failed: {e2}")
            print("If this persists, try running with a smaller model (e.g. 'distilgpt2') or use Python 3.11 and a matching PyTorch wheel.")
            raise
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        # If moving to device fails, continue on CPU
        device = "cpu"
        model.to(device)
    
    # Ensure model is in eval mode for inference
    model.eval()
    
    # Encode input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    # Provide an explicit attention mask (all tokens are real tokens here)
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Generate response using manual generation
    output = generate_manual(model, tokenizer, input_ids, attention_mask, device, 
                            max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    
    # Ensure output is on CPU before decoding
    if device == "cuda":
        output = output.cpu()
    
    # Decode and print
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(f"Prompt: {args.prompt}")
    print("="*50)
    print(f"Response: {response}")
    print("="*50)


if __name__ == "__main__":
    main()
