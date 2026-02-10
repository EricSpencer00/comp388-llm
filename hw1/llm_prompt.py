#!/usr/bin/env python3
"""
LLM Prompting Utility

Loads an LLM from Hugging Face and generates responses to user prompts.
Can be used as a standalone script or imported as a module.
"""

import argparse
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# === Environment Setup (Happy Path for macOS/Python 3.14+) ===

# Prevent multiprocessing issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set multiprocessing start method early on macOS
if sys.platform == 'darwin':
    try:
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
    except Exception:
        pass
    # Force CPU-only PyTorch on macOS to avoid MPS/Metal issues which common cause bus errors
    try:
        torch.set_default_device('cpu')
    except Exception:
        pass

def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer from Hugging Face Hub."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = get_device()
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate(model, tokenizer, prompt, device, max_new_tokens=128, temperature=0.0):
    """
    Generate text using manual token loop to avoid .generate() crashes on some platforms.
    Set temperature=0.0 for greedy decoding.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]
    
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
        
        # Get the last logit
        next_token_logits = outputs.logits[:, -1, :]
        
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append and continue
        generated = torch.cat([generated, next_token], dim=-1)
        # Update attention mask
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        
        # Stop if we hit EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    # Return only the new tokens (after the prompt)
    return generated[0, prompt_len:]

def main():
    parser = argparse.ArgumentParser(description="Generate text using an LLM")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name from Hugging Face Hub")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    
    print(f"Generating for prompt: \"{args.prompt}\"")
    output_ids = generate(model, tokenizer, args.prompt, device, 
                         max_new_tokens=args.max_new_tokens, 
                         temperature=args.temperature)
    
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(f"Response: {response}")
    print("="*50)

if __name__ == "__main__":
    main()
