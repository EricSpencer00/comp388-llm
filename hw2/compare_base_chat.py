#!/usr/bin/env python3
"""
compare_base_chat.py

Sends the same prompt to a base (pretrained) model and its instruction-tuned
(chat) counterpart, then prints both outputs side by side.
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# macOS compatibility
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if sys.platform == 'darwin':
    try:
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
    except Exception:
        pass
    try:
        torch.set_default_device('cpu')
    except Exception:
        pass


def generate_manual(model, tokenizer, input_ids, attention_mask, device,
                    max_new_tokens=64, temperature=0.7):
    """Token-by-token generation with optional sampling."""
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=generated, attention_mask=attention_mask)

        next_token_logits = outputs.logits[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device)], dim=-1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


def load_and_generate(model_name, prompt, device, max_new_tokens, temperature):
    """Load a model, generate a response, and return the output text."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    prompt_len = input_ids.shape[1]

    output = generate_manual(
        model, tokenizer, input_ids, attention_mask, device,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )

    generated_ids = output[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs instruction-tuned model outputs"
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="HuggingFace base model name")
    parser.add_argument("--chat_model", type=str, required=True,
                        help="HuggingFace instruction-tuned model name")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to send to both models")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Base model ---
    torch.manual_seed(args.seed)
    base_output = load_and_generate(
        args.base_model, args.prompt, device,
        args.max_new_tokens, args.temperature,
    )

    # --- Chat model ---
    torch.manual_seed(args.seed)
    chat_output = load_and_generate(
        args.chat_model, args.prompt, device,
        args.max_new_tokens, args.temperature,
    )

    # --- Display ---
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    print(f"\n[Base Model — {args.base_model}]")
    print(base_output)
    print(f"\n[Chat Model — {args.chat_model}]")
    print(chat_output)
    print("=" * 60)


if __name__ == "__main__":
    main()