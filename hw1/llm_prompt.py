"""
loads an LLM and tokenizer from HuggingFace

--model: string, default allowed
--prompt: string, required
--max_new_tokens: int, default 128
--temperature: float, default 0.7
--seed: int, default 0
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def main():
    parser = argparse.ArgumentParser(description="Load an LLM and generate text based on a prompt.")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name or path from HuggingFace')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text to generate from')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Tokenize input prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')

    # Generate text
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()