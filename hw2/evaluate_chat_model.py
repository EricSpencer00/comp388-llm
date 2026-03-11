#!/usr/bin/env python3
"""
evaluate_chat_model.py

Evaluate an instruction-tuned (chat) model on the SNLI natural language
inference task (entailment / contradiction / neutral).
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def normalize_answer(text):
    """Extract entailment/contradiction/neutral from model output."""
    text = text.lower().strip()
    for label in ["entailment", "contradiction", "neutral"]:
        if label in text:
            return label
    return None


def generate_greedy(model, tokenizer, input_ids, attention_mask, device,
                    max_new_tokens=10):
    """Greedy (temperature=0) token-by-token generation."""
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=generated, attention_mask=attention_mask)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device)], dim=-1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


def build_prompt(premise, hypothesis):
    """Zero-shot NLI prompt for an instruction-tuned model."""
    return (
        "Determine the relationship between the premise and the hypothesis.\n"
        'Answer with only one of: entailment, contradiction, neutral.\n\n'
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a chat model on SNLI (NLI)"
    )
    parser.add_argument("--chat_model", type=str, required=True,
                        help="HuggingFace instruction-tuned model name")
    parser.add_argument("--subset_size", type=int, default=100,
                        help="Number of SNLI examples to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model: {args.chat_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.chat_model)
    model = AutoModelForCausalLM.from_pretrained(args.chat_model,
                                                  low_cpu_mem_usage=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    # Load SNLI dataset — keep only examples with valid labels (0, 1, 2)
    print("Loading SNLI dataset...")
    dataset = load_dataset("stanfordnlp/snli", split="test")
    dataset = dataset.filter(lambda x: x["label"] in [0, 1, 2])
    dataset = dataset.shuffle(seed=args.seed).select(range(args.subset_size))
    print(f"Evaluating on {len(dataset)} examples\n")

    correct = 0
    unparseable = 0
    incorrect_examples = []

    for idx, example in enumerate(dataset):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        expected = LABEL_MAP[example["label"]]

        prompt = build_prompt(premise, hypothesis)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids).to(device)

        output = generate_greedy(model, tokenizer, input_ids, attention_mask,
                                 device, max_new_tokens=10)
        generated_text = tokenizer.decode(output[0, prompt_len:],
                                          skip_special_tokens=True)
        model_answer = normalize_answer(generated_text)

        if model_answer is None:
            unparseable += 1
        if model_answer == expected:
            correct += 1
        elif len(incorrect_examples) < 5:
            incorrect_examples.append({
                "idx": idx,
                "premise": premise,
                "hypothesis": hypothesis,
                "expected": expected,
                "model_output": model_answer if model_answer else f"(unparseable: {generated_text!r})",
            })

    accuracy = correct / len(dataset) * 100
    print("=" * 60)
    print("Chat Model Zero-Shot NLI Evaluation")
    print("=" * 60)
    print(f"Model:       {args.chat_model}")
    print(f"Subset size: {len(dataset)}")
    print(f"Accuracy:    {correct}/{len(dataset)} = {accuracy:.1f}%")
    print(f"Unparseable: {unparseable}/{len(dataset)}")
    print()

    if incorrect_examples:
        print("Incorrect Examples (up to 5):")
        print("-" * 60)
        for ex in incorrect_examples:
            print(f"#{ex['idx']}  Expected: {ex['expected']}  "
                  f"Got: {ex['model_output']}")
            print(f"  Premise:    {ex['premise']}")
            print(f"  Hypothesis: {ex['hypothesis']}")
            print()


if __name__ == "__main__":
    main()