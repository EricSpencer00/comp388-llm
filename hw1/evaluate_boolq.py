#!/usr/bin/env python3
"""
Evaluate LLM on BoolQ Dataset

Runs two experiments:
- Experiment A: With passage (reading comprehension)
- Experiment B: Without passage (parametric knowledge)
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
    """Extract yes/no from model output."""
    text = text.lower().strip()
    
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
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


def evaluate_experiment(model, tokenizer, device, examples, use_passage=True, experiment_name=""):
    """Evaluate model on BoolQ examples."""
    correct = 0
    incorrect_examples = []
    
    for idx, example in enumerate(examples):
        passage = example["passage"]
        question = example["question"]
        answer = "yes" if example["answer"] else "no"
        
        # Create appropriate prompt
        if use_passage:
            prompt = f"""Read the passage and answer the question.
Answer only with "yes" or "no".
Passage:
{passage}
Question:
{question}
Answer:"""
        else:
            prompt = f"""Answer the question.
Answer only with "yes" or "no".
Question:
{question}
Answer:"""
        
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
        
        if model_answer == answer:
            correct += 1
        else:
            if len(incorrect_examples) < 5:
                incorrect_examples.append({
                    "idx": idx,
                    "question": question,
                    "expected": answer,
                    "model_output": model_answer if model_answer else "(no valid answer)"
                })
    
    # Print results
    accuracy = correct / len(examples) * 100
    print("="*60)
    print(f"{experiment_name}")
    print("="*60)
    print(f"Accuracy: {correct}/{len(examples)} = {accuracy:.1f}%\n")
    
    if incorrect_examples:
        print("Incorrect Examples (up to 5):")
        print("-"*60)
        for example in incorrect_examples:
            print(f"Example #{example['idx']}")
            print(f"Question: {example['question']}")
            print(f"Expected: {example['expected']}")
            print(f"Model Output: {example['model_output']}")
            print()
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on BoolQ dataset")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name from Hugging Face Hub")
    parser.add_argument("--subset_size", type=int, default=100, help="Size of BoolQ subset to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load BoolQ dataset
    print("Loading BoolQ dataset...")
    dataset = load_dataset("boolq")
    examples = dataset["train"][:args.subset_size]
    
    print(f"Loaded {len(examples)} examples from BoolQ\n")
    
    # Run Experiment A: With passage
    print("Running Experiment A: With Passage (Reading Comprehension)")
    accuracy_with_passage = evaluate_experiment(
        model, tokenizer, device, examples, 
        use_passage=True, 
        experiment_name="Experiment A: With Passage"
    )
    
    # Run Experiment B: Without passage
    print("\nRunning Experiment B: Without Passage (Parametric Knowledge)")
    accuracy_without_passage = evaluate_experiment(
        model, tokenizer, device, examples,
        use_passage=False,
        experiment_name="Experiment B: Without Passage"
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"With Passage:    {accuracy_with_passage:.1f}%")
    print(f"Without Passage: {accuracy_without_passage:.1f}%")
    print(f"Difference:      {accuracy_with_passage - accuracy_without_passage:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
