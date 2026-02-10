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
from datasets import load_dataset
from llm_prompt import load_model_and_tokenizer, generate

def normalize_answer(text):
    """Extract yes/no from model output."""
    text = text.lower().strip()
    
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    else:
        return None


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
        generated_ids = generate(model, tokenizer, prompt, device, max_new_tokens=10)
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
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    
    # Load BoolQ dataset
    print("Loading BoolQ dataset...")
    dataset = load_dataset("boolq")
    examples = dataset["train"].select(range(min(len(dataset["train"]), args.subset_size)))
    
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
