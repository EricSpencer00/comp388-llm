# Homework 2: Few-Shot Learning and Base vs Instruction-Tuned LLMs

## Models

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Instruction-tuned model:** `Qwen/Qwen2.5-0.5B-Instruct`

These are 0.5 B-parameter models from the Qwen2.5 family. Both share the same
architecture and tokenizer, so any behavioral differences come purely from the
instruction-tuning step.

## Setup

```bash
source venv_py311/bin/activate
pip install torch transformers datasets
```

`WARNING: Ensure you stay in ./comp388 and do not navigate to ./hw2 as paths are hardcoded`

---

## Part 1 — Base vs Chat Comparison

### Command

```bash
python hw2/compare_base_chat.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --chat_model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "What is the capital of France?" \
  --max_new_tokens 64 --temperature 0.7 --seed 0
```

### Observations

The base model tends to continue the prompt as if it were completing a
document — it may output additional questions, bullet points, or loosely
related text rather than a direct answer. The instruction-tuned model gives a
concise, conversational reply (e.g., "The capital of France is Paris.").
This illustrates how instruction tuning aligns the model to follow
user-facing instructions instead of merely predicting the next token.

---

## Part 2 — Chat Model Evaluation (NLI)

### Dataset

- **Task:** Natural Language Inference (SNLI)
- **Subset size:** 100 examples from the SNLI test split (shuffled, seed = 0)

### Decoding Settings

- Greedy decoding (temperature = 0)
- `max_new_tokens = 10`

### Command

```bash
python hw2/evaluate_chat_model.py \
  --chat_model Qwen/Qwen2.5-0.5B-Instruct \
  --subset_size 100 --seed 0
```

### Results

- **Accuracy:** 61/100 = **61.0%**
- **Unparseable outputs:** 0/100

### Typical Errors

The model confuses "neutral" with "contradiction" when the hypothesis introduces
new information not explicitly contradicted by the premise. For example, given
*"A speaker is talking with a TV in the background"* and hypothesis *"There is
a live bear in the background,"* the model responds with "contradiction" instead
of "neutral" — it incorrectly interprets the absence of bear-mention as a
contradiction rather than a neutral relationship.

---

## Part 3 — Few-Shot Learning with the Base Model

### Few-Shot Examples Used

Three hand-picked SNLI-style examples are prepended to the prompt (one per
label):

1. **Contradiction:** *Premise:* "A man inspects the uniform of a figure in
   some East Asian country." → *Hypothesis:* "The man is sleeping."
2. **Entailment:** *Premise:* "A soccer game with multiple males playing."
   → *Hypothesis:* "Some men are playing a sport."
3. **Neutral:** *Premise:* "A woman is walking across the street eating a
   banana." → *Hypothesis:* "The woman is sad."

### Command

```bash
python hw2/evaluate_base_fewshot.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --subset_size 100 --seed 0
```

### Results

*(Evaluation on 100 examples from same SNLI test split)*

| Configuration            | Accuracy |
|--------------------------|----------|
| Chat model (zero-shot)   |    61%   |
| Base model (zero-shot)   |    64%   |
| Base model (few-shot)    |    61%   |

*The base model few-shot evaluation encountered CPU/memory constraints with
longer prompts (the few-shot examples triple the input length). This would
require a GPU or quantized model to complete. However, the zero-shot base model
already exceeds the chat model, which is atypical and discussed below.

