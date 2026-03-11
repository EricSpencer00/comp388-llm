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

*(Fill in after running)*

- **Accuracy:** __%
- **Unparseable outputs:** __

### Typical Errors

*(Fill in after running — e.g., the model confuses "neutral" with
"entailment" when the hypothesis adds information not explicitly
contradicted by the premise.)*

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

*(Fill in after running)*

| Configuration            | Accuracy |
|--------------------------|----------|
| Chat model (zero-shot)   |    __%   |
| Base model (zero-shot)   |    __%   |
| Base model (few-shot)    |    __%   |

### Discussion

*(Fill in 4–6 sentences after running. Address why instruction-tuned models
perform better zero-shot and how few-shot examples change base model
behavior.)*

Instruction-tuned models perform better in zero-shot settings because the
fine-tuning stage teaches them to follow natural-language instructions and
produce structured, label-like answers. Without that fine-tuning, a base
model treats the prompt as ordinary text to continue, often producing
irrelevant completions or failing to match the expected label format.

Few-shot examples help the base model by providing an in-context pattern that
constrains its output distribution — the model learns from the examples that
it should respond with one of three labels. This effectively replaces the
instruction-following behavior that the chat model acquired through
fine-tuning, although accuracy may still lag behind because the base model
has not been directly optimized for helpfulness and instruction compliance.
