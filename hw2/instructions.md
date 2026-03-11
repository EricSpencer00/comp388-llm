Homework 2: Few-Shot Learning and 
Base vs Instruction-Tuned LLMs
Overview
In this homework, you will explore few-shot (in-context) learning and the differences 
between base language models and instruction-tuned (chat) models.
You will:
1. Compare the behavior of a base LLM and its instruction-tuned counterpart.
2. Evaluate an instruction-tuned model on a simple NLP classification task.
3. Repeat the evaluation using the base model and improve its performance using 
few-shot prompting.
You are free to choose any publicly available open-source LLM that provides:
a base model, and
a corresponding instruction-tuned / chat model.
Because you are expected to run the models on your own laptop or Google Colab, you 
should choose a small model (around 1 billion parameters or less), or a quantized 
version.
Example model families that satisfy this requirement include small LLaMA-style or 
similar models.
1. Base Model vs Instruction-Tuned Model
Task
Pick a small LLM that has:
a base (pretrained) model, and
an instruction-tuned (chat) version.
Write a Python script called compare_base_chat.py that:
1. Sends the same prompt to both models.
2. Prints both outputs clearly labeled.
Example prompt
Requirements
Your script must:
Load both models from Hugging Face.
Use the same decoding parameters for both models.
Clearly show the difference in outputs.
Required command-line arguments
--base_model (string)
--chat_model (string)
--prompt (string, required)
--max_new_tokens (int, default: 64)
--temperature (float, default: 0.7)
--seed (int, default: 0)
Set the random seed for reproducibility.
Deliverables (Part 1)
compare_base_chat.py
In README.md: the models you chose and a short description (2–3 sentences) of 
how the outputs differ.
What is the capital of France?
2. Evaluation with an Instruction-Tuned Model
Task
Choose one of the following evaluation tasks:
Paraphrase Detection, or
Natural Language Inference (Entailment)
Public datasets you may use include:
Paraphrase Detection (e.g., Microsoft Research Paraphrase Corpus):
https://www.microsoft.com/en-us/download/details.aspx?id=52398
Natural Language Inference (SNLI):
https://nlp.stanford.edu/projects/snli/
You may use a small subset of the dataset (e.g., 50–200 examples) if your hardware 
does not allow full evaluation.
Evaluation Script
Write a script called evaluate_chat_model.py that:
loads a subset of your chosen dataset,
uses the instruction-tuned / chat model,
prompts the model to output a classification label,
computes accuracy.
Example: Paraphrase Detection Prompt
Do the following two sentences have the same meaning?
Answer with only “yes” or “no”.
Sentence 1: {sentence1}
Example: Entailment Prompt
Deliverables (Part 2)
evaluate_chat_model.py
In README.md:
dataset chosen and subset size,
decoding settings used for evaluation,
accuracy score,
brief comment on typical model errors.
3. Few-Shot Learning with the Base Model
Task
Sentence 2: {sentence2}
Answer:
Determine the relationship between the premise and the hypothesis.
Answer with only one of: entailment, contradiction, neutral.
Premise: {premise}
Hypothesis: {hypothesis}
Answer:
Now replace the instruction-tuned model with the base model.
1. First, attempt to run the same evaluation prompt from Part 2 using the base model 
without any examples.
2. Observe whether the evaluation still works (e.g., formatting issues, wrong labels, 
poor accuracy).
If the evaluation does not work well:
Modify your prompt to include few-shot examples (in-context learning).
Use 2–5 labeled examples at the top of the prompt.
Few-Shot Prompt Example (Paraphrase Detection)
Do the following two sentences have the same meaning?
Answer with only “yes” or “no”.
Sentence 1: The cat sat on the mat.
Sentence 2: A cat was sitting on a mat.
Answer: yes
Sentence 1: The sky is blue.
Sentence 2: Grass is green.
Answer: no
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Evaluation and Comparison
Write a script called evaluate_base_fewshot.py that:
evaluates the base model with few-shot prompting,
computes accuracy,
uses the same dataset subset as Part 2.
Compare:
chat model (zero-shot),
base model (zero-shot, if applicable),
base model (few-shot).
Deliverables (Part 3)
evaluate_base_fewshot.py
In README.md:
description of the few-shot examples used,
accuracy of the base model with few-shot prompting,
comparison to the chat model’s accuracy,
4–6 sentences reflecting on:
why instruction-tuned models perform better zero-shot,
how few-shot examples change base model behavior.
Submission Checklist
Your submission should include:
Answer:
compare_base_chat.py
evaluate_chat_model.py
evaluate_base_fewshot.py
README.md with commands, results, and discussion
Notes
Few-shot prompting is not training — you are not updating model weights.
Small models may be sensitive to prompt wording; this is expected.
Clear prompts and careful evaluation matter more than high accuracy.
For evaluation, using temperature = 0.0 is strongly r