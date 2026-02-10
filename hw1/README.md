HW1: First LLM Script and Simple Evaluation

This LLM interaction is through HuggingFace without an auth key (okay for small usage like this). This codebase is meant for MacOS M1-arch running Python 3.11, other configurations have not been tested.

Create environment on Python 3.11 
```
python3.11 -m venv venv
```

`make sure you stay in the comp388 folder as paths are hardcoded.`

Install deps
```
pip install torch transformers datasets
# or pip install hw1/requirements.txt for a list of pinned packages
```

Use
```
python hw1/llm_prompt.py --prompt "word1 word2"
```
`I get a bus error on my mac for one word prompts`

Additional arguments
```
--model gpt2
--max_new_tokens 128
--temperature 0.7
--seed 0
```

Full run:

```
 python hw1/evaluate_wiki_tf.py --model gpt2       
Loading model: gpt2
Loading data from hw1/data/wiki_tf.jsonl
Loaded 10 examples

============================================================
Results on Wikipedia True/False Task
============================================================
Accuracy: 6/10 = 60.0%

Incorrect Examples (up to 5):
------------------------------------------------------------
ID: 4
Statement: Mars is larger than Earth in diameter.
Expected: false
Model Output: true

ID: 6
Statement: The entire Great Wall of China was built during the Ming Dynasty.
Expected: false
Model Output: true

ID: 8
Statement: Python coding language is named after the snake species python.
Expected: false
Model Output: true

ID: 10
Statement: Photosynthesis requires carbon dioxide and water but does not require sunlight.
Expected: false
Model Output: true

```

With GPT2, it seems like every time the model provided a wrong answer, it was the model confident in a correct-sounding fact was to be true when it wasn't. 