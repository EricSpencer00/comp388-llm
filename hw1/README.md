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

BoolQ results:

```
 python hw1/evaluate_boolq.py --subset_size 20
Loading model: gpt2
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 148/148 [00:00<00:00, 6736.38it/s, Materializing param=
GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading BoolQ dataset...
Loaded 20 examples from BoolQ

Running Experiment A: With Passage (Reading Comprehension)
============================================================
Experiment A: With Passage
============================================================
Accuracy: 9/20 = 45.0%

Incorrect Examples (up to 5):
------------------------------------------------------------
Example #0
Question: do iran and afghanistan speak the same language
Expected: yes
Model Output: (no valid answer)

Example #4
Question: is elder scrolls online the same as skyrim
Expected: no
Model Output: yes

Example #5
Question: can you use oyster card at epsom station
Expected: no
Model Output: yes

Example #6
Question: will there be a season 4 of da vinci's demons
Expected: no
Model Output: (no valid answer)

Example #7
Question: is the federal court the same as the supreme court
Expected: no
Model Output: (no valid answer)


Running Experiment B: Without Passage (Parametric Knowledge)
============================================================
Experiment B: Without Passage
============================================================
Accuracy: 9/20 = 45.0%

Incorrect Examples (up to 5):
------------------------------------------------------------
Example #4
Question: is elder scrolls online the same as skyrim
Expected: no
Model Output: yes

Example #5
Question: can you use oyster card at epsom station
Expected: no
Model Output: yes

Example #6
Question: will there be a season 4 of da vinci's demons
Expected: no
Model Output: yes

Example #7
Question: is the federal court the same as the supreme court
Expected: no
Model Output: yes

Example #10
Question: is a wolverine the same as a badger
Expected: no
Model Output: yes


============================================================
Summary
============================================================
With Passage:    45.0%
Without Passage: 45.0%
Difference:      0.0%
============================================================

```

With GPT2, it seems like it was always giving a Yes to every fact.