HW1:


Create environment on Python 3.11
```
python3.11 -m venv venv
```

Install deps
```
pip install torch transformers datasets
```

Use
```
python llm_prompt.py --prompt "x"
```

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
Statement: Python is named after the snake species python.
Expected: false
Model Output: true

ID: 10
Statement: Photosynthesis requires carbon dioxide and water but does not require sunlight.
Expected: false
Model Output: true

```