I used Qwen 2.5-0.5B since it was small enough for my laptop

The zero-shot model performs better than the instruction-tuned model. This should not be the case. This means that a quantized model that fits in 0.5B could already be sufficient for tasks and any new information may overload it. 

Few-shot models sometimes over adhere to their purpose and "overfit" their responses to the instructions. This can lead to poorer accuracy.

