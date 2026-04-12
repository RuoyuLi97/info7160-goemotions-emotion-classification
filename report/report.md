# Fine-grained Emotion Classification Evaluation

## 1. Evaluation Setup

The prediction outputs were originally in multi-hot vector format.
We converted them into label lists using the predefined label order before evaluation.
We evaluate two models on a multi-label emotion classification task using Micro F1, Macro F1, Precision, and Recall.

## 2. Results

| Model | Micro F1 | Macro F1 | Precision | Recall |
|-------|---------|---------|----------|--------|
| Baseline | 0.0 | 0.0 | 0.0 | 0.0 |
| BERT | 0.5119 | 0.2386 | 0.7275 | 0.3948 |

Analysis:
	•	BERT significantly outperforms the baseline.
	•	The baseline fails to learn meaningful patterns (all metrics are 0).
	•	BERT achieves high precision but relatively low recall, indicating it is
        conservative and misses some true labels.

## 3. Per-label Analysis

High-performing labels:
	•	gratitude (0.91)
	•	amusement (0.81)
	•	love (0.79)

Medium-performing labels:
	•	neutral, joy, curiosity

Low-performing labels (F1 = 0):
	•	caring, confusion, disappointment, fear, etc.

Observations:
	•	Frequent and explicit emotions are easier to classify.
	•	Rare or subtle emotions are harder for the model.

## 4. Error Analysis

Missing Predictions (Low Recall)

Example:
	•	True: [remorse]
	•	Predicted: []

The model fails to detect implicit emotions.

Misclassification

Example:
	•	True: [excitement]
	•	Predicted: [neutral]

Weak emotional signals are often classified as neutral.

Multi-label Errors

Example:
	•	True: [annoyance, disapproval]
	•	Predicted: [surprise]

The model struggles with multi-label classification.


## 5. Conclusion

•	BERT clearly outperforms the baseline.
•	However, recall remains relatively low.
•	The model performs well on frequent emotions but struggles with rare and subtle ones.