# Fine-grained Emotion Classification on GoEmotions

**Course:** INFO 7160 — Special Topics in Natural Language Engineering Methods and Tools
**Team:** Ruoyu Li, Jinru Zhang, Rongnan He, Chengcheng Gou

---

## Project Overview

This project compares a TF-IDF + Logistic Regression baseline against a BERT-based transformer for multi-label emotion classification on the GoEmotions dataset (28 labels, ~54k Reddit comments).

---

## Repository Structure

```
info7160-goemotions-emotion-classification/
├── src/                        # Shared utilities
│   ├── preprocessing.py        # Tokenizer, label binarizer
│   └── metrics.py              # Micro/macro F1, precision, recall
├── notebooks/
│   ├── baseline.ipynb          # TF-IDF + Logistic Regression
│   ├── bert_finetune.ipynb     # BERT fine-tuning
│   └── evaluation.ipynb        # Evaluation and error analysis
├── outputs/
│   ├── baseline_preds.jsonl    # Baseline predictions
│   └── bert_preds.jsonl        # BERT predictions
└── report/                     # Final PDF report
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/info7160-goemotions-emotion-classification.git
cd info7160-goemotions-emotion-classification
```

### 2. Install dependencies
```bash
pip install datasets transformers torch scikit-learn numpy reportlab
```

### 3. Load the dataset
```python
from datasets import load_dataset
dataset = load_dataset("go_emotions")
```

---

## Shared Utilities

### `src/preprocessing.py`
```python
from src.preprocessing import get_tokenizer, tokenize, binarize_labels, LABEL_NAMES, NUM_LABELS

tokenizer = get_tokenizer()
encoded = tokenize(["I am so happy"], tokenizer)
vector = binarize_labels([2, 14])  # returns 28-dim binary array
```

### `src/metrics.py`
```python
from src.metrics import compute_metrics, compute_per_label_f1
from src.preprocessing import LABEL_NAMES

metrics = compute_metrics(true_labels, pred_labels)
per_label = compute_per_label_f1(true_labels, pred_labels, LABEL_NAMES)
```

---

## Prediction File Format

One JSON object per line:
```json
{"id": "abc123", "text": "I love this!", "true_labels": ["admiration", "joy"], "pred_labels": ["admiration"]}
```