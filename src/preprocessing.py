from transformers import BertTokenizer
import numpy as np

LABEL_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]

NUM_LABELS = len(LABEL_NAMES)


def get_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)


def tokenize(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def binarize_labels(label_indices, num_labels=NUM_LABELS):
    vector = np.zeros(num_labels, dtype=np.float32)
    for idx in label_indices:
        vector[idx] = 1.0
    return vector