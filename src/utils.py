# src/utils.py
import numpy as np
import csv
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoTokenizer

MAX_LEN = 512

def load_tsv(path):
    texts = []
    labels_list = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    if not rows:
        return [], np.array([])
    # Skip header if present (if second column is not numeric)
    if len(rows[0]) > 1 and not rows[0][1].isdigit():
        rows = rows[1:]
    for row in rows:
        if len(row) < 2:
            continue
        # Determine if an ID column exists (first column numeric and more than 2 columns total)
        if len(row) > 2 and row[0].isdigit():
            text = row[1]
            label_vals = row[2:]
        else:
            text = row[0]
            label_vals = row[1:]
        try:
            labels = [int(x) for x in label_vals]
        except:
            labels = []
        if labels:
            texts.append(text)
            labels_list.append(labels)
    labels_array = np.array(labels_list, dtype=int) if labels_list else np.array(labels_list)
    return texts, labels_array

def tokenize(texts, max_len=MAX_LEN):
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='np',
        return_token_type_ids=False
    )
    input_ids = enc['input_ids'].astype(np.int32)
    attention_mask = enc['attention_mask'].astype(np.int32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def compute_metrics(y_true, y_score, thresholds=None):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    if thresholds is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        thresholds = np.array(thresholds)
        if thresholds.ndim == 1:
            y_pred = (y_score >= thresholds).astype(int)
        else:
            y_pred = (y_score >= thresholds).astype(int)
    # Compute AUC (macro-average)
    try:
        auc = roc_auc_score(y_true, y_score, average='macro')
    except ValueError:
        # Handle case where a class has only one label in y_true
        auc_vals = []
        for j in range(y_true.shape[1]):
            if len(np.unique(y_true[:, j])) == 2:
                auc_vals.append(roc_auc_score(y_true[:, j], y_score[:, j]))
        auc = float(np.mean(auc_vals)) if auc_vals else None
    # Compute macro F1-score
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'auc': float(auc) if auc is not None else None, 'f1': float(f1)}
