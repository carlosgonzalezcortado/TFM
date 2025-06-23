# src/threshold_tuning.py
import os, json, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel
from tqdm import tqdm
from utils import load_tsv, tokenize
from sklearn.metrics import f1_score

def create_model(num_classes):
    base_model = TFAutoModel.from_pretrained("beomi/KcELECTRA-base")
    input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name='attention_mask')
    outputs = base_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = outputs[:, 0, :]
    x = tf.keras.layers.Dropout(0.1)(cls_token)
    logits = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-tsv', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--grid-start', type=float, required=True)
    parser.add_argument('--grid-end', type=float, required=True)
    parser.add_argument('--step', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    X_val, y_val = load_tsv(args.val_tsv)
    data = tokenize(X_val)
    num_classes = y_val.shape[1]
    # Get predictions from all models on validation set
    model_files = sorted([f for f in os.listdir(args.checkpoint-dir) if f.endswith('.h5') or f.endswith('.keras')])
    preds_list = []
    for f in tqdm(model_files, desc="Ensembling models", unit="model"):
        model_path = os.path.join(args.checkpoint_dir, f)
        model = create_model(num_classes)
        model.load_weights(model_path)
        preds = model.predict(data, batch_size=32, verbose=0)
        preds_list.append(preds)
    preds_arr = np.array(preds_list)
    ensemble_preds = np.mean(preds_arr, axis=0)
    # Grid search for best threshold per class
    thresholds = []
    for j in range(num_classes):
        true_labels = y_val[:, j]
        if np.sum(true_labels) == 0:
            best_thr = 1.0
        else:
            best_thr = args.grid_start
            best_f1 = 0.0
            thr = args.grid_start
            while thr <= args.grid_end + 1e-9:
                pred_bin = (ensemble_preds[:, j] >= thr).astype(int)
                f1 = f1_score(true_labels, pred_bin, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
                thr += args.step
        thresholds.append(best_thr)
    with open(args.output, 'w') as f:
        json.dump([float(t) for t in thresholds], f, indent=2)
