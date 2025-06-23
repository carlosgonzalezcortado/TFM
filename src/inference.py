# src/inference.py
import os, json, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel
from tqdm import tqdm
from utils import load_tsv, tokenize, compute_metrics

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
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--test-tsv', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--method', choices=['mean', 'weighted'], default='mean')
    args = parser.parse_args()

    X_test, y_test = load_tsv(args.test_tsv)
    data = tokenize(X_test)
    num_classes = y_test.shape[1] if y_test.size else 0

    model_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.h5') or f.endswith('.keras')])
    preds_list = []
    for f in tqdm(model_files, desc="Ensembling models", unit="model"):
        model_path = os.path.join(args.checkpoint_dir, f)
        model = create_model(num_classes)
        model.load_weights(model_path)
        preds = model.predict(data, batch_size=32, verbose=0)
        preds_list.append(preds)
    preds_arr = np.array(preds_list)  # shape: (num_models, num_samples, num_classes)
    if args.method == 'mean':
        ensemble_preds = np.mean(preds_arr, axis=0)
    else:
        weights = np.arange(1, len(preds_list) + 1, dtype=np.float32)
        weights /= np.sum(weights)
        ensemble_preds = np.average(preds_arr, axis=0, weights=weights)
    metrics = compute_metrics(y_test, ensemble_preds)
    with open(args.output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
