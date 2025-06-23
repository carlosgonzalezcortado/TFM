# src/train_final.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel
from utils import load_tsv, tokenize

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-tsv', required=True)
    parser.add_argument('--val-tsv', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Load and combine training and validation data
    X_train, y_train = load_tsv(args.train_tsv)
    X_val, y_val = load_tsv(args.val_tsv)
    if X_val:
        X_train += X_val
        y_train = np.vstack([y_train, y_val]) if y_val.size else y_train

    data = tokenize(X_train)
    num_classes = y_train.shape[1] if y_train.size else 0

    # Build model (KcELECTRA base + dropout + classification head)
    base_model = TFAutoModel.from_pretrained("beomi/KcELECTRA-base")
    input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name='attention_mask')
    outputs = base_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = outputs[:, 0, :]
    x = tf.keras.layers.Dropout(0.1)(cls_token)
    logits = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.fit(data, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
    model.save(args.output)
