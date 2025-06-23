# src/export_deploy.py
import tensorflow as tf
import argparse
import tf2onnx
import onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model', required=True)
    parser.add_argument('--to', choices=['onnx', 'tflite'], required=True)
    args = parser.parse_args()

    base = args.input_model.rstrip('/')
    if '.' in base:
        base = base[:base.rfind('.')]
    if args.to.lower() == 'onnx':
        out_path = base + '.onnx'
        model = tf.keras.models.load_model(args.input_model, compile=False)
        seq_len = model.inputs[0].shape[1] if model.inputs[0].shape[1] is not None else 512
        input_signature = [
            tf.TensorSpec([None, seq_len], tf.int32, name="input_ids"),
            tf.TensorSpec([None, seq_len], tf.int32, name="attention_mask")
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        onnx.save(onnx_model, out_path)
    elif args.to.lower() == 'tflite':
        out_path = base + '.tflite'
        model = tf.keras.models.load_model(args.input_model, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
