"""
Member 2: TFLite conversion, full int8 quantization, evaluation, and comparison plots.
Requires train_cosine.py outputs: cosine_model.keras, data_splits.npz
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = "cosine_model.keras"
DATA_PATH = "data_splits.npz"
FLOAT_TFLITE_PATH = "cosine_float.tflite"
INT8_TFLITE_PATH = "cosine_int8.tflite"
PLOTS_DIR = "plots"
REPRESENTATIVE_SAMPLES = 100


def load_artifacts():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_cosine.py first (Member 1)."
        )
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run train_cosine.py first (Member 1)."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    data = np.load(DATA_PATH)
    return model, data["x_train"], data["x_test"], data["y_test"]


def convert_float_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_int8_tflite(model, x_train):
    def representative_dataset():
        indices = np.linspace(0, len(x_train) - 1, REPRESENTATIVE_SAMPLES, dtype=int)
        for idx in indices:
            sample = np.array([[x_train[idx]]], dtype=np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def run_interpreter(tflite_bytes, x_inputs):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero = input_details[0]["quantization"]
    out_scale, out_zero = output_details[0]["quantization"]
    is_quantized = input_details[0]["dtype"] == np.int8

    predictions = []
    for x in x_inputs:
        x_arr = np.array([[x]], dtype=np.float32)
        if is_quantized:
            x_q = np.round(x_arr / in_scale + in_zero).astype(np.int8)
            interpreter.set_tensor(input_details[0]["index"], x_q)
        else:
            interpreter.set_tensor(input_details[0]["index"], x_arr.astype(np.float32))

        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        if is_quantized:
            out = (out.astype(np.float32) - out_zero) * out_scale
        predictions.append(float(out.flatten()[0]))

    return np.array(predictions), {
        "input_scale": in_scale,
        "input_zero_point": in_zero,
        "output_scale": out_scale,
        "output_zero_point": out_zero,
        "quantized": is_quantized,
    }


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model, x_train, x_test, y_test = load_artifacts()

    print("Converting Keras model to float TFLite...")
    float_tflite = convert_float_tflite(model)
    with open(FLOAT_TFLITE_PATH, "wb") as f:
        f.write(float_tflite)
    float_size = os.path.getsize(FLOAT_TFLITE_PATH)
    print(f"Saved {FLOAT_TFLITE_PATH} ({float_size:,} bytes)")

    print("Converting to full int8 TFLite (post-training quantization)...")
    int8_tflite = convert_int8_tflite(model, x_train)
    with open(INT8_TFLITE_PATH, "wb") as f:
        f.write(int8_tflite)
    int8_size = os.path.getsize(INT8_TFLITE_PATH)
    print(f"Saved {INT8_TFLITE_PATH} ({int8_size:,} bytes)")
    reduction = (1 - int8_size / float_size) * 100
    print(f"Size reduction (float -> int8): {reduction:.1f}%")

    print("\nEvaluating on held-out test set...")
    y_pred_float, _ = run_interpreter(float_tflite, x_test)
    y_pred_int8, int8_qparams = run_interpreter(int8_tflite, x_test)

    mae_float = np.mean(np.abs(y_pred_float - y_test))
    mae_int8 = np.mean(np.abs(y_pred_int8 - y_test))
    mse_int8 = np.mean((y_pred_int8 - y_test) ** 2)

    print(f"Float TFLite test MAE: {mae_float:.4f}")
    print(f"Int8 TFLite test MAE:  {mae_int8:.4f}")
    print(f"Int8 TFLite test MSE:  {mse_int8:.4f}")
    print(
        f"Int8 quantization params — "
        f"input scale={int8_qparams['input_scale']:.6f}, "
        f"input zero_point={int8_qparams['input_zero_point']}, "
        f"output scale={int8_qparams['output_scale']:.6f}, "
        f"output zero_point={int8_qparams['output_zero_point']}"
    )

    print("\nGenerating three-curve comparison plot...")
    x_dense = np.linspace(0, 2 * np.pi, 500)
    y_true = np.cos(x_dense)
    y_float_curve, _ = run_interpreter(float_tflite, x_dense)
    y_int8_curve, _ = run_interpreter(int8_tflite, x_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_true, label="Ground Truth cos(x)", linewidth=2)
    plt.plot(
        x_dense,
        y_float_curve,
        label="Float TFLite",
        linestyle="--",
        linewidth=1.5,
    )
    plt.plot(
        x_dense,
        y_int8_curve,
        label="Int8 TFLite",
        linestyle=":",
        linewidth=1.5,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ground Truth vs Float vs Int8 TFLite Predictions")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "quantization_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    with open(os.path.join(PLOTS_DIR, "quantization_metrics.txt"), "w") as f:
        f.write(f"cosine_float.tflite bytes: {float_size}\n")
        f.write(f"cosine_int8.tflite bytes: {int8_size}\n")
        f.write(f"size_reduction_percent: {reduction:.2f}\n")
        f.write(f"float_tflite_test_mae: {mae_float:.6f}\n")
        f.write(f"int8_tflite_test_mae: {mae_int8:.6f}\n")
        f.write(f"int8_tflite_test_mse: {mse_int8:.6f}\n")
        f.write(f"input_scale: {int8_qparams['input_scale']}\n")
        f.write(f"input_zero_point: {int8_qparams['input_zero_point']}\n")
        f.write(f"output_scale: {int8_qparams['output_scale']}\n")
        f.write(f"output_zero_point: {int8_qparams['output_zero_point']}\n")

    print("\nQuantization pipeline complete.")


if __name__ == "__main__":
    main()
