import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 1. Data Generation
np.random.seed(42)
x = np.random.uniform(0, 2 * np.pi, 1000)
y = np.cos(x) + np.random.normal(0, 0.1, 1000)

# Shuffle and split
indices = np.random.permutation(1000)
x, y = x[indices], y[indices]
x_train, y_train = x[:600], y[:600]
x_val,   y_val   = x[600:800], y[600:800]
x_test,  y_test  = x[800:], y[800:]

# 2. Model Definition - exact spec: Dense(16)->Dense(16)->Dense(1)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 3. Training - 200 epochs as per spec
history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1
)

# 4. Loss Curves
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('plots/loss_curve.png')
plt.close()
print("Loss curve saved.")

# 5. Evaluation
y_pred = model.predict(x_test, verbose=0).flatten()
mae = float(np.mean(np.abs(y_pred - y_test)))
mse_val = float(np.mean((y_pred - y_test) ** 2))
print(f"Test MSE: {mse_val:.4f}")
print(f"Test MAE: {mae:.4f}")

y_clean = np.cos(x_test)
mae_vs_cos = float(np.mean(np.abs(y_pred - y_clean)))
print(f"Test MAE vs cos(x) (no noise): {mae_vs_cos:.4f}")

# 6. Prediction Plot
x_dense = np.linspace(0, 2 * np.pi, 500)
y_true  = np.cos(x_dense)
y_float = model.predict(x_dense, verbose=0).flatten()

plt.figure()
plt.plot(x_dense, y_true,  label='Ground Truth cos(x)')
plt.plot(x_dense, y_float, label='Float Model Prediction', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Float Model vs Ground Truth')
plt.legend()
plt.savefig('plots/float_predictions.png')
plt.close()
print("Prediction plot saved.")

# 7. Save model and data
model.save('cosine_model.keras')
np.savez('data_splits.npz',
         x_train=x_train, y_train=y_train,
         x_val=x_val,     y_val=y_val,
         x_test=x_test,   y_test=y_test)
print("Model and data saved.")