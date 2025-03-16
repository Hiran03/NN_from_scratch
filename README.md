# Fully Functional Artificial Neural Network from Scratch

This repository contains a custom-built **Artificial Neural Network (ANN)** module implemented from scratch. The model supports various optimizers, activation functions, weight initialization techniques, and hyperparameter tuning using **Weights & Biases (WandB)**.

## Installation

Before running the project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Load & Preprocess the Dataset

Ensure that **X and y** are column vectors. If **X** is an image-like 2D array, flatten it before training.

#### Example (Using Fashion MNIST):

```python
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape images into column vectors
X_train = np.array([x.reshape(-1, 1) for x in X_train])
X_test = np.array([x.reshape(-1, 1) for x in X_test])

# One-hot encode labels
num_classes = int(np.max(y_train)) + 1  
y_train = np.eye(num_classes)[y_train.astype(int)]
y_train = np.array([y.reshape(-1, 1) for y in y_train])
y_test = np.eye(num_classes)[y_test.astype(int)]
y_test = np.array([y.reshape(-1, 1) for y in y_test])
```

---

### 2. Initialize and Train the Model

```python
import numpy as np
from model import Model  # Ensure model.py is in the same directory

# Initialize the model
model = Model(
    num_hidden_layers=2,
    hidden_layer_size=[256, 256],  # Example: Two hidden layers of size 256 each
    weight_decay=0.0005,
    learning_rate=0.001,
    optimizer="Adam",  # Available options: 'SGD', 'momentum', 'nesterov', 'RMSprop'
    activation="tanh",  # Available options: 'sigmoid', 'relu'
    weight_init="xavier",
    loss="cross-entropy"
)

# Train the model
model.train(X_train, y_train, epochs=10, batch_size=64)

# Make predictions
y_hat = model.predict(X_test)

# Compute accuracy
test_acc = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_acc:.4f}")
```

---

### 3. Save & Reload the Model

#### Save the model:

```python
model.save("test_model")
```

#### Reload the model:

```python
loaded_model = Model()
loaded_model = loaded_model.reload("test_model")
```

---

## Hyperparameter Tuning with WandB

### Running Sweeps

#### 1. Configure the Sweep

Modify `sweep_config.json` to specify the hyperparameters for tuning.

#### 2. Run the sweep:

```bash
python sweep.py
```

---

## Additional Files

- **`sweep_config.json`** – Configuration file for running WandB sweeps.
- **`sweep.py`** – Script for hyperparameter tuning using WandB.
- **`train.py`** – Script to initiate training from the command line.
- **`Compilation.ipynb`** – Contains the assignment report.
- **`activations.py`** – Contains the activation functions for model.py.


---

## Notes

- This implementation is **built from scratch**, without using deep learning libraries like TensorFlow or PyTorch.


If you find this project useful, feel free to ⭐ the repository!

