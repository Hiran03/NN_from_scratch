# Fully Functional Artificial Neural Network from Scratch

This repository contains a custom-built **Artificial Neural Network (ANN)** module implemented from scratch. The model supports various optimizers, activation functions, weight initialization techniques, and hyperparameter tuning using **Weights & Biases (WandB)**.

Follow compilation.ipynb for grading

wandb report can be found at https://api.wandb.ai/links/mm21b030-indian-institute-of-technology-madras/94rs04sq

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



### A. Running a Sweep using `sweep.py`

#### 1. Configure the Sweep
Modify `sweep_config.json` to specify the hyperparameters for tuning.

#### 2. Run the Sweep
```bash
python sweep.py
```
This will initialize and run the hyperparameter tuning process using Weights & Biases (W&B).

---

### B. Running `train.py` from Command Line

To train the model manually, use `train.py` with the following command-line arguments.

#### Accepted Arguments
The table below describes the available arguments, their accepted values, and default settings:

| Tag | Argument | Accepted Values | Default |
|------|----------|----------------|---------|
| `-wp` | `--wandb_project` | Any string | `DL` |
| `-we` | `--wandb_entity` | Any string | `mm21b030-indian-institute-of-technology-madras` |
| `-d` | `--dataset` | `mnist`, `fashion_mnist` | `fashion_mnist` |
| `-e` | `--epochs` | Any integer | `10` |
| `-b` | `--batch_size` | Any integer | `128` |
| `-l` | `--loss` | `MSE`, `cross-entropy` | `cross-entropy` |
| `-o` | `--optimizer` | `SGD`, `momentum`, `nesterov`, `RMSprop`, `Adam` | `RMSprop` |
| `-lr` | `--learning_rate` | Any float | `0.001` |
| `-m` | `--momentum` | Any float | `0` |
| `-beta` | `--beta` | Any float | `0.9` |
| `-beta1` | `--beta1` | Any float | `0.9` |
| `-beta2` | `--beta2` | Any float | `0.999` |
| `-eps` | `--epsilon` | Any float | `0.000001` |
| `-w_d` | `--weight_decay` | Any float | `0.0005` |
| `-w_i` | `--weight_init` | `random`, `xavier` | `xavier` |
| `-nhl` | `--num_layers` | Any integer | `3` |
| `-sz` | `--hidden_size` | Any integer | `128` |
| `-a` | `--activation` | `sigmoid`, `tanh`, `relu` | `tanh` |

#### Example Usage
Run training with custom parameters:
```bash
python train.py -e 20 -b 64 -lr 0.0005 -o Adam -a relu
```
This command sets epochs to 20, batch size to 64, learning rate to 0.0005, optimizer to Adam, and activation function to ReLU.

For more details, use:
```bash
python train.py --help
```


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

