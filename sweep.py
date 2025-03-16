import json
import numpy as np
from keras.datasets import fashion_mnist
import wandb
import importlib
import model  # Ensure model.py is imported
importlib.reload(model)  # Reload the module
from model import Model  # Import the class
import numpy as np

# Load the config.json file
CONFIG_FILE = "sweep_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Construct the sweep_config dictionary using values from config.json
sweep_config = {
    "name": config["name"],
    "method": "grid",  # You can change to "random" or "bayes" if needed
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": config["epochs"]},
        "num_hidden_layers": {"values": config["num_hidden_layers"]},
        "hidden_layer_size": {"values": config["hidden_layer_size"]},
        "weight_decay": {"values": config["weight_decay"]},
        "learning_rate": {"values": config["learning_rate"]},
        "optimizer": {"values": config["optimizer"]},
        "batch_size": {"values": config["batch_size"]},
        "weight_initialization": {"values": config["weight_initialization"]},
        "activation_functions": {"values": config["activation_functions"]}
    }
}


sweep_id = wandb.sweep(sweep_config, project="DL")

if config["data"] == "fashion_mnist":
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
elif config["data"] == "mnist":
    from keras.datasets import mnist 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
X_train = X_train/255.0
X_train = np.array([x.reshape(-1, 1) for x in X_train])

num_classes = int(np.max(y_train)) + 1  # Convert to int

y_train = np.eye(num_classes)[y_train.astype(int)]  # Ensure indices are integers
y_train = np.array([each_y.reshape(-1, 1) for each_y in y_train])

val_ratio = 0.1
num_samples = X_train.shape[0]
num_val = int(num_samples * val_ratio)

indices = np.arange(num_samples)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)

# Split the dataset
X_val, y_val = X_train[indices[:num_val]], y_train[indices[:num_val]]
X_train, y_train = X_train[indices[num_val:]], y_train[indices[num_val:]]


def train():
    run = wandb.init(project="DL")  # Initialize first

    # Now safely access wandb.config parameters
    run_name = f"hl_{run.config.num_hidden_layers}_hs_{run.config.hidden_layer_size}_bs_{run.config.batch_size}_ac_{run.config.activation_functions}_wd_{run.config.weight_decay}_lr_{run.config.learning_rate}_opt_{run.config.optimizer}_wi_{run.config.weight_initialization}_ep_{run.config.epochs}"
    run.name = run_name  # Set the dynamically generated name
    # Load the hyperparameters from WandB
    config = wandb.config

    model = Model(
        num_hidden_layers=config.num_hidden_layers,
        hidden_layer_size=[config.hidden_layer_size] * config.num_hidden_layers,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        activation=config.activation_functions,
        weight_init=config.weight_initialization,
        loss="cross-entropy",
    )

    # Train the model
    model.train(X_train, y_train, config.epochs, config.batch_size)

    # Get predictions
    y_hat = model.predict(X_val)

    # Compute accuracy
    accuracy  = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y_val, axis=1))

    # Log metrics
    wandb.log({"accuracy": accuracy})

    run.finish()

wandb.agent(sweep_id, train)
