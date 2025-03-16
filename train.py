import argparse
import wandb
import numpy as np
from model import Model  # Import the Model class from model.py

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with W&B sweeps")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL", help="Project name for W&B")
    parser.add_argument("-we", "--wandb_entity", type=str, default="mm21b030-indian-institute-of-technology-madras", help="W&B Entity")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["MSE", "cross-entropy"], default="cross-entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["SGD", "momentum", "nesterov", "RMSprop", "Adam"], default="RMSprop", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for rmsprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for adam/nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for adam/nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="xavier", help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Size of hidden layers")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "relu"], default="tanh", help="Activation function")
    
    return parser.parse_args()

args = get_args()

# Initialize WandB run
wandb.init(
    project=args.wandb_project, 
    name="Run", 
    entity=args.wandb_entity,
    config={"epochs": args.epochs, "batch_size": args.batch_size}
)

if args.dataset == "fashion_mnist":
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
elif args.dataset == "mnist":
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
model = Model(
    num_hidden_layers=args.num_layers,
    hidden_layer_size=[args.hidden_size]*args.num_layers,
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    optimizer=args.optimizer,
    activation=args.activation,
    weight_init=args.weight_init,
    loss=args.loss,
    beta=args.beta,
    beta1=args.beta1,
    beta2=args.beta2,
    epsilon=args.epsilon
)


# Train the model
model.train(X_train, y_train, args.epochs, args.batch_size)

# Get predictions
y_hat = model.predict(X_val)

# Compute accuracy
accuracy  = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y_val, axis=1))

# Log metrics
wandb.log({"accuracy": accuracy})

wandb.finish() 
