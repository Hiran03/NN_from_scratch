import argparse
import wandb
import numpy as np
from model import Model  # Import the Model class from model.py

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with W&B sweeps")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name for W&B")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="W&B Entity")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for rmsprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for adam/nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for adam/nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Size of hidden layers")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")
    
    return parser.parse_args()

def train_model():
    wandb.init()
    config = wandb.config
    
    model = Model(
        num_hidden_layers=config.num_layers,
        hidden_layer_size=config.hidden_size,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        activation=config.activation,
        weight_init=config.weight_init,
        loss=config.loss
    )
    
    X_train, y_train = np.random.randn(1000, 20), np.random.randint(0, 2, size=1000)
    
    for epoch in range(config.epochs):
        loss = model.train(X_train, y_train, batch_size=config.batch_size)
        wandb.log({"loss": loss, "epoch": epoch})

def main():
    args = get_args()
    
    wandb.login()
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": vars(args)  # Pass all parsed arguments as sweep parameters
    }
    
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, train_model, count=20)

if __name__ == "__main__":
    main()