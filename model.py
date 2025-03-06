# This file contains the code for forward pass and backpropogation, activation functions
import numpy as np
import random

class Model:
    def __init__(self, num_hidden_layers, hidden_layer_size, weight_decay, learning_rate, optimizer, activation, weight_init):
        # Initialize parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation
        self.weight_init = weight_init

        # Initialize weights & biases
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self, input_size, output_size = 10):
        """Initialize weights based on weight_init strategy."""
        self.weights = []
        self.neuron_outputs = [input_size]

        if self.weight_init == 'random':
            prev_layer = input_size
            for i in range(self.num_hidden_layers):
                self.weights.append(np.random.rand(prev_layer, self.hidden_layer_size[i]))  # Uniform distribution from zero to one
                self.neuron_outputs.append(np.zeros(self.hidden_layer_size[i]))
                prev_layer = self.hidden_layer_size[i]
            self.weights.append(np.random.rand(prev_layer, output_size))
            self.neuron_outputs.append(np.zeros(output_size))
            self.weights = np.array(self.weights)
            
    def initialize_biases(self):
        """Initialize biases to zeros."""
        self.biases = np.zeros(self.num_hidden_layers + 2)
        
        

    def activation_function(self, x):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.activation == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 
        if self.activation == 'relu':
            return np.maximum(0,x)

        
    def activation_derivative(self, x):
        """Compute the derivative for backpropagation."""
        if self.activation == "sigmoid":
            sig = self.activation_function(x)  
            return sig * (1 - sig)  
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)  # ReLU derivative
        elif self.activation == "tanh":
            tanh_x = self.activation_function(x) 
            return 1 - tanh_x ** 2  


    def forward(self, x):
        """Forward pass through the network."""
        
        pass  # Implement forward propagation

    def backward(self, loss):
        """Backward pass for weight updates."""
        pass  # Implement backpropagation

    def update_weights(self):
        """Apply optimization algorithm to update weights."""
        pass  # Implement SGD, Momentum, Nesterov, RMSProp, Adam, Nadam

    def train(self, X, y, epochs, batch_size):
        """Train the model."""
        pass  # Implement training loop with batches

    def predict(self, X):
        """Make predictions on new data."""
        pass  # Implement inference
