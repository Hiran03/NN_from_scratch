# This file contains the code for forward pass and backpropogation, activation functions
import numpy as np
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
        self.training_loss = []


    def initialize_weights(self, input_size, output_size = 10):
        """Initialize weights based on weight_init strategy."""
        self.weights = []
        self.neuron_outputs = [np.zeros(input_size)]

        if self.weight_init == 'random':
            prev_layer = input_size
            for i in range(self.num_hidden_layers):
                self.weights.append(np.random.rand(prev_layer, self.hidden_layer_size[i]))  # Uniform distribution from zero to one
                self.neuron_outputs.append(np.zeros(self.hidden_layer_size[i]))
                prev_layer = self.hidden_layer_size[i]
            self.weights.append(np.random.rand(prev_layer, output_size))
            self.neuron_outputs.append(np.zeros(output_size))
            self.weights = np.array(self.weights)
            self.dw = np.zeros_like(self.weights)
            
    def initialize_biases(self, output_size = 10):
        """Initialize biases to zeros."""
        self.biases = []
        for i in range(self.num_hidden_layers):
            self.biases.append(np.zeros(1, self.hidden_layer_size[i]))
        self.biases.append(np.zeros(1, output_size))
        self.biases = np.array(self.biases)
        self.db = np.zeros_like(self.biases)
        
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
        self.neuron_outputs[0] = x
        for i in range(self.num_hidden_layers):
            self.neuron_outputs[i + 1] = self.activation_function(np.dot(self.weights[i].T, self.neuron_outputs[i]) + self.bias[i])
        """Forward pass through the network."""
        self.neuron_outputs[-1] = self.activation_function(np.dot(self.weights[-1].T, self.neuron_outputs[-2]) + self.bias[-1])
        self.neuron_outputs[-1] = np.exp(self.neuron_outputs[-1]) / np.sum(np.exp(self.neuron_outputs[-1])) # Softmax
        pass  # Implement forward propagation

    def backward(self, true_output):
        
        self.error = np.zeros_like(self.neuron_outputs)
        self.error[-1] = -(true_output - self.neuron_outputs[-1])
        for i in range(self.error.shape[0] - 1, 0, -1) :
            self.error[i] = np.dot(self.weights[i], self.error(i + 1)) * self.activation_derivative(self.neuron_outputs[i])
    
    def gradients(self):

        for i in range(self.weight.shape[0]) :
            self.dw[i] = np.dot(self.error[i + 1].T, self.neuron_outputs[i])
            self.db[i] = self.error[i + 1]
            
        return self.dw, self.db
        
    def update_weights(self, dw, db):
        """Apply optimization algorithm to update weights."""
        
        if self.optimizer == 'SGD' :
            self.weights = self.weights - self.learning_rate * dw
            self.biases = self.biases - self.learning_rate * db
                
        # Implement SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
        

    def train(self, X, y, epochs, batch_size):
        """Train the model."""
        self.initialize_weights(X.shape[1], y.shape[1])
        self.initialize_biases(y.shape[1])
        
        for epoch in epochs:
            print(f"Epoch number: {epoch + 1}\n")
            num_batches = X.shape[0]//batch_size
            for start_index in range(0, X.shape[0], batch_size) :
                X_batch = X[start_index: start_index + batch_size:]
                y_batch = y[start_index: start_index + batch_size:]
                dw, db = 0, 0  # Initialize gradients

                for X, y in zip(X_batch, y_batch):  
                    
                    self.forward(X)
                    self.backward(y)
                    dW_curr, dB_curr = self.gradients()  # Get gradients
                    dw += dW_curr  # Accumulate weight gradient
                    db += dB_curr  # Accumulate bias gradient

                self.update_weights(dw, db)  # Update model parameters
                
                # Logging the training loss
                for X, y in zip(X_batch, y_batch):  
                    correct_predictions = 0
                    predicted_index = np.argmax(self.predict(X))  
                    actual_index = np.argmax(y)
                    # Check if prediction is correct
                    if predicted_index == actual_index:
                        correct_predictions += 1  # Increment the correct prediction count
                    self.training_loss.append[correct_predictions/batch_size]
                    print(f"Epoch: {epoch + 1}, batch: {(start_index/batch_size)}/{num_batches} completed....Accuracy: {correct_predictions/batch_size}")
                                

    def predict(self, X):
        """Make predictions on new data."""
        self.forward(X)
        return self.neuron_outputs[-1]
