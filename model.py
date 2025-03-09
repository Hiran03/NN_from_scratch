# This file contains the code for forward pass and backpropogation, activation functions
import numpy as np
class Model:
    def __init__(self, num_hidden_layers, hidden_layer_size, weight_decay, learning_rate, optimizer, activation, weight_init, loss, beta = 0.9):
        # Initialize parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation = activation
        self.weight_init = weight_init
        self.loss = loss
        self.training_loss = []
        self.beta = beta
        
        
    def initialize_weights(self, input_size, output_size = 10):
        """Initialize weights based on weight_init strategy."""
        self.weights = []
        self.neuron_outputs = [np.zeros((input_size,1))]
        self.dw = [] 
        self.error = [np.zeros((input_size,1))]
        
        if self.weight_init == 'random':
            prev_layer = input_size
            for i in range(self.num_hidden_layers):
                self.weights.append(np.random.randn(prev_layer, self.hidden_layer_size[i])*0.01)  
                self.neuron_outputs.append(np.zeros((self.hidden_layer_size[i],1)))
                self.error.append(np.zeros((self.hidden_layer_size[i],1)))
                self.dw.append(np.zeros_like(self.weights[-1]))
                prev_layer = self.hidden_layer_size[i]
            self.weights.append(np.random.randn(prev_layer, output_size)*0.01)
            self.neuron_outputs.append(np.zeros((output_size,1)))
            self.error.append(np.zeros((output_size,1)))
            self.dw.append(np.zeros_like(self.weights[-1])) 
            
        if self.weight_init == 'xavier':
            prev_layer = input_size
            for i in range(self.num_hidden_layers):
                std = np.sqrt(2 / (prev_layer + self.hidden_layer_size[i]))  # Xavier Initialization
                self.weights.append(np.random.normal(0, std, (prev_layer, self.hidden_layer_size[i])))  
                self.neuron_outputs.append(np.zeros((self.hidden_layer_size[i],1)))
                self.error.append(np.zeros((self.hidden_layer_size[i],1)))
                self.dw.append(np.zeros_like(self.weights[-1]))
                prev_layer = self.hidden_layer_size[i]
            std = np.sqrt(2 / (prev_layer + output_size))  # Xavier Initialization
            self.weights.append(np.random.normal(0, std, (prev_layer, output_size)))
            self.neuron_outputs.append(np.zeros((output_size,1)))
            self.error.append(np.zeros((output_size,1)))
            self.dw.append(np.zeros_like(self.weights[-1]))
            
        self.u_w = []
        for i in self.weights:
            self.u_w.append(np.zeros_like(i))
            
    def initialize_biases(self, output_size = 10):
        """Initialize biases to zeros."""
        self.biases = []
        self.db = []
        for i in range(self.num_hidden_layers):
            self.biases.append(np.zeros((self.hidden_layer_size[i],1)))
            self.db.append(np.zeros((self.hidden_layer_size[i],1)))
        self.biases.append(np.zeros((output_size,1)))
        self.db.append(np.zeros((output_size,1)))
        self.u_b = []
        for i in self.biases:
            self.u_b.append(np.zeros_like(i))

    def activation_function(self, x):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            x = np.clip(x, -500, 500) 
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
            self.neuron_outputs[i + 1] = self.activation_function(np.dot(self.weights[i].T, self.neuron_outputs[i]) + self.biases[i])
        """Forward pass through the network."""
        self.neuron_outputs[-1] = self.activation_function(np.dot(self.weights[-1].T, self.neuron_outputs[-2]) + self.biases[-1])
        exps = np.exp(self.neuron_outputs[-1] - np.max(self.neuron_outputs[-1]))  # Subtract max for stability
        self.neuron_outputs[-1] = exps / np.sum(exps)  # Normalize


    def backward(self, true_output):
        
        if self.loss == 'cross-entropy' :
            self.error[-1] = -true_output + self.neuron_outputs[-1]
        elif self.loss == 'MSE' :
            self.error[-1] = -true_output + self.neuron_outputs[-1]
        for i in range(len(self.error) - 2, 0, -1) :
            self.error[i] = np.dot(self.weights[i], self.error[i + 1]) * self.activation_derivative(self.neuron_outputs[i])
    
    def gradients(self, X, y):
        self.forward(X)
        self.backward(y)
        for i in range(len(self.weights)) :
            self.dw[i] = np.dot(self.neuron_outputs[i], self.error[i + 1].T)
            self.db[i] = self.error[i + 1]
            
        return self.dw, self.db
        
    def update_weights(self, dw, db, X, y):
        """Apply optimization algorithm to update weights."""
        
        if self.optimizer == 'SGD' :
            for i in range(len(self.weights)): 
                self.weights[i] = self.weights[i] - self.learning_rate * dw[i]
                self.biases[i] = self.biases[i] - self.learning_rate * db[i]
                
        if self.optimizer == 'momentum':
            for i in range(len(self.weights)) :
                self.u_w[i] = self.beta*self.u_w[i] + dw[i]
                self.u_b[i] = self.beta*self.u_b[i] + db[i]
            
            for i in range(len(self.weights)): 
                self.weights[i] = self.weights[i] - self.learning_rate * self.u_w[i]
                self.biases[i] = self.biases[i] - self.learning_rate * self.u_b[i]
            
        if self.optimizer == 'nesterov':
            
            # Look first
            for i in range(len(self.weights)): 
                self.weights[i] = self.weights[i] - self.beta * self.u_w[i]
                self.biases[i] = self.biases[i] - self.beta * self.u_b[i]
                
            dw = [np.zeros_like(w) for w in self.weights]  # Create zero arrays matching the shape of gradients
            db = [np.zeros_like(b) for b in self.biases]
            
            for X_i, y_i in zip(X, y):  
                dW_curr, dB_curr = self.gradients(X_i, y_i)  # Get gradients
                # Accumulate gradients for each layer
                for i in range(len(dw)):
                    dw[i] += dW_curr[i]
                    db[i] += dB_curr[i]

            
            for i in range(len(self.weights)) :
                self.u_w[i] = self.beta*self.u_w[i] + dw[i]
                self.u_b[i] = self.beta*self.u_b[i] + db[i]
            
            for i in range(len(self.weights)): 
                self.weights[i] = self.weights[i] - self.learning_rate * self.u_w[i]
                self.biases[i] = self.biases[i] - self.learning_rate * self.u_b[i]
                
            
            
            
            
        # Implement SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
        

    def train(self, X, y, epochs, batch_size):
        """Train the model."""
        X = np.array([x.reshape(-1, 1) for x in X])
        y = np.array([each_y.reshape(-1, 1) for each_y in y])
        self.initialize_weights(X[0].shape[0], y.shape[1])
        self.initialize_biases(y.shape[1])
        num_batches = X.shape[0]//batch_size
        
        for epoch in range(epochs):
            print(f"Epoch number: {epoch + 1}\n")
            for start_index in range(0, X.shape[0], batch_size) :
                X_batch = X[start_index: start_index + batch_size]
                y_batch = y[start_index: start_index + batch_size]
                dw = [np.zeros_like(w) for w in self.weights]  # Create zero arrays matching the shape of gradients
                db = [np.zeros_like(b) for b in self.biases]
                
                for X_i, y_i in zip(X_batch, y_batch):  
                                        
                    dW_curr, dB_curr = self.gradients(X_i, y_i)  # Get gradients


                    # Accumulate gradients for each layer
                    for i in range(len(dw)):
                        dw[i] += dW_curr[i]
                        db[i] += dB_curr[i]
                
                self.update_weights(dw, db, X_batch, y_batch)  # Update model parameters
                
                
                # Logging the training loss
                correct_predictions = 0
                for X_i, y_i in zip(X_batch, y_batch):  
                    predicted_index = np.argmax(self.predict(X_i))  
                    actual_index = np.argmax(y_i)
                    # Check if prediction is correct
                    if predicted_index == actual_index:
                        correct_predictions += 1  # Increment the correct prediction count
                self.training_loss.append(correct_predictions/batch_size)
                if (start_index/batch_size % 100 == 0):
                    print(f"Epoch: {epoch + 1}, batch: {int(start_index/batch_size)}/{num_batches} completed....Accuracy: {correct_predictions/batch_size}")
                
            
    def predict(self, X):
        """Make predictions on new data."""
        self.forward(X)
        return self.neuron_outputs[-1]
