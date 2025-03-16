import numpy as np

def activation_function(activation, x):
    """Apply activation function."""
    if activation == 'sigmoid':
        x = np.clip(x, -700, 700) 
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        x = np.clip(x, -700, 700) 
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif activation == 'relu':
        return np.maximum(0,x)
    else:
        raise ValueError("Choose one of 'sigmoid' or 'relu' or 'tanh' for activations")

def activation_derivative(activation, x):
    """Compute the derivative for backpropagation."""
    if activation == "sigmoid":
        x = np.clip(x, -700, 700)
        sig_x = 1 / (1 + np.exp(-x))  # Compute sigmoid directly
        return sig_x * (1 - sig_x)  

    elif activation == "relu":
        return np.where(x > 0, 1, 0)  

    elif activation == "tanh":
        x = np.clip(x, -700, 700)
        tanh_x = np.tanh(x)  # Compute tanh directly
        return 1 - tanh_x ** 2 
    
def softmax(x):
        expx = np.exp(x - np.max(x))
        return expx / np.sum(expx) 