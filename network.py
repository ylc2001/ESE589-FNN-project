"""
network.py
~~~~~~~~~~

A Feedforward Neural Network (FNN) implementation with backpropagation.
Follows the structure from Michael Nielsen's neural-networks-and-deep-learning.

This module implements a feedforward neural network with stochastic gradient
descent learning algorithm using backpropagation.
"""

import random
import numpy as np


class Network(object):
    """
    A Feedforward Neural Network class.
    
    Attributes:
        num_layers (int): The number of layers in the network.
        sizes (list): A list containing the number of neurons in each layer.
        biases (list): A list of bias vectors for each layer (except input).
        weights (list): A list of weight matrices connecting adjacent layers.
    """

    def __init__(self, sizes):
        """
        Initialize the neural network.
        
        Args:
            sizes (list): A list containing the number of neurons in each layer.
                          For example, [784, 128, 64, 32, 10] creates a network
                          with 784 input neurons, three hidden layers with 128,
                          64, and 32 neurons, and 10 output neurons.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases randomly from standard normal distribution
        # No bias for input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights randomly from standard normal distribution
        # weights[i] connects layer i to layer i+1
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Compute the output of the network for a given input.
        
        Args:
            a (numpy.ndarray): Input vector (shape: (n, 1) where n is input size).
        
        Returns:
            numpy.ndarray: Output vector after feedforward propagation.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, verbose=True):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Args:
            training_data (list): A list of tuples (x, y) representing training
                                  inputs and desired outputs.
            epochs (int): Number of training epochs.
            mini_batch_size (int): Size of mini-batches for gradient descent.
            eta (float): Learning rate.
            test_data (list, optional): If provided, the network will be evaluated
                                        against the test data after each epoch.
            verbose (bool): If True, print progress during training.
        
        Returns:
            list: A list of evaluation results (accuracy) for each epoch if 
                  test_data is provided, otherwise empty list.
        """
        training_data = list(training_data)
        n = len(training_data)
        
        evaluation_results = []
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                correct = self.evaluate(test_data)
                accuracy = correct / n_test
                evaluation_results.append(accuracy)
                if verbose:
                    print(f"Epoch {j}: {correct} / {n_test} ({accuracy*100:.2f}%)")
            else:
                if verbose:
                    print(f"Epoch {j} complete")
        
        return evaluation_results

    def update_mini_batch(self, mini_batch, eta):
        """
        Update network weights and biases by applying gradient descent using
        backpropagation to a single mini-batch.
        
        Args:
            mini_batch (list): A list of tuples (x, y) for the mini-batch.
            eta (float): Learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases using average gradient
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Compute the gradient of the cost function using backpropagation.
        
        Args:
            x (numpy.ndarray): Input vector.
            y (numpy.ndarray): Desired output vector.
        
        Returns:
            tuple: A tuple (nabla_b, nabla_w) representing the gradient for
                   the cost function. nabla_b and nabla_w are layer-by-layer
                   lists of numpy arrays, similar to self.biases and self.weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors (weighted inputs), layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        # Output layer error (using quadratic cost function)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Backpropagate through hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs correctly classified.
        
        Args:
            test_data (list): A list of tuples (x, y) where x is the input
                              and y is the correct classification (integer).
        
        Returns:
            int: Number of correctly classified inputs.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the derivative of the quadratic cost function.
        
        Args:
            output_activations (numpy.ndarray): Network output.
            y (numpy.ndarray): Desired output.
        
        Returns:
            numpy.ndarray: Partial derivatives dC/da for output activations.
        """
        return (output_activations - y)

    def predict(self, x):
        """
        Predict the class for a single input.
        
        Args:
            x (numpy.ndarray): Input vector.
        
        Returns:
            int: Predicted class (index of highest output activation).
        """
        return np.argmax(self.feedforward(x))


def sigmoid(z):
    """
    The sigmoid activation function.
    
    Args:
        z (numpy.ndarray): Input.
    
    Returns:
        numpy.ndarray: Sigmoid of input.
    """
    # Clip to prevent overflow in exp for large negative values
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    
    Args:
        z (numpy.ndarray): Input.
    
    Returns:
        numpy.ndarray: Derivative of sigmoid at input.
    """
    return sigmoid(z) * (1 - sigmoid(z))
