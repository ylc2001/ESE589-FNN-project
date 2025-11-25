"""
mnist_loader.py
~~~~~~~~~~~~~~~

A library to load the MNIST image data using PyTorch's torchvision.
"""

import os
import numpy as np
from torchvision import datasets


def load_data():
    """
    Load the MNIST dataset using PyTorch's torchvision. Downloaded data is stored at './data'.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Download and load MNIST using torchvision
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
    
    # Convert to numpy arrays and normalize to [0, 1]
    train_images = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy()
    
    test_images = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy()
    
    # Split training into training (50000) and validation (10000)
    training_data = (train_images[:50000], train_labels[:50000])
    validation_data = (train_images[50000:], train_labels[50000:])
    test_data = (test_images, test_labels)
    
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Load the MNIST dataset and return it in a format suitable for training. Downloaded data is stored at './data'.
    
    Returns:
        tuple: A tuple containing (training_data, validation_data, test_data).
               
               - training_data: list of 50,000 tuples (x, y) where x is a 784x1
                 numpy array (input image) and y is a 10x1 numpy array 
                 (one-hot encoded label).
               
               - validation_data, test_data: list of 10,000 tuples (x, y) where x is a 784x1
                 numpy array and y is the digit value (0-9).
    """
    tr_d, va_d, te_d = load_data()
    
    # Training data: reshape inputs and one-hot encode outputs
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Validation data: reshape inputs, keep labels as integers
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Test data: reshape inputs, keep labels as integers
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    Convert a digit (0-9) into a one-hot encoded vector.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
