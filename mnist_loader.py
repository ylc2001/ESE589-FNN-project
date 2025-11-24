"""
mnist_loader.py
~~~~~~~~~~~~~~~

A library to load the MNIST image data.
Downloads from UCI ML Repository (https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits)
or uses local cache.
"""

import gzip
import os
import pickle
import urllib.request
import numpy as np


def load_data():
    """
    Load the MNIST dataset.
    
    Downloads the dataset if not present locally.
    
    Returns:
        tuple: A tuple containing (training_data, validation_data, test_data).
               - training_data: 50,000 images
               - validation_data: 10,000 images
               - test_data: 10,000 images
               
               Each is a tuple of (images, labels) where images is a numpy
               array of shape (n, 784) and labels is a numpy array of shape (n,).
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    mnist_pkl = os.path.join(data_dir, 'mnist.pkl.gz')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(mnist_pkl):
        _download_mnist(data_dir)
        _create_mnist_pkl(data_dir, mnist_pkl)
    
    with gzip.open(mnist_pkl, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Load the MNIST dataset and return it in a format suitable for training.
    
    Returns:
        tuple: A tuple containing (training_data, validation_data, test_data).
               
               - training_data: list of 50,000 tuples (x, y) where x is a 784x1
                 numpy array (input image) and y is a 10x1 numpy array 
                 (one-hot encoded label).
               
               - validation_data: list of 10,000 tuples (x, y) where x is a 784x1
                 numpy array and y is the digit value (0-9).
               
               - test_data: list of 10,000 tuples (x, y) where x is a 784x1
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
    
    Args:
        j (int): A digit from 0 to 9.
    
    Returns:
        numpy.ndarray: A 10x1 numpy array with a 1.0 in the j-th position
                       and zeros elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def _download_mnist(data_dir):
    """
    Download MNIST dataset files from UCI ML Repository.
    
    Args:
        data_dir (str): Directory to save the downloaded files.
    """
    import zipfile
    
    # UCI ML Repository URL for MNIST dataset
    # https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits
    uci_url = "https://archive.ics.uci.edu/static/public/683/mnist+database+of+handwritten+digits.zip"
    
    print("Downloading MNIST dataset from UCI ML Repository...")
    print(f"  URL: {uci_url}")
    
    # Download the zip file
    zip_path = os.path.join(data_dir, "mnist_uci.zip")
    urllib.request.urlretrieve(uci_url, zip_path)
    
    # Extract the zip file
    print("  Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Remove the zip file after extraction
    os.remove(zip_path)
    
    print("Download complete.")


def _read_images(filepath):
    """
    Read images from an IDX file.
    
    Args:
        filepath (str): Path to the gzipped IDX file.
    
    Returns:
        numpy.ndarray: Array of images, shape (n_images, 784).
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, n_rows * n_cols)
        
        # Normalize to [0, 1]
        return data.astype(np.float32) / 255.0


def _read_labels(filepath):
    """
    Read labels from an IDX file.
    
    Args:
        filepath (str): Path to the gzipped IDX file.
    
    Returns:
        numpy.ndarray: Array of labels.
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and count
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def _create_mnist_pkl(data_dir, pkl_path):
    """
    Create a pickled version of the MNIST dataset.
    
    Args:
        data_dir (str): Directory containing the raw MNIST files.
        pkl_path (str): Path for the output pickle file.
    """
    print("Processing MNIST data...")
    
    # Read training data
    train_images = _read_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = _read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    
    # Read test data
    test_images = _read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = _read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    # Split training into training (50000) and validation (10000)
    training_data = (train_images[:50000], train_labels[:50000])
    validation_data = (train_images[50000:], train_labels[50000:])
    test_data = (test_images, test_labels)
    
    # Save as pickle
    with gzip.open(pkl_path, 'wb') as f:
        pickle.dump((training_data, validation_data, test_data), f)
    
    print("MNIST data processed and saved.")
