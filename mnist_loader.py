"""
mnist_loader.py
~~~~~~~~~~~~~~~

A library to load the MNIST image data.
Downloads from UCI ML Repository or uses local cache.
Can also generate synthetic data for testing when MNIST is unavailable.
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
    Falls back to synthetic data if download fails.
    
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
        try:
            _download_mnist(data_dir)
            _create_mnist_pkl(data_dir, mnist_pkl)
        except Exception as e:
            print(f"Warning: Could not download MNIST data: {e}")
            print("Using synthetic data for testing purposes.")
            return _generate_synthetic_data()
    
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
    Download MNIST dataset files from the internet.
    
    Args:
        data_dir (str): Directory to save the downloaded files.
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    print("Downloading MNIST dataset...")
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Downloading {filename}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
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


def _generate_synthetic_data():
    """
    Generate synthetic MNIST-like data for testing.
    
    Creates random images with patterns based on digit labels.
    This allows the network to be tested even without real MNIST data.
    
    Returns:
        tuple: A tuple containing (training_data, validation_data, test_data).
    """
    np.random.seed(42)  # For reproducibility
    
    def create_digit_pattern(digit, n_samples):
        """Create simple patterns that can be learned by the network."""
        images = np.random.rand(n_samples, 784).astype(np.float32) * 0.1
        labels = np.full(n_samples, digit, dtype=np.uint8)
        
        # Add distinctive features based on digit
        # Each digit has a unique pattern in specific regions
        start_col = (digit % 5) * 5
        start_row = (digit // 5) * 10
        
        for i in range(n_samples):
            # Add a bright region that identifies the digit
            idx_start = start_row * 28 + start_col
            for r in range(5):
                for c in range(5):
                    idx = idx_start + r * 28 + c
                    if idx < 784:
                        images[i, idx] = 0.8 + np.random.rand() * 0.2
            
            # Add some noise
            noise_idx = np.random.choice(784, 50, replace=False)
            images[i, noise_idx] += np.random.rand(50) * 0.3
        
        images = np.clip(images, 0, 1)
        return images, labels
    
    # Generate data for each digit
    train_per_digit = 5000
    val_per_digit = 1000
    test_per_digit = 1000
    
    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    test_images_list = []
    test_labels_list = []
    
    for digit in range(10):
        # Training
        imgs, lbls = create_digit_pattern(digit, train_per_digit)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)
        
        # Validation
        imgs, lbls = create_digit_pattern(digit, val_per_digit)
        val_images_list.append(imgs)
        val_labels_list.append(lbls)
        
        # Test
        imgs, lbls = create_digit_pattern(digit, test_per_digit)
        test_images_list.append(imgs)
        test_labels_list.append(lbls)
    
    train_images = np.vstack(train_images_list)
    train_labels = np.concatenate(train_labels_list)
    val_images = np.vstack(val_images_list)
    val_labels = np.concatenate(val_labels_list)
    test_images = np.vstack(test_images_list)
    test_labels = np.concatenate(test_labels_list)
    
    # Shuffle training data
    perm = np.random.permutation(len(train_images))
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    
    training_data = (train_images, train_labels)
    validation_data = (val_images, val_labels)
    test_data = (test_images, test_labels)
    
    return (training_data, validation_data, test_data)
