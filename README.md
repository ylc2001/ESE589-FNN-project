# ESE589-FNN-project

A Feedforward Neural Network (FNN) implementation with backpropagation for MNIST digit classification.

## Overview

This project implements a Feedforward Neural Network from scratch using Python and NumPy. The implementation follows the structure from [Michael Nielsen's neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master) repository.

### Features

- **Feedforward Neural Network** with configurable architecture (supports 3+ hidden layers)
- **Backpropagation algorithm** for training
- **Stochastic Gradient Descent (SGD)** with mini-batch support
- **MNIST dataset loader** with automatic download via PyTorch/torchvision
- **Small and large scale examples** for validation and benchmarking

## Architecture

The network uses:
- Sigmoid activation function for all layers
- Quadratic cost function
- Random weight initialization from standard normal distribution

### Default Architecture

For MNIST classification:
- Input layer: 784 neurons (28x28 pixel images)
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Hidden layer 3: 32 neurons
- Output layer: 10 neurons (digits 0-9)

## Installation

### Requirements

- Python 3.7+
- NumPy
- PyTorch & torchvision (for dataset download and management)

### Setup

```bash
# Clone the repository
git clone https://github.com/ylc2001/ESE589-FNN-project.git
cd ESE589-FNN-project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run default small example (XOR + small MNIST subset)
python run_mnist.py

# Run small MNIST example only
python run_mnist.py small

# Run large benchmark on full MNIST
python run_mnist.py large

# Run XOR problem example
python run_mnist.py xor

# Visualize predictions
python run_mnist.py visualize

# Run all examples
python run_mnist.py all
```

### Python API

```python
import network
import mnist_loader

# Load MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create a network with 3 hidden layers
# Architecture: [input_size, hidden1, hidden2, hidden3, output_size]
net = network.Network([784, 128, 64, 32, 10])

# Train the network
# Parameters: training_data, epochs, mini_batch_size, learning_rate, test_data
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Make predictions
prediction = net.predict(test_data[0][0])
```

## Examples

### Small Illustrating Example

Uses 1,000 training samples and 200 test samples with a smaller network `[784, 30, 20, 15, 10]` for quick validation:

```bash
python run_mnist.py small
```

Expected output: ~70-85% accuracy after 10 epochs

### Large Benchmark

Uses full MNIST dataset (50,000 training, 10,000 test) with architecture `[784, 128, 64, 32, 10]`:

```bash
python run_mnist.py large
```

Expected output: ~97%+ accuracy after 30 epochs

### XOR Problem

Classic neural network validation test demonstrating the ability to learn non-linearly separable patterns:

```bash
python run_mnist.py xor
```

## Testing

Run unit tests to validate the implementation:

```bash
python -m unittest test_network -v
```

## File Structure

```
.
├── network.py          # Core FNN implementation with backpropagation
├── mnist_loader.py     # MNIST dataset loading and preprocessing
├── run_mnist.py        # Main script with examples and benchmarks
├── test_network.py     # Unit tests
├── README.md           # This file
└── data/               # MNIST data (auto-downloaded)
```

## Algorithm Details

### Feedforward

For each layer $l$:
$$a^l = \sigma(w^l \cdot a^{l-1} + b^l)$$

Where:
- $a^l$ is the activation of layer $l$
- $w^l$ is the weight matrix
- $b^l$ is the bias vector
- $\sigma$ is the sigmoid function

### Backpropagation

1. Compute output error: $\delta^L = \nabla_a C \odot \sigma'(z^L)$
2. Backpropagate: $\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$
3. Compute gradients:
   - $\frac{\partial C}{\partial b^l} = \delta^l$
   - $\frac{\partial C}{\partial w^l} = \delta^l (a^{l-1})^T$

### Stochastic Gradient Descent

Update weights and biases:
$$w \leftarrow w - \frac{\eta}{m} \sum_x \nabla_w C_x$$
$$b \leftarrow b - \frac{\eta}{m} \sum_x \nabla_b C_x$$

Where:
- $\eta$ is the learning rate
- $m$ is the mini-batch size

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) (downloaded via torchvision)

## License

MIT License