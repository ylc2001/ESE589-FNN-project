"""
run_mnist.py
~~~~~~~~~~~~

Main script to train and evaluate the Feedforward Neural Network on MNIST.
Provides both small illustrating examples and large benchmark tests.
"""

import numpy as np
import network
import mnist_loader


def small_example():
    """
    Small illustrating example for validation.
    
    Uses a subset of MNIST (1000 training, 200 test) with a smaller network
    to quickly validate the implementation.
    """
    print("=" * 60)
    print("SMALL ILLUSTRATING EXAMPLE")
    print("=" * 60)
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Use a small subset for quick validation
    small_training = training_data[:1000]
    small_test = test_data[:200]
    
    print(f"Training samples: {len(small_training)}")
    print(f"Test samples: {len(small_test)}")
    
    # Create a small network with 3 hidden layers
    # Input: 784, Hidden: 30, 20, 15, Output: 10
    print("\nCreating network with architecture: [784, 30, 20, 15, 10]")
    net = network.Network([784, 30, 20, 15, 10])
    
    print("\nTraining for 10 epochs with mini-batch size 10 and learning rate 3.0...")
    print("-" * 40)
    
    results = net.SGD(small_training, epochs=10, mini_batch_size=10, eta=3.0,
                      test_data=small_test)
    
    print("-" * 40)
    print(f"\nFinal accuracy on small test set: {results[-1]*100:.2f}%")
    print("\n[Small example completed]")
    
    return results


def large_benchmark():
    """
    Large benchmark test on full MNIST dataset.
    
    Trains on all 50,000 training samples and evaluates on 10,000 test samples.
    """
    print("\n" + "=" * 60)
    print("LARGE BENCHMARK TEST")
    print("=" * 60)
    print("\nLoading full MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    print(f"Training samples: {len(training_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create a network with 3 hidden layers
    # Input: 784, Hidden: 128, 64, 32, Output: 10
    print("\nCreating network with architecture: [784, 128, 64, 32, 10]")
    net = network.Network([784, 128, 64, 32, 10])
    
    print("\nTraining for 30 epochs with mini-batch size 10 and learning rate 3.0...")
    print("-" * 40)
    
    results = net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0,
                      test_data=test_data)
    
    print("-" * 40)
    
    best_accuracy = max(results)
    # Find the epoch with best accuracy (0-indexed internally, display as 0-indexed to match output)
    best_epoch = results.index(best_accuracy)
    
    print(f"\nBest accuracy: {best_accuracy*100:.2f}% (at Epoch {best_epoch})")
    print(f"Final accuracy: {results[-1]*100:.2f}%")
    print("\n[Large benchmark completed]")
    
    return results


def visualize_predictions(num_samples=5):
    """
    Visualize network predictions on random test samples.
    
    Args:
        num_samples (int): Number of samples to display.
    """
    print("\n" + "=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)
    
    print("\nLoading data and creating trained network...")
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    
    # Quick training on subset
    net = network.Network([784, 30, 20, 15, 10])
    net.SGD(training_data[:5000], epochs=5, mini_batch_size=10, eta=3.0, 
            verbose=False)
    
    print(f"\nShowing predictions for {num_samples} random test samples:")
    print("-" * 40)
    
    # Random sample indices
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    
    correct = 0
    for idx in indices:
        x, y = test_data[idx]
        prediction = net.predict(x)
        is_correct = prediction == y
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"Sample {idx}: Predicted={prediction}, Actual={y} {status}")
    
    print("-" * 40)
    print(f"Correct: {correct}/{num_samples}")


def xor_example():
    """
    XOR problem example - a classic neural network validation test.
    
    This demonstrates that the network can learn non-linearly separable patterns.
    """
    print("\n" + "=" * 60)
    print("XOR PROBLEM EXAMPLE")
    print("=" * 60)
    print("\nThe XOR problem is a classic test for neural networks.")
    print("Input patterns: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0")
    
    # Create XOR training data
    # Inputs: [0,0], [0,1], [1,0], [1,1]
    # Outputs: 0, 1, 1, 0
    training_data = [
        (np.array([[0], [0]]), np.array([[1], [0]])),  # 0 XOR 0 = 0 (class 0)
        (np.array([[0], [1]]), np.array([[0], [1]])),  # 0 XOR 1 = 1 (class 1)
        (np.array([[1], [0]]), np.array([[0], [1]])),  # 1 XOR 0 = 1 (class 1)
        (np.array([[1], [1]]), np.array([[1], [0]])),  # 1 XOR 1 = 0 (class 0)
    ]
    
    # Test data with integer labels for evaluation
    test_data = [
        (np.array([[0], [0]]), 0),
        (np.array([[0], [1]]), 1),
        (np.array([[1], [0]]), 1),
        (np.array([[1], [1]]), 0),
    ]
    
    # Create a small network with 3 hidden layers
    print("\nCreating network with architecture: [2, 4, 4, 3, 2]")
    net = network.Network([2, 4, 4, 3, 2])
    
    print("Training for 1000 epochs...")
    
    # Train with many epochs since dataset is tiny
    # Replicate training data for mini-batches
    expanded_training = training_data * 25  # 100 samples
    net.SGD(expanded_training, epochs=1000, mini_batch_size=4, eta=0.5,
            test_data=test_data, verbose=False)
    
    print("\nResults:")
    print("-" * 40)
    for x, y in test_data:
        output = net.feedforward(x)
        prediction = np.argmax(output)
        status = "✓" if prediction == y else "✗"
        print(f"Input: [{int(x[0][0])}, {int(x[1][0])}] -> "
              f"Predicted: {prediction}, Expected: {y} {status}")
    
    accuracy = net.evaluate(test_data)
    print("-" * 40)
    print(f"Accuracy: {accuracy}/4 ({accuracy*25}%)")
    print("\n[XOR example completed]")


if __name__ == "__main__":
    import sys
    
    print("Feedforward Neural Network - MNIST Classification")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "small":
            small_example()
        elif mode == "large":
            large_benchmark()
        elif mode == "xor":
            xor_example()
        elif mode == "visualize":
            visualize_predictions()
        elif mode == "all":
            xor_example()
            small_example()
            large_benchmark()
            visualize_predictions()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python run_mnist.py [small|large|xor|visualize|all]")
    else:
        # Default: run small example for quick validation
        print("\nRunning small example (use 'python run_mnist.py all' for full tests)")
        print()
        xor_example()
        small_example()
