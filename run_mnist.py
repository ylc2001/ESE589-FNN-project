"""
run_mnist.py
~~~~~~~~~~~~

Main script to train and evaluate the Feedforward Neural Network on MNIST.
Provides large benchmark tests with memory and execution time tracking,
and visualization of predictions.
"""

import time
import tracemalloc
import numpy as np
import network
import mnist_loader


# Global variable to store the trained model
_trained_model = None
_test_data = None


def large_benchmark():
    """
    Large benchmark test on full MNIST dataset.
    
    Trains on all 50,000 training samples and evaluates on 10,000 test samples.
    Shows memory usage and execution time.
    
    Returns:
        tuple: (results, net, test_data) - evaluation results, trained network, and test data
    """
    global _trained_model, _test_data
    
    print("\n" + "=" * 60)
    print("LARGE BENCHMARK TEST")
    print("=" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
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
    
    # Stop timing and get memory stats
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    best_accuracy = max(results)
    # Find the epoch with best accuracy (0-indexed internally, display as 0-indexed to match output)
    best_epoch = results.index(best_accuracy)
    
    print(f"\nBest accuracy: {best_accuracy*100:.2f}% (at Epoch {best_epoch})")
    print(f"Final accuracy: {results[-1]*100:.2f}%")
    
    # Display performance metrics
    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Peak memory usage: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Current memory usage: {current_mem / (1024 * 1024):.2f} MB")
    print("\n[Large benchmark completed]")
    
    # Store for visualization
    _trained_model = net
    _test_data = test_data
    
    return results, net, test_data


def visualize_predictions(num_samples=5, net=None, test_data=None):
    """
    Visualize network predictions on random test samples.
    
    Uses the same model from large_benchmark if no model is provided.
    
    Args:
        num_samples (int): Number of samples to display (default: 5).
        net: Trained network (if None, uses the model from large_benchmark).
        test_data: Test data (if None, uses data from large_benchmark).
    """
    global _trained_model, _test_data
    
    print("\n" + "=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)
    
    # Use provided model or the one from large_benchmark
    if net is None:
        if _trained_model is None:
            print("\nNo trained model available. Running large_benchmark first...")
            large_benchmark()
        net = _trained_model
        
    if test_data is None:
        if _test_data is None:
            print("\nNo test data available. Loading MNIST data...")
            _, _, test_data = mnist_loader.load_data_wrapper()
            _test_data = test_data
        else:
            test_data = _test_data
    
    print(f"\nShowing predictions for {num_samples} random test samples:")
    print("-" * 40)
    
    # Random sample indices
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    
    correct = 0
    for i, idx in enumerate(indices):
        x, y = test_data[idx]
        prediction = net.predict(x)
        is_correct = prediction == y
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"Image {i+1} (Sample #{idx}): Predicted={prediction}, Actual={y} {status}")
    
    print("-" * 40)
    print(f"Correct: {correct}/{num_samples}")
    
    return correct, num_samples


if __name__ == "__main__":
    import sys
    
    print("Feedforward Neural Network - MNIST Classification")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "large":
            large_benchmark()
        elif mode == "visualize":
            # Run large_benchmark first to train the model, then visualize
            large_benchmark()
            visualize_predictions()
        elif mode == "all":
            large_benchmark()
            visualize_predictions()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python run_mnist.py [large|visualize|all]")
    else:
        # Default: run large benchmark and then visualize
        print("\nRunning large benchmark followed by visualization...")
        print("(Use 'python run_mnist.py large' for benchmark only)")
        print()
        large_benchmark()
        visualize_predictions()
