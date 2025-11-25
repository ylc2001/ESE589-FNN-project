"""
compare_classifiers.py
~~~~~~~~~~~~~~~~~~~~~~

Simple comparison of classification algorithms on MNIST dataset.
Compares our FNN implementation with SVM and Random Forest.
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import network
import mnist_loader


def load_mnist_for_sklearn():
    """
    Load MNIST data in a format suitable for scikit-learn classifiers.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) as numpy arrays
    """
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    
    # Convert to sklearn format (flatten and stack)
    X_train = np.array([x.flatten() for x, _ in training_data])
    y_train = np.array([np.argmax(y) for _, y in training_data])
    
    X_test = np.array([x.flatten() for x, _ in test_data])
    y_test = np.array([y for _, y in test_data])
    
    return X_train, y_train, X_test, y_test


def train_fnn(X_train, y_train, X_test, y_test, epochs=10):
    """
    Train our FNN implementation.
    """
    # Convert back to FNN format
    training_inputs = [np.reshape(x, (784, 1)) for x in X_train]
    training_results = [mnist_loader.vectorized_result(y) for y in y_train]
    training_data = list(zip(training_inputs, training_results))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in X_test]
    test_data = list(zip(test_inputs, y_test))
    
    # Create and train network
    net = network.Network([784, 128, 64, 32, 10])
    
    start_time = time.time()
    net.SGD(training_data, epochs=epochs, mini_batch_size=10, lr=3.0,
            test_data=test_data, verbose=False)
    train_time = time.time() - start_time
    
    # Evaluate
    correct = net.evaluate(test_data)
    accuracy = correct / len(test_data)
    
    return accuracy, train_time


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM classifier.
    """
    # Use a subset for faster training (SVM is slow on large datasets)
    n_samples = min(10000, len(X_train))
    
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    
    start_time = time.time()
    clf.fit(X_train[:n_samples], y_train[:n_samples])
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, train_time, n_samples


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, train_time


def main():
    """
    Compare classification algorithms on MNIST.
    """
    print("=" * 60)
    print("MNIST Classification Comparison")
    print("=" * 60)
    
    print("\nLoading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist_for_sklearn()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = []
    
    # 1. FNN (our implementation)
    print("\n" + "-" * 40)
    print("Training FNN (our implementation)...")
    fnn_acc, fnn_time = train_fnn(X_train, y_train, X_test, y_test, epochs=10)
    results.append(("FNN (ours)", fnn_acc, fnn_time, len(X_train)))
    print(f"FNN Accuracy: {fnn_acc*100:.2f}%")
    print(f"Training time: {fnn_time:.2f}s")
    
    # 2. SVM
    print("\n" + "-" * 40)
    print("Training SVM...")
    svm_acc, svm_time, svm_samples = train_svm(X_train, y_train, X_test, y_test)
    results.append(("SVM", svm_acc, svm_time, svm_samples))
    print(f"SVM Accuracy: {svm_acc*100:.2f}% (trained on {svm_samples} samples)")
    print(f"Training time: {svm_time:.2f}s")
    
    # 3. Random Forest
    print("\n" + "-" * 40)
    print("Training Random Forest...")
    rf_acc, rf_time = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(("Random Forest", rf_acc, rf_time, len(X_train)))
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"Training time: {rf_time:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Accuracy':>10} {'Time (s)':>10} {'Samples':>10}")
    print("-" * 50)
    for name, acc, t, samples in results:
        print(f"{name:<20} {acc*100:>9.2f}% {t:>10.2f} {samples:>10}")
    
    print("\nNote: SVM uses a subset of training data for faster training.")


if __name__ == "__main__":
    main()
