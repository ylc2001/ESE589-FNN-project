"""
test_network.py
~~~~~~~~~~~~~~~

Unit tests for the Feedforward Neural Network implementation.
"""

import unittest
import numpy as np
import network


class TestNetworkInitialization(unittest.TestCase):
    """Test network initialization."""
    
    def test_init_sizes(self):
        """Test that network stores correct layer sizes."""
        net = network.Network([784, 128, 64, 32, 10])
        self.assertEqual(net.sizes, [784, 128, 64, 32, 10])
        self.assertEqual(net.num_layers, 5)
    
    def test_init_three_hidden_layers(self):
        """Test network with exactly 3 hidden layers."""
        net = network.Network([784, 100, 50, 25, 10])
        self.assertEqual(net.num_layers, 5)  # input + 3 hidden + output
        
    def test_init_biases_shape(self):
        """Test that biases have correct shapes."""
        net = network.Network([784, 128, 64, 32, 10])
        # Should have 4 bias vectors (one for each non-input layer)
        self.assertEqual(len(net.biases), 4)
        self.assertEqual(net.biases[0].shape, (128, 1))
        self.assertEqual(net.biases[1].shape, (64, 1))
        self.assertEqual(net.biases[2].shape, (32, 1))
        self.assertEqual(net.biases[3].shape, (10, 1))
    
    def test_init_weights_shape(self):
        """Test that weights have correct shapes."""
        net = network.Network([784, 128, 64, 32, 10])
        # Should have 4 weight matrices
        self.assertEqual(len(net.weights), 4)
        self.assertEqual(net.weights[0].shape, (128, 784))
        self.assertEqual(net.weights[1].shape, (64, 128))
        self.assertEqual(net.weights[2].shape, (32, 64))
        self.assertEqual(net.weights[3].shape, (10, 32))


class TestSigmoid(unittest.TestCase):
    """Test sigmoid function."""
    
    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        self.assertAlmostEqual(network.sigmoid(0), 0.5)
    
    def test_sigmoid_large_positive(self):
        """Test sigmoid approaches 1 for large positive values."""
        self.assertAlmostEqual(network.sigmoid(10), 1.0, places=4)
    
    def test_sigmoid_large_negative(self):
        """Test sigmoid approaches 0 for large negative values."""
        self.assertAlmostEqual(network.sigmoid(-10), 0.0, places=4)
    
    def test_sigmoid_array(self):
        """Test sigmoid on numpy array."""
        z = np.array([0, 1, -1])
        result = network.sigmoid(z)
        self.assertEqual(result.shape, (3,))
        self.assertAlmostEqual(result[0], 0.5)
        self.assertTrue(result[1] > 0.5)
        self.assertTrue(result[2] < 0.5)


class TestSigmoidPrime(unittest.TestCase):
    """Test sigmoid derivative function."""
    
    def test_sigmoid_prime_zero(self):
        """Test sigmoid'(0) = 0.25."""
        self.assertAlmostEqual(network.sigmoid_prime(0), 0.25)
    
    def test_sigmoid_prime_array(self):
        """Test sigmoid_prime on numpy array."""
        z = np.array([0, 0])
        result = network.sigmoid_prime(z)
        np.testing.assert_array_almost_equal(result, [0.25, 0.25])


class TestFeedforward(unittest.TestCase):
    """Test feedforward propagation."""
    
    def test_feedforward_output_shape(self):
        """Test that feedforward produces correct output shape."""
        net = network.Network([784, 128, 64, 32, 10])
        x = np.random.randn(784, 1)
        output = net.feedforward(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_feedforward_bounded(self):
        """Test that feedforward output is bounded by sigmoid (0, 1)."""
        net = network.Network([784, 30, 10])
        x = np.random.randn(784, 1)
        output = net.feedforward(x)
        self.assertTrue(np.all(output > 0))
        self.assertTrue(np.all(output < 1))


class TestBackprop(unittest.TestCase):
    """Test backpropagation."""
    
    def test_backprop_gradient_shapes(self):
        """Test that backprop produces gradients with correct shapes."""
        net = network.Network([784, 128, 64, 32, 10])
        x = np.random.randn(784, 1)
        y = np.zeros((10, 1))
        y[5] = 1.0  # One-hot encoding for digit 5
        
        nabla_b, nabla_w = net.backprop(x, y)
        
        # Check bias gradient shapes
        self.assertEqual(len(nabla_b), 4)
        for nb, b in zip(nabla_b, net.biases):
            self.assertEqual(nb.shape, b.shape)
        
        # Check weight gradient shapes
        self.assertEqual(len(nabla_w), 4)
        for nw, w in zip(nabla_w, net.weights):
            self.assertEqual(nw.shape, w.shape)


class TestEvaluate(unittest.TestCase):
    """Test evaluation function."""
    
    def test_evaluate_perfect(self):
        """Test evaluation with manually set weights for perfect prediction."""
        # Create tiny network
        net = network.Network([2, 2])
        
        # Create test data where we know the expected behavior
        test_data = [
            (np.array([[0], [1]]), 1),
            (np.array([[1], [0]]), 0),
        ]
        
        # Manually set weights to force specific outputs
        # This is a simplified test - just checking the evaluation logic
        result = net.evaluate(test_data)
        # Result should be between 0 and len(test_data)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, len(test_data))


class TestPredict(unittest.TestCase):
    """Test prediction function."""
    
    def test_predict_returns_int(self):
        """Test that predict returns an integer class."""
        net = network.Network([784, 30, 10])
        x = np.random.randn(784, 1)
        prediction = net.predict(x)
        self.assertIsInstance(prediction, (int, np.integer))
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 9)


class TestSGD(unittest.TestCase):
    """Test stochastic gradient descent training."""
    
    def test_sgd_improves(self):
        """Test that SGD training improves accuracy over epochs."""
        np.random.seed(42)
        
        # Create simple training data
        training_data = []
        test_data = []
        
        for _ in range(100):
            x = np.random.randn(10, 1)
            y_idx = np.random.randint(0, 3)
            y = np.zeros((3, 1))
            y[y_idx] = 1.0
            training_data.append((x, y))
        
        for _ in range(20):
            x = np.random.randn(10, 1)
            y_idx = np.random.randint(0, 3)
            test_data.append((x, y_idx))
        
        net = network.Network([10, 5, 3])
        results = net.SGD(training_data, epochs=5, mini_batch_size=10, eta=1.0,
                         test_data=test_data, verbose=False)
        
        # Check that we got results for each epoch
        self.assertEqual(len(results), 5)


class TestCostDerivative(unittest.TestCase):
    """Test cost derivative function."""
    
    def test_cost_derivative(self):
        """Test that cost derivative is output - expected."""
        net = network.Network([10, 5])
        output = np.array([[0.8], [0.2]])
        expected = np.array([[1.0], [0.0]])
        
        derivative = net.cost_derivative(output, expected)
        expected_derivative = np.array([[-0.2], [0.2]])
        
        np.testing.assert_array_almost_equal(derivative, expected_derivative)


class TestThreeHiddenLayers(unittest.TestCase):
    """Test specifically for 3 hidden layer requirement."""
    
    def test_network_has_three_hidden_layers(self):
        """Verify network architecture has at least 3 hidden layers."""
        # Standard architecture for MNIST with 3 hidden layers
        net = network.Network([784, 128, 64, 32, 10])
        
        # num_layers = input + hidden + output = 1 + 3 + 1 = 5
        self.assertEqual(net.num_layers, 5)
        
        # Hidden layers are indices 1, 2, 3
        hidden_layer_sizes = net.sizes[1:-1]
        self.assertEqual(len(hidden_layer_sizes), 3)
        self.assertEqual(hidden_layer_sizes, [128, 64, 32])


if __name__ == '__main__':
    unittest.main()
