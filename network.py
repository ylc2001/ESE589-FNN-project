"""
network.py
~~~~~~~~~~

A Feedforward Neural Network (FNN) implementation with backpropagation.
Supports both CPU (NumPy) and GPU (PyCUDA) execution.
"""

import random
import numpy as np

# Try to import PyCUDA components
_PYCUDA_AVAILABLE = False
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    _PYCUDA_AVAILABLE = True
except ImportError:
    pass
except Exception:
    # PyCUDA might be installed but CUDA runtime not available
    pass


def is_gpu_available():
    """Check if GPU is available for computation."""
    return _PYCUDA_AVAILABLE


# CUDA kernel code for neural network operations
_CUDA_KERNEL_CODE = """
__global__ void sigmoid_kernel(float *z, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = z[idx];
        // Clip to prevent overflow
        if (val > 500.0f) val = 500.0f;
        if (val < -500.0f) val = -500.0f;
        result[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void sigmoid_prime_kernel(float *z, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = z[idx];
        // Clip to prevent overflow
        if (val > 500.0f) val = 500.0f;
        if (val < -500.0f) val = -500.0f;
        float sig = 1.0f / (1.0f + expf(-val));
        result[idx] = sig * (1.0f - sig);
    }
}

__global__ void matrix_vector_mult(float *matrix, float *vector, float *bias, 
                                   float *result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = bias[row];
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

__global__ void matrix_vector_mult_transpose(float *matrix, float *vector, 
                                             float *result, int rows, int cols) {
    // Computes matrix^T * vector where matrix is rows x cols
    // Result is cols x 1
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += matrix[row * cols + col] * vector[row];
        }
        result[col] = sum;
    }
}

__global__ void outer_product(float *a, float *b, float *result, int m, int n) {
    // Computes outer product a * b^T where a is m x 1 and b is n x 1
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        result[row * n + col] = a[row] * b[col];
    }
}

__global__ void elementwise_multiply(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_subtract(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_add(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void scale_subtract(float *w, float *nw, float scale, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = w[idx] - scale * nw[idx];
    }
}
"""


import threading


class CUDAKernels:
    """Wrapper class for CUDA kernels (thread-safe singleton)."""
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not CUDAKernels._initialized and _PYCUDA_AVAILABLE:
            with CUDAKernels._lock:
                # Double-check to avoid race condition
                if CUDAKernels._initialized:
                    return
                try:
                    self.mod = SourceModule(_CUDA_KERNEL_CODE)
                    self.sigmoid_kernel = self.mod.get_function("sigmoid_kernel")
                    self.sigmoid_prime_kernel = self.mod.get_function("sigmoid_prime_kernel")
                    self.matrix_vector_mult = self.mod.get_function("matrix_vector_mult")
                    self.matrix_vector_mult_transpose = self.mod.get_function("matrix_vector_mult_transpose")
                    self.outer_product = self.mod.get_function("outer_product")
                    self.elementwise_multiply = self.mod.get_function("elementwise_multiply")
                    self.elementwise_subtract = self.mod.get_function("elementwise_subtract")
                    self.elementwise_add = self.mod.get_function("elementwise_add")
                    self.scale_subtract = self.mod.get_function("scale_subtract")
                    CUDAKernels._initialized = True
                except Exception as e:
                    print(f"Warning: CUDA kernel compilation failed: {e}")
                    CUDAKernels._initialized = False


class Network(object):

    def __init__(self, sizes, device='cpu'):
        """
        Initialize the neural network.
        
        Args:
            sizes: List of layer sizes (e.g., [784, 128, 64, 10])
            device: 'cpu' or 'gpu'. If 'gpu' is specified but not available,
                    falls back to 'cpu' with a warning.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1).astype(np.float32) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x).astype(np.float32) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        # Set device
        self.device = device.lower()
        if self.device == 'gpu':
            if not _PYCUDA_AVAILABLE:
                print("Warning: GPU requested but PyCUDA/CUDA not available. Falling back to CPU.")
                self.device = 'cpu'
            else:
                try:
                    self.cuda_kernels = CUDAKernels()
                    if not CUDAKernels._initialized:
                        print("Warning: CUDA kernels failed to compile. Falling back to CPU.")
                        self.device = 'cpu'
                except Exception as e:
                    print(f"Warning: GPU initialization failed: {e}. Falling back to CPU.")
                    self.device = 'cpu'

    def feedforward(self, a):
        """
        Perform feedforward pass through the network.
        
        Args:
            a: Input activation (numpy array)
            
        Returns:
            Output activation (numpy array)
        """
        if self.device == 'gpu':
            return self._feedforward_gpu(a)
        else:
            return self._feedforward_cpu(a)
    
    def _feedforward_cpu(self, a):
        """CPU implementation of feedforward."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def _feedforward_gpu(self, a):
        """GPU implementation of feedforward using PyCUDA."""
        kernels = self.cuda_kernels
        block_size = 256
        
        # Convert input to float32 and flatten
        a_flat = a.astype(np.float32).flatten()
        a_gpu = gpuarray.to_gpu(a_flat)
        
        for b, w in zip(self.biases, self.weights):
            rows, cols = w.shape
            
            # Transfer weights and biases to GPU
            w_gpu = gpuarray.to_gpu(w.astype(np.float32).flatten())
            b_gpu = gpuarray.to_gpu(b.astype(np.float32).flatten())
            
            # Allocate output for matrix-vector multiplication
            z_gpu = gpuarray.empty(rows, dtype=np.float32)
            
            # Compute z = w * a + b
            grid_size = (rows + block_size - 1) // block_size
            kernels.matrix_vector_mult(
                w_gpu, a_gpu, b_gpu, z_gpu,
                np.int32(rows), np.int32(cols),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            # Allocate output for sigmoid
            a_gpu = gpuarray.empty(rows, dtype=np.float32)
            
            # Apply sigmoid activation
            kernels.sigmoid_kernel(
                z_gpu, a_gpu, np.int32(rows),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
        
        # Transfer result back to CPU
        result = a_gpu.get()
        return result.reshape(-1, 1)

    def SGD(self, training_data, epochs, mini_batch_size, lr,
            test_data=None, verbose=True):
        """
        Train the network using stochastic gradient descent.
        
        Args:
            training_data: List of (x, y) tuples
            epochs: Number of training epochs
            mini_batch_size: Size of each mini-batch
            lr: Learning rate
            test_data: Optional test data for evaluation
            verbose: Whether to print progress
            
        Returns:
            List of accuracy values per epoch (if test_data provided)
        """
        training_data = list(training_data)
        n = len(training_data)
        
        evaluation_results = []
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        if verbose:
            print(f"Training on device: {self.device.upper()}")
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            
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
        """Update weights and biases using gradient descent on a mini-batch."""
        if self.device == 'gpu':
            self._update_mini_batch_gpu(mini_batch, eta)
        else:
            self._update_mini_batch_cpu(mini_batch, eta)
    
    def _update_mini_batch_cpu(self, mini_batch, eta):
        """CPU implementation of mini-batch update."""
        nabla_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop_cpu(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        scale = eta / len(mini_batch)
        self.weights = [(w - scale * nw).astype(np.float32)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [(b - scale * nb).astype(np.float32)
                       for b, nb in zip(self.biases, nabla_b)]
    
    def _update_mini_batch_gpu(self, mini_batch, eta):
        """GPU implementation of mini-batch update."""
        nabla_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop_gpu(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases on GPU
        kernels = self.cuda_kernels
        block_size = 256
        scale = np.float32(eta / len(mini_batch))
        
        new_weights = []
        for w, nw in zip(self.weights, nabla_w):
            n = w.size
            grid_size = (n + block_size - 1) // block_size
            
            w_gpu = gpuarray.to_gpu(w.flatten())
            nw_gpu = gpuarray.to_gpu(nw.flatten())
            result_gpu = gpuarray.empty(n, dtype=np.float32)
            
            kernels.scale_subtract(
                w_gpu, nw_gpu, scale, result_gpu, np.int32(n),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            new_weights.append(result_gpu.get().reshape(w.shape))
        
        new_biases = []
        for b, nb in zip(self.biases, nabla_b):
            n = b.size
            grid_size = (n + block_size - 1) // block_size
            
            b_gpu = gpuarray.to_gpu(b.flatten())
            nb_gpu = gpuarray.to_gpu(nb.flatten())
            result_gpu = gpuarray.empty(n, dtype=np.float32)
            
            kernels.scale_subtract(
                b_gpu, nb_gpu, scale, result_gpu, np.int32(n),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            new_biases.append(result_gpu.get().reshape(b.shape))
        
        self.weights = new_weights
        self.biases = new_biases

    def backprop(self, x, y):
        """Compute gradients using backpropagation."""
        if self.device == 'gpu':
            return self._backprop_gpu(x, y)
        else:
            return self._backprop_cpu(x, y)
    
    def _backprop_cpu(self, x, y):
        """CPU implementation of backpropagation."""
        nabla_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        
        activation = x.astype(np.float32)
        activations = [x.astype(np.float32)]  # stores all activations, layer by layer
        zs = []  # stores all z vectors (weighted inputs)
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def _backprop_gpu(self, x, y):
        """GPU implementation of backpropagation."""
        kernels = self.cuda_kernels
        block_size = 256
        
        nabla_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        nabla_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        
        # Forward pass - store all activations and z values
        activation = x.astype(np.float32)
        activations = [activation]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            rows, cols = w.shape
            
            # Transfer to GPU
            w_gpu = gpuarray.to_gpu(w.flatten())
            b_gpu = gpuarray.to_gpu(b.flatten())
            a_gpu = gpuarray.to_gpu(activation.flatten())
            
            # Compute z = w * a + b
            z_gpu = gpuarray.empty(rows, dtype=np.float32)
            grid_size = (rows + block_size - 1) // block_size
            kernels.matrix_vector_mult(
                w_gpu, a_gpu, b_gpu, z_gpu,
                np.int32(rows), np.int32(cols),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            z = z_gpu.get().reshape(-1, 1)
            zs.append(z)
            
            # Apply sigmoid
            activation_gpu = gpuarray.empty(rows, dtype=np.float32)
            kernels.sigmoid_kernel(
                z_gpu, activation_gpu, np.int32(rows),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            activation = activation_gpu.get().reshape(-1, 1)
            activations.append(activation)
        
        # Backward pass
        # Compute delta for output layer
        output_size = activations[-1].size
        grid_size = (output_size + block_size - 1) // block_size
        
        # cost_derivative: output - y
        output_gpu = gpuarray.to_gpu(activations[-1].flatten())
        y_gpu = gpuarray.to_gpu(y.astype(np.float32).flatten())
        cost_deriv_gpu = gpuarray.empty(output_size, dtype=np.float32)
        kernels.elementwise_subtract(
            output_gpu, y_gpu, cost_deriv_gpu, np.int32(output_size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        # sigmoid_prime(z[-1])
        z_last_gpu = gpuarray.to_gpu(zs[-1].flatten())
        sp_gpu = gpuarray.empty(output_size, dtype=np.float32)
        kernels.sigmoid_prime_kernel(
            z_last_gpu, sp_gpu, np.int32(output_size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        # delta = cost_derivative * sigmoid_prime
        delta_gpu = gpuarray.empty(output_size, dtype=np.float32)
        kernels.elementwise_multiply(
            cost_deriv_gpu, sp_gpu, delta_gpu, np.int32(output_size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        delta = delta_gpu.get().reshape(-1, 1)
        nabla_b[-1] = delta
        
        # nabla_w[-1] = delta * activations[-2]^T (outer product)
        m, n = delta.shape[0], activations[-2].shape[0]
        a_prev_gpu = gpuarray.to_gpu(activations[-2].flatten())
        nw_gpu = gpuarray.empty(m * n, dtype=np.float32)
        
        block_2d = (16, 16, 1)
        grid_2d = ((n + 15) // 16, (m + 15) // 16, 1)
        kernels.outer_product(
            delta_gpu, a_prev_gpu, nw_gpu, np.int32(m), np.int32(n),
            block=block_2d, grid=grid_2d
        )
        nabla_w[-1] = nw_gpu.get().reshape(m, n)
        
        # Backpropagate through remaining layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            z_size = z.size
            grid_size = (z_size + block_size - 1) // block_size
            
            # sigmoid_prime(z)
            z_gpu = gpuarray.to_gpu(z.flatten())
            sp_gpu = gpuarray.empty(z_size, dtype=np.float32)
            kernels.sigmoid_prime_kernel(
                z_gpu, sp_gpu, np.int32(z_size),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            # delta = w^T * delta_prev * sp
            w_next = self.weights[-l+1]
            rows, cols = w_next.shape  # rows = current layer size, cols = prev layer size
            
            w_next_gpu = gpuarray.to_gpu(w_next.flatten())
            delta_prev_gpu = gpuarray.to_gpu(delta.flatten())
            
            # w^T * delta: result has size cols
            wt_delta_gpu = gpuarray.empty(cols, dtype=np.float32)
            grid_size_t = (cols + block_size - 1) // block_size
            kernels.matrix_vector_mult_transpose(
                w_next_gpu, delta_prev_gpu, wt_delta_gpu,
                np.int32(rows), np.int32(cols),
                block=(block_size, 1, 1), grid=(grid_size_t, 1)
            )
            
            # Element-wise multiply with sigmoid_prime
            delta_gpu = gpuarray.empty(z_size, dtype=np.float32)
            kernels.elementwise_multiply(
                wt_delta_gpu, sp_gpu, delta_gpu, np.int32(z_size),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
            delta = delta_gpu.get().reshape(-1, 1)
            nabla_b[-l] = delta
            
            # nabla_w[-l] = delta * activations[-l-1]^T
            m, n = delta.shape[0], activations[-l-1].shape[0]
            a_prev_gpu = gpuarray.to_gpu(activations[-l-1].flatten())
            nw_gpu = gpuarray.empty(m * n, dtype=np.float32)
            
            block_2d = (16, 16, 1)
            grid_2d = ((n + 15) // 16, (m + 15) // 16, 1)
            kernels.outer_product(
                delta_gpu, a_prev_gpu, nw_gpu, np.int32(m), np.int32(n),
                block=block_2d, grid=grid_2d
            )
            nabla_w[-l] = nw_gpu.get().reshape(m, n)
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Evaluate the network on test data."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Compute the cost derivative."""
        return (output_activations - y)

    def predict(self, x):
        """Predict the label for a single input."""
        return np.argmax(self.feedforward(x))
    
    def save(self, filename):
        """Save the network to a file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump((self.sizes, self.biases, self.weights, self.device), f)
    
    def get_device(self):
        """Return the current device being used."""
        return self.device


def sigmoid(z):
    """Sigmoid activation function (CPU)."""
    z_clipped = np.clip(z, -500, 500)  # clip to prevent overflow in exp
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_prime(z):
    """Derivative of the sigmoid function (CPU)."""
    return sigmoid(z) * (1 - sigmoid(z))
