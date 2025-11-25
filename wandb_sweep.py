"""
wandb_sweep.py
~~~~~~~~~~~~~~

Use wandb's sweep to explore how model parameters (epochs, batch size, learning rate)
affect the model's performance on MNIST dataset.
"""

import wandb
import network
import mnist_loader


# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for efficient hyperparameter search
    'metric': {
        'name': 'best_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10, 15, 20, 30]
        },
        'batch_size': {
            'values': [8, 16, 32, 64, 128]
        },
        'learning_rate': {
            'min': 0.1,
            'max': 5.0
        }
    }
}


def train():
    """
    Training function for wandb sweep.
    
    Initializes a wandb run, loads MNIST data, trains the network with
    hyperparameters from the sweep, and logs the results.
    """
    # Initialize wandb run
    run = wandb.init()
    
    # Get hyperparameters from sweep config
    config = wandb.config
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    
    # Load MNIST data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Create network with architecture: [784, 128, 64, 32, 10]
    net = network.Network([784, 128, 64, 32, 10])
    
    # Train the network
    results = net.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=batch_size,
        lr=learning_rate,
        test_data=test_data,
        verbose=True
    )
    
    # Log accuracy for each epoch
    for epoch_num, accuracy in enumerate(results, start=1):
        wandb.log({
            'epoch': epoch_num,
            'accuracy': accuracy
        })
    
    # Log final metrics
    best_accuracy = max(results) if results else 0
    final_accuracy = results[-1] if results else 0
    
    wandb.log({
        'best_accuracy': best_accuracy,
        'final_accuracy': final_accuracy
    })
    
    print(f"\nSweep run completed:")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate:.4f}")
    print(f"  Best accuracy: {best_accuracy*100:.2f}%")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    
    run.finish()


def main():
    """
    Main function to initialize and run the wandb sweep.
    """
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="mnist-fnn-sweep")
    
    print("=" * 60)
    print("WANDB HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"\nSweep ID: {sweep_id}")
    print(f"Sweep method: {sweep_config['method']}")
    print(f"\nHyperparameter ranges:")
    print(f"  Epochs: {sweep_config['parameters']['epochs']['values']}")
    print(f"  Batch size: {sweep_config['parameters']['batch_size']['values']}")
    print(f"  Learning rate: {sweep_config['parameters']['learning_rate']['min']} - {sweep_config['parameters']['learning_rate']['max']}")
    print("\nStarting sweep agent...")
    print("-" * 60)
    
    # Run the sweep agent
    # count=10 means run 10 trials; adjust as needed
    wandb.agent(sweep_id, train, count=10)
    
    print("\n" + "=" * 60)
    print("Sweep completed!")
    print("View results at: https://wandb.ai/your-username/mnist-fnn-sweep")
    print("=" * 60)


if __name__ == "__main__":
    main()
