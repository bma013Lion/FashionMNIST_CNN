import torch
from data_loader import get_data_loaders
from train import train_model
from utils import plot_metrics, visualize_predictions, save_model
from config import TRAIN_CONFIG

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get data loaders
    print("Loading data...")
    train_loader, test_loader = get_data_loaders()
    
    # Train the model
    print("\nStarting training...")
    model, metrics = train_model(train_loader, test_loader)
    
    # Plot training metrics
    print("\nGenerating training metrics...")
    plot_metrics(
        metrics['train_losses'],
        metrics['test_losses'],
        metrics['train_accs'],
        metrics['test_accs']
    )
    
    # Visualize some predictions
    print("Generating sample predictions...")
    device = torch.device(TRAIN_CONFIG['device'])
    visualize_predictions(model, test_loader, device)
    
    # Save the model
    save_model(model)
    print("\nTraining complete! Check 'training_metrics.png' and 'sample_predictions.png' for results.")

if __name__ == "__main__":
    main()
