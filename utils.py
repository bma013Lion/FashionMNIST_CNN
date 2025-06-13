import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from config import CLASS_NAMES

def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    """Plot training and testing metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def visualize_predictions(model, test_loader, device, num_images=12):
    """Visualize model predictions with confidence scores on test images."""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move tensors to the device
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images[:num_images])
        # Get probabilities using softmax
        probs = F.softmax(outputs, dim=1)
        # Get top predicted class and its probability
        confidences, preds = torch.max(probs, 1)
    
    # Convert to numpy for plotting
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    confidences = confidences.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    for idx in range(min(num_images, len(images))):
        plt.subplot(3, 4, idx + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        
        # Set title with true and predicted labels
        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[preds[idx]]
        confidence = confidences[idx] * 100  # Convert to percentage
        
        # Color code based on correct/incorrect prediction
        color = 'green' if labels[idx] == preds[idx] else 'red'
        
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                 color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()
    
    # Print average confidence
    avg_confidence = np.mean(confidences) * 100
    print(f'\nAverage prediction confidence: {avg_confidence:.1f}%')
    
    # Return the figure object in case it's needed programmatically
    return fig

def save_model(model, path='fashion_mnist_cnn.pth'):
    """Save the model state dictionary."""
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(path='fashion_mnist_cnn.pth'):
    """Load a model from a saved state dictionary."""
    from model import get_model
    model = get_model()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
