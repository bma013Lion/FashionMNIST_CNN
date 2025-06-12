import matplotlib.pyplot as plt
import numpy as np
import torch
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
    """Visualize model predictions on test images."""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move tensors to the device
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move tensors back to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    # Plot the images with predictions
    fig = plt.figure(figsize=(12, 8))
    for idx in np.arange(min(num_images, len(images))):
        ax = fig.add_subplot(3, 4, idx+1, xticks=[], yticks=[])
        img = images[idx].squeeze()
        img = img / 2 + 0.5  # Unnormalize
        plt.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {CLASS_NAMES[predicted[idx]]}\nTrue: {CLASS_NAMES[labels[idx]]}', 
                     color='green' if predicted[idx] == labels[idx] else 'red')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

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
