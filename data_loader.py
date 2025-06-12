import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import DATA_CONFIG, TRAIN_CONFIG

def get_data_loaders():
    """
    Returns train and test data loaders for FashionMNIST dataset.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Define transforms
    # ----------------
    # - transforms.Compose([...]) Combines multiple 
    #   transformations into one pipeline   
    transform = transforms.Compose([
        # - transforms.ToTensor(): Converts images
        #   from PIL format to PyTorch tensors
        # - Changes the data type to float32
        # - Scales pixel values from [0, 255] to [0.0, 1.0]
        # - Example: A pixel with value 128 becomes 0.5019
        transforms.ToTensor(),
        
        # - transforms.Normalize((0.5,), (0.5,)): Normalizes 
        #   the image by subtracting the mean and dividing 
        #   by the standard deviation
        # - This is a common preprocessing step in 
        #   image classification tasks
        # - Formula (pixel - mean) / std
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    # ------------
    # - torchvision.datasets.FashionMNIST(): Loads the 
    #   FashionMNIST dataset from torchvision
    # - root=DATA_CONFIG['data_dir']: Specifies the 
    #   directory where the dataset will be stored
    # - train=True: Loads the training dataset
    # - download=True: Downloads the dataset if it 
    #   is not already present
    # - transform=transform: Applies the defined 
    #   transformations to the dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_CONFIG['data_dir'],
        train=True,
        download=True,
        transform=transform
    )
    
    # - torchvision.datasets.FashionMNIST(): Loads the 
    #   FashionMNIST dataset from torchvision
    # - root=DATA_CONFIG['data_dir']: Specifies the 
    #   directory where the dataset will be stored
    # - train=False: Loads the test dataset
    # - download=True: Downloads the dataset if it 
    #   is not already present
    # - transform=transform: Applies the defined 
    #   transformations to the dataset
    test_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_CONFIG['data_dir'],
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    # -------------------
    # - DataLoader: Creates a data loader for the 
    #   dataset
    # - batch_size=TRAIN_CONFIG['batch_size']: 
    #   Specifies the batch size for training
    # - shuffle=True: Shuffles the data at each epoch
    # - num_workers=DATA_CONFIG['num_workers']: 
    #   Specifies the number of CPU cores to use 
    #   for loading the data
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DATA_CONFIG['num_workers']
    )
    
    # - DataLoader: Creates a data loader for the 
    #   dataset
    # - batch_size=TRAIN_CONFIG['batch_size']: 
    #   Specifies the batch size for training
    # - shuffle=False: Shuffles the data at each epoch
    # - num_workers=DATA_CONFIG['num_workers']: 
    #   Specifies the number of CPU cores to use 
    #   for loading the data
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DATA_CONFIG['num_workers']
    )
    
    return train_loader, test_loader
