# FashionMNIST Classifier with PyTorch

A Convolutional Neural Network (CNN) implementation for classifying the FashionMNIST dataset using PyTorch. This project demonstrates how to build, train, and evaluate a deep learning model for image classification.

## Project Structure

```
FashionMNIST_CNN/
├── config.py         # Configuration parameters
├── data_loader.py    # Data loading and preprocessing
├── model.py          # CNN model architecture
├── train.py          # Training and evaluation logic
├── utils.py          # Utility functions and visualization
├── main.py           # Main script to run the training
└── requirements.txt  # Project dependencies
```

## Features

- Implements a CNN with two convolutional layers and two fully connected layers
- Uses ReLU activation and max pooling
- Includes dropout for regularization
- Tracks and visualizes training progress
- Saves model checkpoints and training metrics

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- torchvision
- matplotlib
- numpy
- tqdm

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd FashionMNIST_CNN
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the training script:

   ```bash
   python main.py
   ```

2. The script will:
   - Download the FashionMNIST dataset (if not already present)
   - Train the model for the specified number of epochs
   - Save the trained model
   - Generate visualizations of training metrics and sample predictions

## Configuration

You can modify the model and training parameters in `config.py`:

- `MODEL_CONFIG`: Model architecture parameters
- `TRAIN_CONFIG`: Training hyperparameters
- `DATA_CONFIG`: Data loading settings

## Results

After training, the following files will be generated:

- `training_metrics.png`: Plots of training and validation loss/accuracy
- `sample_predictions.png`: Visualizations of model predictions on test samples
- `fashion_mnist_cnn.pth`: Saved model weights

## Model Architecture

The CNN consists of:

- Two convolutional layers with ReLU activation and max pooling
- Dropout for regularization
- Two fully connected layers for classification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
