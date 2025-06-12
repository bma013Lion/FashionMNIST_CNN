import torch

# Model hyperparameters
MODEL_CONFIG = {
    # - The input images in FashionMNIST are 
    #   grayscale (black and white). 
    # - If they were color images, then we
    #   would use 3 for input_channels to
    #   represent Red, Green, and Blue channels.
    'input_channels': 1, 
    
    # - The first convolutional layer will
    #   have 32 output channels. This means 
    #   that the first layer will learn 32 
    #   different features of the input image. 
    # - We can think of these as 32 different 
    #   'filters' each looking for different 
    #   patterns like edges and textures.
    'conv1_out_channels': 32,
    
    # - The second convolutional layer will 
    #   learn 64 more complex features by combining 
    #   the 32 simpler features from the first 
    #   convolutional layer
    'conv2_out_channels': 64,
    
    # - After the convolutional layers, we'll 
    #   have a fully connected layer with 128 
    #   neurons.This layer helps make the final
    # decision about what's in the image.
    'fc1_units': 128,
    
    # - This model will classify images into 
    #   10 different categories. 
    'num_classes': 10,
    
    # - Dropout is a regualrization technique 
    #   where 50% of the neurons are randomly 
    #   "deactivated/turned-off" during training 
    #   to prevent overfitting to the training data
    'dropout': 0.5
}


# Training hyperparameters
TRAIN_CONFIG = {
    # - This batch size means that the model will 
    #   look at 64 images at one before updating 
    #   its knowledgebase.
    # - Kind of like styudying 64 flashcards at 
    #   a time instead of one by one or the entire 
    #   deck at once
    'batch_size': 64,
    
    # - The learning rate controls how big a step 
    #   the model will take when learning from its mistakes.
    # - For example, if the learning rate is 0.01, 
    #   the model will take a step of size 0.01 
    #   when updating its weights.
    # - During a learning step, a model makes predictions on a 
    #   batch of images, calculates how wrong those predictions 
    #   were (the loss), and then uses this information to
    #   figure out how to adjust its internal parameters to 
    #   increase accuracy. Finally, the parameters of the 
    #   model are updated towards that new direction.
    'learning_rate': 0.001,
    
    # - An epoch is one complete pass through the entire 
    #   training dataset
    # - having 10 epochs means that the model wll see all 
    #   of the training data 10 times
    'num_epochs': 10,
    
    # - Device is used to specify where the model 
    #   will be run (CPU or GPU)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Data configurations
DATA_CONFIG = {
    # - The data directory is the forlder where thr FashionMNIST 
    #   dataset is downloaded and stored. 
    # - './' means it will create this folder in the same 
    #   directory as the script.
    # - Will appear after running the script
    'data_dir': './data',
    
    # - Image size refferes to the pixel size of the images in the 
    #   dataset.
    # - The FashionMNIST images are 28x28 pixels.
    'image_size': 28,
    
    # - num_workers reffers to the number of CPU cores the 
    #   computer should use to load and prepare the data in the 
    #   background while the model is training.
    # - More workers may speed up the datat loading, however too 
    #   many can slow things down if your CPU cannot handle them.
    'num_workers': 2
}

# Class names for FashionMNIST dataset
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]