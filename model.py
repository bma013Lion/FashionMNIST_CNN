import torch.nn as nn
from config import MODEL_CONFIG

class FashionMNIST_CNN(nn.Module):
    """CNN for FashionMNIST classification."""
    
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        
        # This is the first convolutional layer (1)
        # ------------------------------------------
        # - nn.Conv2d creates a 2D convolutional layer,
        # - which is good for image processing
        #   partly because it uses local connectivity
        #   - looks into smaller parts of the images 
        #     at a time rather than the whole thing at once
        self.conv1 = nn.Conv2d(
            
            # - The input images in FashionMNIST are grayscale 
            #   as per what is recorded in the config file. 
            in_channels = MODEL_CONFIG['input_channels'],
            
            # - The layer will learn 32 different filters/features.
            # - Each filter will detect different patterns like 
            #   edges, curves, or textures.
            out_channels = MODEL_CONFIG['conv1_out_channels'],
            
            # - Kernel size is the size of the filter which 
            #   slides across the image to detect features.
            # - Here, 3x3 kernel is used.
            kernel_size = 3,
            
            # - Padding is added to the input image to 
            #   preserve the spatial dimensions of the 
            #   image while extracting features.
            # - Padding of 1 means that 1 pixel of padding 
            #   is added to each side of the image.
            # - Without the padding, the image would get
            #   slightly shrunken with each layer
            padding = 1
        )
        
        # This is the second convolutional layer (2)
        # -------------------------------------------
        self.conv2 = nn.Conv2d(
            
            # - The input for this layer are the 32 feature maps
            #   produced by the previous convolutional layer.
            in_channels = MODEL_CONFIG['conv1_out_channels'],
            
            # - The second layer will learn and output 64 new, 
            #   more complex features by combining the 32 
            #   simpler features from the first layer
            # - Each of the 64 features will combine the 32 input
            #   features to detect more complex patterns.
            out_channels = MODEL_CONFIG['conv2_out_channels'],
            
            # Same as layer 1
            kernel_size = 3,
            padding = 1
        )
        
        # Pooling layer
        # -------------
        # - The pooling layer is a key component of a CNN
        # - It reduces the spatial dimensions of the feature maps
        # - Here, the 2x2 max pooling is used which means that the
        #   window of pixels being analyzed is 2x2
        # - The max value in the window is taken and the rest 
        #   are discarded
        # - Helps capture important features while reducing data
        self.pool = nn.MaxPool2d(2, 2)
        
        # First fully connected layer
        # ---------------------------
        # - Takes the features extracted from the convolutional 
        #   layers and makes a final classification decision.
        # - connects every input to every output
        self.fc1 = nn.Linear(
            # This represents the number of input feautres
            # - We consuider the output channels of the second 
            #   convolutional layer (64) and the spatial dimensions
            #   of the feature maps (7x7) after the pooling layers.
            MODEL_CONFIG['conv2_out_channels'] * 7 * 7,
            
            # This represents the total number of neurons in this layer
            # -  Each one of these neurons will learn to detect 
            #    specific patters and image features
            MODEL_CONFIG['fc1_units']
        )
        
        # Second fully connected layer
        # ---------------------------
        # - Takes 128 features from the previous layer
        #   and creates the final prediction
        #
        # How it works: 
        # - Takes the 128 numbers from the previous layer
        # - Weighs and combines them in different ways to produce 10 scores
        # - Higher score = higher confidence that the image is that particular class
        # - The final prediction is the class with the highest score
        self.fc2 = nn.Linear(
            # - This represents the number of input features 
            #   (from the previous layer) [128]
            # - Each of these represents some combination of the 
            #   image's features
            MODEL_CONFIG['fc1_units'],
            
            # - This represents the total number of output types, 
            #   or classes, that the model can output as final prediction
            MODEL_CONFIG['num_classes']
        )
        
        # Activation function: ReLU (Rectified Linear Unit)
        # -----------------------------------------------
        # - Takes in any number as an input
        # - If the number is positive, it keeps the number as is
        # - If the number is negative, it changes the number to 0
        # 
        # For example:
        #   - ReLU(5) = 5
        #   - ReLU(-2) = 0
        #   - ReLU(0.1) = 0.1
        # 
        # - When ReLU outputs zero (for negative inputs), 
        #   those neurons effectively "turn off"
        # - This helps the network focus only on the most important features
        # - It's like highlighting only the most relevant parts of a document
        self.relu = nn.ReLU()
        
        # Dropout layer
        # ------------- 
        # - Dropout is a regularization technique 
        #   where 50% of the neurons are randomly 
        #   "deactivated/turned-off" during training 
        #   to prevent overfitting to the training data
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout'])
        
        # Forward pass
        # ------------
        # - This defines how the data flows through the network
        # - It takes the input data (x) and passes it through 
        #   the network layer by layer
    def forward(self, x):
            # First convolutional block
            x = self.pool(self.relu(self.conv1(x)))
            # Second convolutional block
            x = self.pool(self.relu(self.conv2(x)))
            # Flatten the tensor
            x = x.view(-1, MODEL_CONFIG['conv2_out_channels'] * 7 * 7)
            # Fully connected layers
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

def get_model():
    """Returns an instance of the FashionCNN model."""
    return FashionMNIST_CNN()