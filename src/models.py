import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 128)
        self.hidden2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10) 

    def forward(self, inputs):
        hidden1_output = F.relu(self.hidden1(inputs))
        hidden2_output = F.relu(self.hidden2(hidden1_output)) 
        return self.out(hidden2_output)

class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.hidden1 = nn.Linear(64 * 64 * 3, 128)
        self.hidden2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10)
        
    def forward(self, inputs):
        inputs = inputs.reshape(-1, 12288)
        answer1 = self.hidden1(inputs) 
        answer2 = self.hidden2(F.relu(answer1)) 
        return self.out(F.relu(answer2))

class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)
    Activation function: ReLU for both hidden layers
    There should be a maxpool after each convolution.
    The sequence of operations looks like this:
        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2
    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]
    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.layer1 = nn.Conv2d(3, 16, kernel_size = kernel_size[0], stride = stride[0]) 
        self.layer2 = nn.Conv2d(16, 32, kernel_size = kernel_size[1], stride = stride[1]) 
        self.pool = nn.MaxPool2d(2)
        self.out = nn.Linear(32 * 13 * 13, 10)
    def forward(self, inputs):
        inputs = inputs.permute((0, 3, 1, 2)) 
        answer1 = self.pool(F.relu(self.layer1(inputs))) 
        answer2 = self.pool(F.relu(self.layer2(answer1))) 
        return self.out(torch.flatten(answer2, start_dim = 1))


