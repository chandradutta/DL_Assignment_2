# DL_Assignment_2
## 1. Libraries Used
- Wandb: Used for experiment tracking and visualization.
- PyTorch: Deep learning framework for building and training the CNN model.
- torchvision: Provides datasets, models, and transformations for computer vision tasks.
- scikit-learn: Used for splitting the dataset into training and validation sets.
- matplotlib: Library for creating plots and visualizations.
- PIL: Python Imaging Library, used for image processing.
- numpy: Library for numerical computing.

# 2 Convolutional Neural Network (CNN) for Image Classification

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for image classification tasks. The CNN architecture is defined in the `ConvNetwork` class within the `conv_network.py` file.

## 3ConvNetwork Class

The `ConvNetwork` class defines a CNN architecture with configurable parameters for flexibility in model design. Here's a brief overview of the class structure:

- **Initialization:** The constructor `__init__` initializes the CNN architecture with the following configurable parameters:
  - `n_filters`: Number of filters for the convolutional layers.
  - `filter_size`: Size of the filters (kernels) used in the convolutional layers.
  - `activationFun`: Activation function to be used in the CNN layers ('relu', 'gelu', 'silu', or 'mish').
  - `denseLay`: Number of neurons in the fully connected layer.
  - `drop_out`: Dropout probability for regularization.
  - `norm`: Whether to apply batch normalization ('true' or 'false').

- **Forward Pass:** The `forward` method defines the forward pass of the CNN. It takes input image tensors and applies convolutional layers, activation functions, pooling layers, and fully connected layers according to the defined architecture. Activation function for each layer is configurable through the `activationFun` parameter.

- **Activation Functions:** The `_get_activation_function` method returns the appropriate activation function based on the input argument.

## 4Usage

1. **Instantiate the Model:** Create an instance of the `ConvNetwork` class, specifying the desired configuration parameters.

2. **Forward Pass:** Pass input images through the model instance using the `forward` method, specifying the activation function for the layers.

3. **Training:** Train the model using your dataset and chosen optimizer.

## .5 Configurable Hyperparameters
You can easily customize the architecture of the Convolutional Neural Network (CNN) by adjusting the following hyperparameters:

- **Number of Filters:** This parameter determines the depth of the feature maps extracted by each convolutional layer. You can specify the number of filters to control the complexity and capacity of the model.

- **Size of Filters:** The size of the filters (or kernels) defines the spatial extent of the convolution operation. By changing the size of the filters, you can adjust the receptive field of the network and influence the feature extraction process.

- **Activation Function:** The activation function introduces non-linearity into the network, enabling it to learn complex patterns from the input data. You can choose from various activation functions such as ReLU, GELU, SiLU (Sigmoid Linear Unit), or Mish, each offering different properties and performance characteristics.

To modify these hyperparameters, simply update the corresponding values in the sweep_config file. By experimenting with different combinations of these parameters, you can explore the model's behavior and performance across various configurations.
