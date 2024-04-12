# DL_Assignment_2
## 1. Libraries Used
- Wandb: Used for experiment tracking and visualization.
- PyTorch: Deep learning framework for building and training the CNN model.
- torchvision: Provides datasets, models, and transformations for computer vision tasks.
- scikit-learn: Used for splitting the dataset into training and validation sets.
- matplotlib: Library for creating plots and visualizations.
- PIL: Python Imaging Library, used for image processing.
- numpy: Library for numerical computing.

## 2. CNN Model Explanation
The CNN model used in this assignment consists of multiple convolutional layers followed by max-pooling layers. Batch normalization and dropout are applied for regularization. The architecture can be customized by adjusting parameters such as the number of filters, kernel size, activation function, etc.

## 3. Configurable Hyperparameters
You can easily customize the architecture of the Convolutional Neural Network (CNN) by adjusting the following hyperparameters:

- **Number of Filters:** This parameter determines the depth of the feature maps extracted by each convolutional layer. You can specify the number of filters to control the complexity and capacity of the model.

- **Size of Filters:** The size of the filters (or kernels) defines the spatial extent of the convolution operation. By changing the size of the filters, you can adjust the receptive field of the network and influence the feature extraction process.

- **Activation Function:** The activation function introduces non-linearity into the network, enabling it to learn complex patterns from the input data. You can choose from various activation functions such as ReLU, GELU, SiLU (Sigmoid Linear Unit), or Mish, each offering different properties and performance characteristics.

To modify these hyperparameters, simply update the corresponding values in the sweep_config file. By experimenting with different combinations of these parameters, you can explore the model's behavior and performance across various configurations.
