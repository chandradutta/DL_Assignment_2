## 1. Libraries Used
- Wandb: Used for experiment tracking and visualization.
- PyTorch: Deep learning framework for building and training the CNN model.
- torchvision: Provides datasets, models, and transformations for computer vision tasks.
- scikit-learn: Used for splitting the dataset into training and validation sets.
- matplotlib: Library for creating plots and visualizations.
- PIL: Python Imaging Library, used for image processing.
- numpy: Library for numerical computing.

  # RESNET50_1 Function

This repository contains a Python function `RESNET50_1` that returns a ResNet50 model with the option to freeze the first `k` layers. This function can be useful when you want to fine-tune a pre-trained ResNet50 model on a specific task while keeping some of the initial layers fixed.

## Usage

```python
from torchvision import models

def RESNET50_1(k, NUM_OF_CLASSES):
    """
    This function returns the ResNet50 model with the first k layers frozen.

    Args:
        k (int): Number of layers to freeze.
        NUM_OF_CLASSES (int): Number of output classes.

    Returns:
        model (torch.nn.Module): ResNet50 model with specified layers frozen.
    """
    model = models.resnet50(pretrained=True)

    params = list(model.parameters())
    for param in params[:k]:
        param.requires_grad = False  # Freezing

    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)

    return model

# Example usage
k = 10
NUM_OF_CLASSES = 1000
resnet_model = RESNET50_1(k, NUM_OF_CLASSES)

