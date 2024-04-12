## 1. Libraries Used
- Wandb: Used for experiment tracking and visualization.
- PyTorch: Deep learning framework for building and training the CNN model.
- torchvision: Provides datasets, models, and transformations for computer vision tasks.
- scikit-learn: Used for splitting the dataset into training and validation sets.
- matplotlib: Library for creating plots and visualizations.
- PIL: Python Imaging Library, used for image processing.
- numpy: Library for numerical computing.

## 2 RESNET50 Function

This repository contains a Python function `RESNET50_1` that returns a ResNet50 model with the option to freeze the first `k` layers. This function can be useful when you want to fine-tune a pre-trained ResNet50 model on a specific task while keeping some of the initial layers fixed.

## 3 Purpose

When working with pre-trained models such as ResNet50, it's common to fine-tune them on a new dataset for a specific task, such as image classification. However, freezing some of the initial layers can be beneficial, especially when dealing with limited computational resources or when the dataset is relatively small. Freezing these layers prevents their weights from being updated during training, allowing the model to retain valuable feature extraction capabilities learned from the original dataset while adapting to the new task.

## 4 Functionality

The `RESNET50` function takes two arguments:
- `k`: Number of layers to freeze.
- `NUM_OF_CLASSES`: Number of output classes for the final fully connected layer.

It returns a ResNet50 model with the specified number of initial layers frozen and a custom fully connected layer with the specified number of output classes. The function achieves this by loading the pre-trained ResNet50 model, setting the `requires_grad` attribute to `False` for the first `k` layers to freeze them, and replacing the final fully connected layer with a new one suitable for the given task.

## 5 Usage
```python
k = 10
NUM_OF_CLASSES = 1000
resnet_model = RESNET50_1(k, NUM_OF_CLASSES)
