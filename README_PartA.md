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
## 3. Model Training
The model is trained using the data augmentation method in Keras to enhance performance and outcomes of machine learning models. Initially, images from the training directory are resized to 128x128 dimensions. The training images are then fitted to the compiled model using the `model.fit()` function, specifying the number of epochs for training.

