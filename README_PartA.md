# DL_Assignment_2
Train a CNN model from scratch and learn how to tune the hyperparameters and visualise filters.

Libraries Used:
Tensorflow and keras for training the models and getting the pre-trained models.
ImageGrid library was used to handle images.
matplotlib and mpl_toolkits library was used for plotting the grid for the predicted images and visualise the filters in the first layer of our best model for a random image from the test set .
We have used numpy for most of the mathematical operations while playing around with the data.
wand to find insightful observations and the best hyperparameter configuration and the plots.
sklearn for splitting the training data and validation data.
Installations:
pip package manager is used to install the required libraries
How to USE?
The entire project has been modularised using functions to make it as scalable as possible for future developments and extensions. To train a model the project makes a call to train_with_params function with the required parameters.
The parameters in build are the following
arg1 : x_train : Input features in the dataset
arg2 : y_train : Input labels in the dataset
arg3 : x_val : Validation features in the dataset
arg4 : y_val : Validation labels in the dataset
arg5 : x_test : Test features in the dataset
arg6 : y_test : Test labels in the dataset
arg7 : filter : No of filters in each layer in a network
arg8 : kernel_size : Kernel sizes in each layer
arg9 : input_shape : The shape of the input
arg10 : weight_decay : Weight decay used in the model
arg11 : activation_name : The activation function we are using in each layer.
arg12 : do_batch_normalisation : A boolean value (True or False) to denote if batch normalization needs to be performed for the input at every layer
arg13 : neurons_in_dense : Number of neurons in dense layer
arg14 : dropout : The dropout rate signifies the probability of keeping a neuron. Hence the probability of dropping a neuron is 1-dropout_rate.
arg15 : num_classes : Number of classes in the output layer to classify.
arg16 : learning_rate : Learning rate being used in the model
arg15 : batch_size : Batch size set for the model
arg16 : do_data_augmentation : A boolean value describes if data augumentation needs to be performed or not
arg16 : manual_run : Describes if the model is run manually

How is the model trained?
We utilised the data augumentation method in Keras to improve performance and outcomes of machine learning models by forming new and different examples to train datasets. We take the images from the train directory and set it to 128 * 128 size and fit the training images on the compiled model using model.fit() by giving number of epochs.
Example:
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),epochs = 20,verbose = 1,validation_data= (x_val, y_val),batch_size=batch_size,callbacks = [WandbCallback()])

How to change number of filters, size of filters and activation function?
To change the number of filters, size of filters and activation function we can pass the required parameters in sweep_config file.

How is the model evaluated?
The images from the test directory are chosen randomly and set to (128,128) dimension irrespective of the original size of the image. The best model is saved and all the configurations were given as a dictionary and accuracy on the model is evaluated by calling calculateTestAccuracy which is defined seperately to report the accuracy.
Example
To report the accuracy.
calculateTestAccuracy(model, x_test, y_test)
print('accuracy: ', accuracy)


Link to Report
The wandb report is available here.


Acknowledgements
For the most of project we referred the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
https://github.com/
https://wandb.ai
http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture11.pdf
http://www.cse.iitm.ac.in/~miteshk/CS6910/Slides/Lecture12.pdf
