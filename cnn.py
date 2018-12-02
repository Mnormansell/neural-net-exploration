# Convolutional Neural Network
# Source: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd?fbclid=IwAR0v9M0oxzaZ5divb6NkslKdNdHrWaM594g0RfRGPEujaZiDQRdlgDMWMGg

import tensorflow as tf
# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

from IPython.display import display
from PIL import Image
import pickle

# Im following this https://www.youtube.com/watch?v=WvoLTXIjBYU

# Data from preprocessing
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)


# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling each 2x2 neighborhood of convoluted image and choosing max
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening into 1D Array to be able to input into Neural Net
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training classifier on training dataset
# With more epochs, there will be higher accuracy
# 10 epochs with 8,000 steps/enoch will take 1-2 hours to train

# TODO: split shuffle somehow
# TODO: we need to feed in the path to a directory containing our training data

classifier.fit_generator(
        training_set,
        steps_per_epoch = 500,
        epochs = 8,
        validation_data=test_set,
        validation_steps=800)
