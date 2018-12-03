# Convolutional Neural Network
# Source: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd?fbclid=IwAR0v9M0oxzaZ5divb6NkslKdNdHrWaM594g0RfRGPEujaZiDQRdlgDMWMGg

import tensorflow as tf
# Importing Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Activation, Flatten, MaxPooling2D

from IPython.display import display
from PIL import Image
import pickle

# Im following this https://www.youtube.com/watch?v=WvoLTXIjBYU

# Data from preprocessing
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
# Rescale rgb values
X = X/255       

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(64, (3, 3), input_shape = X.shape[1:], activation = 'relu'))

# Step 2 - Pooling each 2x2 neighborhood of convoluted image and choosing max
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Repeats steps for another layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening into 1D Array to be able to input into Neural Net
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(64))
classifier.add(Dense(1))

# Step 5 - Activation
classifier.add(Activation("sigmoid"))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, Y, batch_size=32, epochs=3, validation_split=.1)
# Training classifier on training dataset
# With more epochs, there will be higher accuracy
# 10 epochs with 8,000 steps/enoch will take 1-2 hours to train

# TODO: split shuffle somehow
# TODO: we need to feed in the path to a directory containing our training data
