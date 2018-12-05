# Convolutional Neural Network
# Source: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd?fbclid=IwAR0v9M0oxzaZ5divb6NkslKdNdHrWaM594g0RfRGPEujaZiDQRdlgDMWMGg

import tensorflow as tf
# Importing Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard

from IPython.display import display
from PIL import Image
import pickle
import time
#will remove once i fix data nput
import numpy as np
# Imported to split training data, giving a validating test to help accuracy
from sklearn.model_selection import train_test_split

# Im following this https://www.youtube.com/watch?v=WvoLTXIjBYU

NAME = "Pokedex-%f" % time.time()

tensorboard = TensorBoard(log_dir='logs/%s' % NAME)

# Data from preprocessing
pickle_in = open("local_data/train_x.pickle", "rb")
train_x = pickle.load(pickle_in)

pickle_in = open("local_data/train_y.pickle", "rb")
train_y = pickle.load(pickle_in)
# convert to array
train_y = np.asarray(train_y)

pickle_in = open("local_data/test_x.pickle", "rb")
test_x = pickle.load(pickle_in)

pickle_in = open("local_data/test_y.pickle", "rb")
test_y = pickle.load(pickle_in)
test_y = np.asarray(test_y)

# Splits up the training data
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=13)

batch_size = 16
epochs = 5
num_classes = 9

# Start of the model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3,3), activation='linear', input_shape=(100,100,3), padding='same'))
model.add(Activation('relu'))  
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, (3, 3), activation='linear',padding='same'))
model.add(Activation('relu'))  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Convolution2D(128, (3, 3), activation='linear',padding='same'))
model.add(Activation('relu'))                   
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(Activation('relu'))                  
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
# uncomment to see model summary
# model.summary()

model_training = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_x, valid_y))