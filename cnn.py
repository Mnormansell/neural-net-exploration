# Convolutional Neural Network
# Source: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd?fbclid=IwAR0v9M0oxzaZ5divb6NkslKdNdHrWaM594g0RfRGPEujaZiDQRdlgDMWMGg


# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from IPython.display import display
from PIL import Image

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
classifier.fit_generator(
        training_set,
        steps_per_epoch = 500,
        epochs = 8,
        validation_data=test_set,
        validation_steps=800)
