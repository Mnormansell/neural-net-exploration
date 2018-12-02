#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
#To save
import pickle
#import cnn
#from keras.models import Sequential
#from keras.layers import Convolution2D

#Dictionary for labels, unfortunately labels are one less than the pokedex entry
LABELS = {
    '000' : 'Bulbasaur',
    '001' : 'Ivysaur',
    '002' : 'Venasaur',
    '003' : 'Charmander',
    '004' : 'Charmeleon',
    '005' : 'Charizard',
    '006' : 'Squirtle',
    '007' : 'Wartortle',
    '008' : 'Blastoise',
}

file_paths = input_data.getPaths('images')
training_data = input_data.training_data(file_paths)

# Splitting up training data
X = []
Y = []
for images, labels in training_data: 
    X.append(images)
    Y.append(labels)

# Have to convert the image list to an array
# Arguments of reshape are -1, Img_width, Img_height, num_columns (3 for RBG, 1 for Gray)
X = np.array(X).reshape(-1, 100, 100, 3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

training_iters = 200
learning_rate = .001
batch_size = 128

# machine learning bit https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
