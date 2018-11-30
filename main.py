#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
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
dataset = input_data.imageArray(file_paths)
print(dataset[0])


img = dataset[0]
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

training_iters = 200
learning_rate = .001
batch_size = 128

# machine learning bit https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
