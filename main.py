#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import input_data
#import cnn
#from keras.models import Sequential
#from keras.layers import Convolution2D


dataset = input_data.imageArray('images')
imgplot = plt.imshow(dataset[0][0])
plt.show()

print('First 4 labels {} {} {} {}'.format([dataset[1][1]], dataset[2][1], dataset[3][1], dataset[4][0]))

classes = os.listdir('images')
num_classes = len(classes)
