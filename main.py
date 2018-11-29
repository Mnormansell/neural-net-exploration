#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import input_data
from keras.models import Sequential
from keras.layers import Convolution2D


dataset = input_data.imageArray('images')
print(dataset[0])

classes = os.listdir('images')
num_classes = len(classes)
