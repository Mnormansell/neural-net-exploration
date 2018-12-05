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

# Load stored data
# CODE FROM MAIN IF NEEDED
file_paths = input_data.getPaths('images')
training, testing = input_data.getData(file_paths)

train_x, train_one_hot_list_y, test_x, test_one_hot_list_y = input_data.dataHandling(training, testing)

pickle_out = open("train_x.pickle", "wb")
pickle.dump(train_x, pickle_out)
pickle_out.close()

pickle_out = open("train_y.pickle", "wb")
pickle.dump(train_one_hot_list_y, pickle_out)
pickle_out.close()


pickle_out = open("test_x.pickle", "wb")
pickle.dump(test_x, pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle", "wb")
pickle.dump(test_one_hot_list_y, pickle_out)
pickle_out.close()

print("Len train x: %s Len train y: %s Len test x: %s Len test y %s " % (len(train_x), len(train_one_hot_list_y), len(test_x), len(test_one_hot_list_y)))

# machine learning bit https://www.datacamp.com/community/tutorials/cnn-tensorflow-python