#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
<<<<<<< HEAD
=======
#import cnn
#from keras.models import Sequential
#from keras.layers import Convolution2D
>>>>>>> 924eacdab6a47b918060527d8e2d18fd45902d71

#Dictionary for labels, unfortunately labels are one less than the pokedex entry
labels = {
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

training_iters = 200
learning_rate = .001
batch_size = 128

# machine learning bit https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
