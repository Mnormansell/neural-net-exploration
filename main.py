#Here we go
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow 
import input_data

dataset = input_data.imageArray('images')

print(dataset[0][0])