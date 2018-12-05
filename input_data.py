import numpy as np
import os
import cv2
from PIL import Image
from resizeimage import resizeimage
from random import shuffle
from random import seed

# Returns a list of the file paths in one image
def getPaths(root):
  paths = []
  classes = []
  # Creates list of directories in root
  for dir in os.listdir(root):
    classes.append(os.path.join(root, dir))
  
  # Iterate through those directories
  for dir in classes:
    # Add the file paths to the files array
    files = []
    for file in os.listdir(dir):
      files.append(os.path.join(dir, file))
    # Creates the 2D array
    paths.append(files)
  
  return paths

def getData(paths):
  # Need a counter for the labels
  training_data = []
  testing_data = []
  for counter, dir in enumerate(paths):
    
    size = len(dir)

    # First 80% is training data
    for x in range( int((size * .8)) ):
      try:
        img = cv2.imread(dir[x])
        b,g,r = cv2.split(img)      
        rgb_img = cv2.merge([r,g,b]) 
        resized = cv2.resize(rgb_img, (100, 100))
        # Append a tuple of image, label (label - 1 as counts start at zero)
        training_data.append([resized, counter])
        

      except:
        print('Image Exception')

    # Last 20% is testing data
    for x in range( int((size * .8)), size ):
      try:
        img = cv2.imread(dir[x])
        b,g,r = cv2.split(img)      
        rgb_img = cv2.merge([r,g,b]) 
        resized = cv2.resize(rgb_img, (100, 100))
        # Append a tuple of image, label (label - 1 as counts start at zero)
        testing_data.append([resized, counter])
      except:
        print('Image Exception')
  shuffle(training_data)
  shuffle(testing_data)
  return training_data, testing_data

# This function takes the data from getData, splits up the data into images and labels,
# converts the images to arrays and reshapes them while converting the y labes to one-hot vectors
def dataHandling(train, test,  num_classes=9):
  train_x = []
  train_y = []
  for images, labels in train: 
    train_x.append(images)
    train_y.append(labels)

  test_x = []
  test_y = []
  for images, labels in test: 
    test_x.append(images)
    test_y.append(labels)

  train_x = np.asarray(train_x)
  test_x = np.asarray(test_x)
  # Reshape data
  train_x.reshape(-1, 100, 100, 3)
  test_x.reshape(-1, 100, 100, 3)
  train_x.astype('float32')
  test_x.astype('float32')
  # Divide pixels by 255 to get range between 0 - 1
  train_x = train_x / 255
  test_x = test_x / 255

  train_one_hot_list_y = []
  for label in train_y:
    one_hot_vector = []
    for i in range(num_classes):
      if (i == label):
        one_hot_vector.append(1)
      else:
        one_hot_vector.append(0)
    train_one_hot_list_y.append(one_hot_vector)
  
  test_one_hot_list_y = []
  for label in test_y:
    one_hot_vector = []
    for i in range(num_classes):
      if (i == label):
        one_hot_vector.append(1)
      else:
        one_hot_vector.append(0)
    test_one_hot_list_y.append(one_hot_vector)
  
  return train_x, train_one_hot_list_y, test_x, test_one_hot_list_y

