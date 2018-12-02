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

# takes paths array or images (2d) 
def imageArray(paths, root=None):
  # Due to optional parameter, file_path will default to path but will use the root if provided
  file_paths = []
  dataset = []
  if root is None:
    file_paths = paths
  else:
    try:
      file_paths = getPaths(root)
    except:
      print('Error reading directories')
  
  # Like above, the code iterates through a 2D array and converts the images to an array of pixels
  
  for dir in file_paths:

    pixels = []
    for file in dir:
      # img = Image.open(file)
      # Convert image to array of pixels
      img = cv2.imread(file)
      b,g,r = cv2.split(img)      
      rgb_img = cv2.merge([r,g,b]) 
      
      # resize image
      resized = cv2.resize(rgb_img, (100, 100))
      # Append a tuple of image, label (label - 1 as counts start at zero)
      pixels.append(resized)

    dataset.append(pixels)

  return dataset

def training_data(paths):
  # Need a counter for the labels
  training_data = []
  for counter, dir in enumerate(paths):
    
    size = len(dir)

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

  shuffle(training_data)
  return training_data

# CODE FROM MAIN IF NEEDED
#file_paths = input_data.getPaths('images')
#training_data = input_data.training_data(file_paths)

# Splitting up training data
#X = []
#Y = []
#for images, labels in training_data: 
    #X.append(images)
    #Y.append(labels)

# Have to convert the image list to an array
# Arguments of reshape are -1, Img_width, Img_height, num_columns (3 for RBG, 1 for Gray)
#X = np.array(X).reshape(-1, 100, 100, 3)

#pickle_out = open("X.pickle", "wb")
#pickle.dump(X, pickle_out)
#pickle_out.close()

#pickle_out = open("Y.pickle", "wb")
#pickle.dump(Y, pickle_out)
#pickle_out.close()

