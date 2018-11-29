import glob
import numpy as np
import os
import cv2

# takes image directory and createsr 2D array of immages
def imageArray(root):
  
  #create list of directories in root
  classes = []
  for dir in os.listdir(root):
    classes.append(os.path.join(root, dir))

  # iterate through each directory
  count = 0
  file_paths = []

  for file in os.listdir(classes[0]):
    if count < 20:
      file_paths.append(os.path.join(classes[0], file))
      count += 1
    else: 
      break

  images = []
  for path in file_paths:
    #Create a tuple of the pixel and it's label
    pixels = cv2.imread(path)
    height, width = cv2.shape[:2]
    print('height: %s' % str(height) + ' width: %s' %str(width))
    images.append( pixels ) 

  images = np.asarray(images)
  images = images / 255
  
  return images

  #for dir in classes:
    # iterate through each file in current directory and add it to a file path list
    #file_paths = []
    #for file in os.listdir(dir):
      #file_paths.append(os.path.join(dir, file))
    
    # iterate through those files and decode them
    #images = []
    #for path in file_paths:
      #images.append(misc.imread(path))

    # convert image list to array
    #images = np.asarray(images)
    # scale
    #images = images / 225 

    # add the array to dataset 
    #dataset.append(images)

  # convert data set into an array
  #dataset = np.asarray(dataset)

