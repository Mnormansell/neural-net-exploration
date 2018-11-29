import numpy as np
import os
import cv2

# takes image directory and createsr 2D array of immages
def imageArray(root):
  #Create the overall data set
  dataset = []
  #create list of directories in root
  classes = []
  for dir in os.listdir(root):
    classes.append(os.path.join(root, dir))
  
  #add label index
  label = 0
  for dir in classes:
  # iterate through each directory
    file_paths = []

    for file in os.listdir(dir):
      file_paths.append(os.path.join(dir, file))

    for path in file_paths:
      #Create a tuple of the pixel and it's label
      pixels = cv2.imread(path, 1)
      # Append a tuple of image, label (label - 1 as counts start at zero)
      dataset.append( (pixels, label))
    
    label += 1

  return dataset