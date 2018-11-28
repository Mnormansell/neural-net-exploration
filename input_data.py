import glob
import numpy as np
import os
from scipy import misc

# takes image directory and createsr 2D array of immages
def imageArray(root):
  
  classes = os.listdir(root)
  
  dataset = []
  # iterate through each directory
  for dir in classes:

    # iterate through each file in current directory and add it to a file path list
    file_paths = []
    for file in os.listdir(dir):
      file_paths.append(os.path.join(dir, file))
    
    # iterate through those files and decode them
    images = []
    for path in file_paths:
      images.append(misc.imread(path))

    # convert image list to array
    images = np.asarray(images)
    # scale
    images = images / 225 

    # add the array to dataset 
    dataset.append(images)

  # convert data set into an array
  dataset = np.asarray(dataset)

  return dataset


