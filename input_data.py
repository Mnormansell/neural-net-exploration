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
  
  for dir in classes:
  # iterate through each directory
    file_paths = []

    for file in os.listdir(dir):
      file_paths.append(os.path.join(dir, file))

    for path in file_paths:
      #Create a tuple of the pixel and it's label
      pixels = cv2.imread(path)
      height, width = pixels.shape[:2]
      print('file %s | height: %s width: %s' % (path, str(height), str(width)))
      dataset.append( pixels )
  
  return dataset
    


 

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

