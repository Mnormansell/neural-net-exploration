import numpy as np
import os
from PIL import Image
from resizeimage import resizeimage
from random import shuffle

# takes image directory and createsr 2D array of immages
def imageArray(root):
  #Create the overall data set
  dataset = []
  #create list of directories in root
  classes = []
  for dir in os.listdir(root):
    classes.append(os.path.join(root, dir))

  for counter, dir in enumerate(classes):
  # iterate through each directory
    file_paths = []

    for file in os.listdir(dir):
      file_paths.append(os.path.join(dir, file))

    for path in file_paths:
      # Open and resize image to height 200 (width scalled accordingt to ratio)
      img = Image.open(path)
      img = resizeimage.resize_height(img, 200)
      # Convert image to array of pixels
      pixels = np.array(img)
      height, width = pixels.shape[:2]
      print('file %s | height: %s width: %s' % (path, str(height), str(width)))
      # Append a tuple of image, label (label - 1 as counts start at zero)
      dataset.append( (pixels, int(classes[counter][-3::]) -1) )

  # shuffle the data
  shuffle(dataset)

  return dataset
