import numpy as np
import os
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
  if root is None:
    file_paths = paths
  else:
    try:
      file_paths = getPaths(root)
    except:
      print('Error reading directories')
  
  # Like above, the code iterates through a 2D array and converts the images to an array of pixels
  pixels = []
  for dir in file_paths:

    for file in dir:
      img = Image.open(file)
      # Resize
      img = resizeimage.resize_height(img, 200)
      # Convert image to array of pixels
      imgArray = np.array(img)
      height, width = imgArray.shape[:2]
      print('file %s | height: %s width: %s' % (file, str(height), str(width)))
      # Append a tuple of image, label (label - 1 as counts start at zero)
      pixels.append(imgArray)
  
  return pixels

