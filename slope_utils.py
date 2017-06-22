from __future__ import division  

import numpy as np

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def numpy_to_asciiraster(npy_array_path, dir_for_raster, final_shape):

    npy_array = np.load(npy_array_path)

    if dir_for_raster == 'same':
        dir_for_raster = os.path.dirname(npy_array_path) + '/'

    npy_filename = os.path.splitext(os.path.basename(npy_array_path))
    raster_file = open(dir_for_raster + '/' + npy_filename + '.asc', 'wa')

    rows = final_shape[0]
    cols = final_shape[1]

    raster_file.close()

    return None

if __name__ == '__main__':
    
    print "\n"
    print "This module contains the following utility functions:"
    print "1. numpy_to_asciiraster: To convert 2D numpy arrays which are saved as "\
    "numpy binary files (i.e. with a .npy extension) to ascii raster images."
    print "2. asciiraster_to_numpy: Inverse of the previous conversion."
    print "\nYou must import this code as a module to use the functions."
    print "E.g. import slope_utils as su"
    print "Make sure that this module lies on your PYTHONPATH."
    print "Check documentation for individual functions for more details."
    print "\n"

    sys.exit(0)