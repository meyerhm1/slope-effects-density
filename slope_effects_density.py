from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

desktop = 'C:\Users\Heather\Desktop\\'

if __name__ == '__main__':
    
    # read in catalogs
    vertices_cat = np.genfromtxt(desktop + 'HF_vertices.csv', dtype=None, names=True, delimiter=',')
    craters_cat = np.genfromtxt(desktop + 'CRATER_FullHF_RZPHM.csv', dtype=None, names=True, delimiter=',')

    # create arrays for more convenient access
    vertices_x_coord = vertices_cat['x_coord']
    vertices_y_coord = vertices_cat['y_coord']

    craters_x_coord = craters_cat['x_coord']
    craters_y_coord = craters_cat['y_coord']
    craters_diam = craters_cat['Diam_km']

    # plots
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(vertices_x_coord, vertices_y_coord, 'o', color='k', markersize=2)
    
    #for i in range(len()):


    plt.show()

    sys.exit(0)