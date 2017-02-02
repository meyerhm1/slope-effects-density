from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

# Checks if os is windows or unix (mac is same as unix)
if os.name == 'posix':
    home = os.getenv('HOME')  # does not have a trailing slash
    desktop = home + '/Desktop/'
    slopedir = desktop + '/slope-effects-density/'
elif os.name == 'nt':
    desktop = 'C:\Users\Heather\Desktop\\'
    slopedir = desktop + '\\slope-effects-density\\'

def plot_region(vert_x, vert_y, vert_x_cen, vert_y_cen, eff_rad, valid_in, valid_out,\
    region_name='orientale', save=False, with_craters=False):
    # plots the region of interest
    # showing the annulus with the inner and outer polygons in different colors

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot dividing circle
    circle = plt.Circle((vertices_x_center, vertices_y_center), eff_rad, color='black', fill=False, ls='--')
    ax.add_artist(circle)

    # plot vertices inside and outside dividing circle in different colors
    ax.plot(vertices_x[valid_out], vertices_y[valid_out], '.-', color='r', markersize=2, markeredgecolor='r')
    ax.plot(vertices_x[valid_in], vertices_y[valid_in], '.-', color='g', markersize=2, markeredgecolor='g')

    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')

    #if with_craters:

    if save:
        fig.savefig(slopedir + region_name + '.png', dpi=300)
    else:
        plt.show()

    return None

if __name__ == '__main__':
    
    # read in catalogs
    vertices_cat = np.genfromtxt(slopedir + 'HF_vertices_m.csv', dtype=None, names=True, delimiter=',')
    craters_cat = np.genfromtxt(slopedir + 'CRATER_FullHF_m.csv', dtype=None, names=True, delimiter=',')

    # create arrays for more convenient access
    vertices_x = vertices_cat['x_coord_m']
    vertices_y = vertices_cat['y_coord_m']

    craters_x = craters_cat['x_coord_m']
    craters_y = craters_cat['y_coord_m']
    craters_diam = craters_cat['Diameter_m']

    # delete offending points
    off_x_idx1 = np.argmin(abs(vertices_x - -2.41165e6))
    off_y_idx1 = np.argmin(abs(vertices_y - 176717))

    off_x_idx2 = np.argmin(abs(vertices_x - -3.61074e6))
    off_y_idx2 = np.argmin(abs(vertices_y - 41808.2))

    off_x_idx3 = np.argmin(abs(vertices_x - -3.61526e6))
    off_y_idx3 = np.argmin(abs(vertices_y - 41295.4))

    off_x_idx = np.array([off_x_idx1, off_x_idx2, off_x_idx3])
    off_y_idx = np.array([off_y_idx1, off_y_idx2, off_y_idx3])
    vertices_x = np.delete(vertices_x, off_x_idx, axis=None)
    vertices_y = np.delete(vertices_y, off_y_idx, axis=None)

    # define radius and centers for vertices
    # eyeballed for now
    vertices_x_center = -2.94e6
    vertices_y_center = -5.87e5
    rad_vertices = np.sqrt((vertices_x - vertices_x_center)**2 + (vertices_y - vertices_y_center)**2)
    eff_rad = np.min(rad_vertices) + 25e4  # number put in by trial and error
    
    # define valid indices for vertices inside and outside dividing effective radius
    valid_out = np.where(rad_vertices > eff_rad)[0]
    valid_in = np.where(rad_vertices < eff_rad)[0]

    plot_region(vertices_x, vertices_y, vertices_x_center, vertices_y_center, eff_rad, valid_in, valid_out,\
    region_name='orientale', save=True, with_craters=False)

    # ----------------    ---------------- # 
    # define pixel array
    pix_x = np.arange(-4e6, -1.5e6, 1e3)
    pix_y = np.arange(-2e6, 0.5e6, 1e3)



    sys.exit(0)