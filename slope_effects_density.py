from __future__ import division

import numpy as np

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

# Checks if os is windows or unix (mac is same as unix)
if os.name == 'posix':
    home = os.getenv('HOME')  # does not have a trailing slash
    desktop = home + '/Desktop/'
    slopedir = desktop + '/slope-effects-density/'
elif os.name == 'nt':
    desktop = 'C:\Users\Heather\Desktop\\'
    slopedir = desktop + '\\slope-effects-density\\'

def check_point_polygon(x, y, poly):
    """
    This function, which is the most important one, came from a 
    solution online. I simply copy pasted it.

    It checks whether the supplied coordinates of a point are within the area
    enclosed by the supplied polygon vertices and returns True or False.
    The algorithm is called the Ray Casting algorithm. 
    """

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

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
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

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

    #plot_region(vertices_x, vertices_y, vertices_x_center, vertices_y_center, eff_rad, valid_in, valid_out,\
    #region_name='orientale', save=True, with_craters=False)

    # define inner and outer polygons
    poly_outer = zip(vertices_x[valid_out], vertices_y[valid_out])
    poly_inner = zip(vertices_x[valid_in], vertices_y[valid_in])
    #print check_point_polygon(-3.5e6, -0.5e6, poly_outer)
    #print check_point_polygon(-3.5e6, -0.5e6, poly_inner)

    # ----------------  measure and populate pixel area function array  ---------------- # 
    # read in pixel slope info
    # this file also gives the x and y centers of pixels
    slope_arr = np.load(home + '/Documents/plots_codes_for_heather/slope_effects_files/3km_slope_points.npy')

    pix_x_cen_arr = slope_arr['pix_x_cen']
    pix_y_cen_arr = slope_arr['pix_y_cen']

    pix_centers = zip(pix_x_cen_arr, pix_y_cen_arr)

    pix_area_arr = np.zeros(len(pix_centers))

    # this can be optimized a LOT!!
    # not all pixels have to be checked with both polygons
    # in fact, most pixels don't need to be checked at all
    for i in range(len(pix_centers)):

        tl_x = pix_centers[i][0] - 5e2
        tr_x = pix_centers[i][0] + 5e2
        bl_x = pix_centers[i][0] - 5e2
        br_x = pix_centers[i][0] + 5e2

        tl_y = pix_centers[i][1] + 5e2
        tr_y = pix_centers[i][1] + 5e2
        bl_y = pix_centers[i][1] - 5e2
        br_y = pix_centers[i][1] - 5e2

        tl = [tl_x, tl_y]
        tr = [tr_x, tr_y]
        bl = [bl_x, bl_y]
        br = [br_x, br_y]

        pixel_corners = [tl, tr, bl, br]  # top and bottom, left and right
        for corner in pixel_corners:

            out_bool_tl = check_point_polygon(tl[0], tl[1], poly_outer)
            in_bool_tl = check_point_polygon(tl[0], tl[1], poly_inner)

            out_bool_tr = check_point_polygon(tr[0], tr[1], poly_outer)
            in_bool_tr = check_point_polygon(tr[0], tr[1], poly_inner)            

            out_bool_bl = check_point_polygon(bl[0], bl[1], poly_outer)
            in_bool_bl = check_point_polygon(bl[0], bl[1], poly_inner)

            out_bool_br = check_point_polygon(br[0], br[1], poly_outer)
            in_bool_br = check_point_polygon(br[0], br[1], poly_inner)

            out_bool = out_bool_tl and out_bool_tr and out_bool_bl and out_bool_br
            in_bool = in_bool_tl and in_bool_tr and in_bool_bl and in_bool_br

        if not out_bool:
            pix_area_arr[i] = 0
        if in_bool:
            pix_area_arr[i] = 1

        pix_area_arr[i] = pixel_area(x_cen, y_cen, poly)

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)