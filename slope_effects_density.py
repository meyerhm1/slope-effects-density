from __future__ import division

import numpy as np
import Polygon as pg

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

# Checks if os is windows or unix (mac is same as unix)
if os.name == 'posix':
    home = os.getenv('HOME')  # does not have a trailing slash
    desktop = home + '/Desktop/'
    mdir = desktop + '/m-effects-density/'
elif os.name == 'nt':
    desktop = 'C:\Users\Heather\Desktop\\'
    mdir = desktop + '\\m-effects-density\\'

def check_point_polygon(x, y, poly):
    """
    This function, which is a very important one, came from a 
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

def get_pixel_area(polygon_x, polygon_y, pix_x_cen, pix_y_cen, case):
    """ 
    I'm not developing this function anymore. The problem of finding the 
    intersection of two polygons, which can be convex or concave, seems
    to be more complicated than I expected. A generic mathematical solution
    will take far too long to derive on my own. 

    I'm switching over to using the General Polygon Clipping (GPC) Library. 
    This was written by ALan Murta at U. Manchester. http://www.cs.man.ac.uk/~toby/gpc/
    Seems like it took him a couple years to write. I'm not sure what algorithms
    he uses to find polygon intersection but it appears to do what I want it 
    to do.

    # this function does not check if the supplied bounding polygon is the inner one
    # or the outer one. The preceding code must make sure the correct polygon is given.

    # find number of bounding polygon points inside pixel
    # if there are none:
    # then find the closest two points on either side of the pixel
    # else if there are some points inside the pixel: 
    # then those are to be taken into account for finding the intersecting shape
    # again you need to find the two closest points on the bounding polygon 
    # that are closest to an edge
    """

    polygon_inside_idx_x = np.where((polygon_x > pix_x_cen - 500) & (polygon_x < pix_x_cen + 500))[0]
    polygon_inside_idx_y = np.where((polygon_y > pix_y_cen - 500) & (polygon_y < pix_y_cen + 500))[0]

    polygon_inside_idx = np.intersect1d(polygon_inside_idx_x, polygon_inside_idx_y)

    # define intersecting polygon as empty list
    intersect_poly = []

    if polygon_inside_idx.size:
        # ie. there are bounding polygon vertices inside the pixel
        # this can happen for all cases

        if (case == 'tttt') or (case == 'ffff'):

            if (0 in polygon_inside_idx) or ((len(polygon_x)-1) in polygon_inside_idx):
                # returning wrong value # will fix this later
                if case == 'tttt':
                    return 1
                elif case == 'ffff':
                    return 0 
            elif (max(polygon_inside_idx) - min(polygon_inside_idx)) == len(polygon_inside_idx)-1:
                start_idx = min(polygon_inside_idx)
                end_idx = max(polygon_inside_idx)

            # loop over all segments
            seg_count = 0
            total_segments = len() + 2
            for i in range(int(end_idx) - int(start_idx) + 2):

                if start_idx+i >= len(polygon_x):
                    # must check that end_idx + 1 does not lead to an index outside the array indices
                    # it should come back around to 0 if that is the case
                    x1 = polygon_x[start_idx+i-1 - len(polygon_x)]
                    y1 = polygon_y[start_idx+i-1 - len(polygon_x)]
                    x2 = polygon_x[start_idx+i - len(polygon_x)]
                    y2 = polygon_y[start_idx+i - len(polygon_x)]
                else:
                    x1 = polygon_x[start_idx+i-1]
                    y1 = polygon_y[start_idx+i-1]
                    x2 = polygon_x[start_idx+i]
                    y2 = polygon_y[start_idx+i] 

                # find if this segment can intersect any pixel edge
                seg_inter_begin = np.where((x1 >= pix_x_cen - 500) & (x1 <= pix_x_cen + 500) &\
                    (y1 > pix_y_cen - 500) & (y1 < pix_y_cen + 500))[0]

                seg_inter_finish = np.where((x2 >= pix_x_cen - 500) & (x2 <= pix_x_cen + 500) &\
                    (y2 >= pix_y_cen - 500) & (y2 <= pix_y_cen + 500))[0]

                # redundant checks
                # should never be trigerred if the program logic is correct
                if (seg_count == 0) and seg_inter_begin.size:
                    print "Something went wrong with assigning indices to bounding polygon vertices inside pixel. Exiting..."
                    sys.exit(0)
                elif (seg_count == 0) and not seg_inter_finish.size:
                    print "Something went wrong with assigning indices to bounding polygon vertices inside pixel. Exiting..."
                    sys.exit(0)

                # the beginning of the last segment by default must be part of the intersecting polygon 
                if seg_count == total_segments - 1:
                    intersect_poly.append((x1,y1))
                    seg_count += 1

                    continue

                # start checking which segment ends are inside pixel
                if seg_inter_begin.size or seg_inter_finish.size:
                    # i.e. the case where one or both of the ends of the segments is inside the pixel
                    # find the point of intersection

                    # if both segment ends are inside then 
                    # check that the previous 
                    if seg_inter_begin.size and seg_inter_finish.size:
                        intersect_poly.append((x2,y2))
                        # check that the previous one did actually a
                        seg_count += 1
                        continue

                    # if only one end is inside then find where the segment intersects the edge
                    for j in range(4):

                        x3 = pixel_corners[j-1][0]
                        y3 = pixel_corners[j-1][1]
                        x4 = pixel_corners[j][0]
                        y4 = pixel_corners[j][1]

                        m1 = (y2 - y1) / (x2 - x1)
                        m2 = (y4 - y3) / (x4 - x3)

                        c1 = y1 - m1 * x1
                        c2 = y3 - m2 * x3

                        xi = (c2 - c1) / (m1 - m2)
                        yi = (m1*c2 - m2*c1) / (m1 - m2)

                        # check that the intersection point found actually does lie on the edge and the segment
                        if (x1 <= xi <= x2) and (x3 <= xi <= x4) and (y1 <= yi <= y2) and (y3 <= yi <= y4):
                            intersect_poly.append((xi,yi))

                            # the end of the first segment by default must be part of the intersecting polygon 
                            if seg_count == 0:
                                intersect_poly.append((x2,y2))

                            edge_inter_count.append()
                            seg_count += 1
                            if (in edge_inter_count) and (seg_count == total_segments - 1):
                                intersect_poly.append((pixel_corners[edge_count][0], pixel_corners[edge_count][1]))

                            break
                        else:
                            continue

                else:
                    # move to the next segment
                    seg_count += 1
                    continue

            return get_area(intersect_poly)

        elif case == 'tttf':
            # dummy return # to be fixed
            return None

    # check that there are no bounding polygon vertices on the pixel edge
    # this can only occur for the middle 3 cases with non-zero intersecting area
    if polygon_edge_idx.size:
        #ie. there are bounding polygon vertices on the pixel edge
        # dummy return # to be fixed
        return None

    else:
        # find the two closest points on either side
        # dummy return # to be fixed
        return None

    else:
        # these are the couple cases where the pixel intersecting area is either 0 or 1
        # in this case there will be no polygon points that are inside the pixel
        if (case == 'tttt'):
            return 1

        elif (case == 'ffff'):
            return 0

        elif (case == 'tttf'):
            # dummy return # to be fixed
            return None

def return_unzipped_list(poly):
    """
    This function can take a polygon object from the Polygon module
    or it can take a numpy array.
    In both cases the polygon is a list of coordinate pairs.
    """

    if type(poly) is pg.cPolygon.Polygon:
        px, py = zip(*poly[0])
    elif type(poly) is np.ndarray:
        px, py = zip(*poly)

    return px, py

def polygon_plot_prep(poly):

    px, py = return_unzipped_list(poly)

    if px[0] != px[-1]:
        px = np.append(px, px[0])
        py = np.append(py, py[0])

    return px, py

def plot_polygon_intersection(poly1, poly2):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    px1, py1 = polygon_plot_prep(poly1)
    px2, py2 = polygon_plot_prep(poly2)

    ax.plot(px1, py1, '-', color='b', lw=2)
    ax.plot(px2, py2, '-', color='r', lw=2)

    poly_i = poly1 & poly2
    px_i, py_i = polygon_plot_prep(poly_i)

    ax.plot(px_i, py_i, '-', color='g', lw=3)

    ax.set_xlim()

    return None

def plot_region(vert_x, vert_y, vert_x_cen, vert_y_cen, eff_rad, valid_in, valid_out,\
    region_name='orientale', save=False, with_craters=False, show_rect=False):
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

    #if with_craters:

    if show_rect:
        # Plot the rectangles within which the pixel corners are not checked
        in_rect_x = [-3.35e6, -2.55e6, -2.55e6, -3.35e6, -3.35e6]
        in_rect_y = [-9.5e5, -9.5e5, -3.3e5, -3.3e5, -9.5e5]

        out_rect_x = [-3.55e6, -2.32e6, -2.32e6, -3.59e6, -3.55e6]
        out_rect_y = [-1.47e6, -1.47e6, 5e4, 5e4, -1.47e6]

        ax.plot(in_rect_x, in_rect_y, '-', color='b')
        ax.plot(out_rect_x, out_rect_y, '-', color='b')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')

    if save:
        fig.savefig(mdir + region_name + '.png', dpi=300)
    else:
        plt.show()

    return None

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # read in catalogs
    vertices_cat = np.genfromtxt(mdir + 'HF_vertices_m.csv', dtype=None, names=True, delimiter=',')
    craters_cat = np.genfromtxt(mdir + 'CRATER_FullHF_m.csv', dtype=None, names=True, delimiter=',')

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
    #region_name='orientale', save=True, with_craters=False, show_rect=False)

    # define inner and outer polygons
    poly_outer = zip(vertices_x[valid_out], vertices_y[valid_out])
    poly_inner = zip(vertices_x[valid_in], vertices_y[valid_in])

    # ----------------  measure and populate pixel area function array  ---------------- # 
    # read in pixel m info
    # this file also gives the x and y centers of pixels
    m_arr = np.load(home + '/Documents/plots_codes_for_heather/m_effects_files/m_points.npy')

    pix_x_cen_arr = m_arr['pix_x_cen']
    pix_y_cen_arr = m_arr['pix_y_cen']

    pix_centers = zip(pix_x_cen_arr, pix_y_cen_arr)
    pix_area_arr = np.zeros(len(pix_centers))

    # loop over all pixels
    for i in range(len(pix_centers)):

        # check if pixel center falls "well" inside the inner excluding rectangle
        if (min(in_rect_x) + 500 < pix_x_cen_arr[i]) and (pix_x_cen_arr[i] < max(in_rect_x) - 500) and \
        (min(in_rect_y) + 500 < pix_y_cen_arr[i]) and (pix_y_cen_arr[i] < max(in_rect_y) - 500):
            pix_area_arr[i] = 0
            continue

        # in any other case you'll have to define the corners and proceed
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

        # if the pixel center is "well" inside the outer excluding rectangle then 
        # you only need to check the pixel corners with respect to the inner polygon
        if (min(out_rect_x) + 500 < pix_x_cen_arr[i]) and (pix_x_cen_arr[i] < max(out_rect_x) - 500) and \
        (min(out_rect_y) + 500 < pix_y_cen_arr[i]) and (pix_y_cen_arr[i] < max(out_rect_y) - 500):
            # so its inside the outer rect
            # check corners with inner polygon
            in_bool_tl = check_point_polygon(tl[0], tl[1], poly_inner)
            in_bool_tr = check_point_polygon(tr[0], tr[1], poly_inner)
            in_bool_bl = check_point_polygon(bl[0], bl[1], poly_inner)
            in_bool_br = check_point_polygon(br[0], br[1], poly_inner)

            # all cases are either pixels 
            # that intersect the edge or the pix in the annulus
            # the 5 cases are:
            # Case 1: All vertices True -- TTTT
            # Case 2: One False -- TTTF
            # Case 3: two False -- TTFF
            # Case 4: three False -- TFFF
            # Case 5: All veritces False -- FFFF

            # Case 1:
            if in_bool_tl and in_bool_tr and in_bool_bl and in_bool_br:
                pix_poly = get_pixel_intersect_shape()

                if pix_poly is None:
                    pix_area_arr[i] = 0
                    continue
                else:
                    continue

        # check corners 
        out_bool_tl = check_point_polygon(tl[0], tl[1], poly_outer)
        out_bool_tr = check_point_polygon(tr[0], tr[1], poly_outer)
        out_bool_bl = check_point_polygon(bl[0], bl[1], poly_outer)
        out_bool_br = check_point_polygon(br[0], br[1], poly_outer)
        

        # Case 1: All pixels True (i.e. TTTT)
        if out_bool_tl and out_bool_tr and out_bool_bl and out_bool_br:
            if in_bool_tl and in_bool_tr and in_bool_bl and in_bool_br:
                pix_area_arr[i] = 0

        pix_area_arr[i] = pixel_area(x_cen, y_cen, poly)

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)