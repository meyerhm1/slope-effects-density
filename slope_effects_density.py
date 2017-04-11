from __future__ import division

import numpy as np
import Polygon as pg
from Polygon.Shapes import Circle

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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

def part_of_old_main():
    """
    This is part of the main code that I was writing earlier,
    which was to be used with the get_pixel_area() function and others.
    I've kept it here so that I can come back to it should I ever need to.
    """

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

    return None

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
                            #if (in edge_inter_count) and (seg_count == total_segments - 1):
                            #    intersect_poly.append((pixel_corners[edge_count][0], pixel_corners[edge_count][1]))
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

    #else:
    #    # these are the couple cases where the pixel intersecting area is either 0 or 1
    #    # in this case there will be no polygon points that are inside the pixel
    #    if (case == 'tttt'):
    #        return 1

    #    elif (case == 'ffff'):
    #        return 0

    #    elif (case == 'tttf'):
    #        # dummy return # to be fixed
    #        return None

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
    print poly_i
    px_i, py_i = polygon_plot_prep(poly_i)

    ax.plot(px_i, py_i, '-', color='k', lw=3)

    # shade intersection region
    poly_i_patch = Polygon(poly_i[0], closed=True, fill=True, color='gray', alpha=0.5)
    ax.add_patch(poly_i_patch)

    # get bounding boxes to set limits automatically
    xmin1, xmax1, ymin1, ymax1 = poly1.boundingBox()
    xmin2, xmax2, ymin2, ymax2 = poly2.boundingBox()

    xmin = min(xmin1, xmin2) - 3
    xmax = max(xmax1, xmax2) + 3
    ymin = min(ymin1, ymin2) - 3  #0.1 * min(ymin1, ymin2)
    ymax = max(ymax1, ymax2) + 3  #0.1 * max(ymax1, ymax2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.show()

    return None

def plot_single_polygon(poly):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    px, py = polygon_plot_prep(poly)

    ax.plot(px, py, '-', color='k', lw=2)

    xmin, xmax, ymin, ymax = poly.boundingBox()

    xmin = xmin - 300
    xmax = xmax + 300
    ymin = ymin - 300
    ymax = ymax + 300

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.show()

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

def get_rows_columns(pix_x_cen_arr, pix_y_cen_arr):

    orig = pix_x_cen_arr[0]
    for j in range(1, len(pix_x_cen_arr)):
        if pix_x_cen_arr[j] == orig:
            columns = j
            break

    orig = pix_y_cen_arr[0]
    rows = 1
    for j in range(0, len(pix_y_cen_arr)):
        if pix_y_cen_arr[j] != orig:
            rows += 1
            orig = pix_y_cen_arr[j]

    return rows, columns

def closest_pixel_indices(xp, yp, X, Y):

    x_dist_arr = np.abs(X-xp)
    y_dist_arr = np.abs(Y-yp)
    idx_arr = np.where((x_dist_arr == np.min(x_dist_arr)) & (y_dist_arr == np.min(y_dist_arr)))

    row_idx, col_idx = int(idx_arr[0]), int(idx_arr[1])

    return row_idx, col_idx

def get_pixels_in_bbox(bbox, pix_x_cen_arr, pix_y_cen_arr, mode='run'):

    # get limits from the bounding box
    xmin = bbox[0]
    xmax = bbox[1]
    ymin = bbox[2]
    ymax = bbox[3]

    # turn the limits into search area for pixels
    # i.e. be conservative and search an additional 500 units on each side
    # the _s is for search
    xmin_s = xmin - 500
    xmax_s = xmax + 500
    ymin_s = ymin - 500
    ymax_s = ymax + 500

    # now look for pixels within the search area
    # all you need to do is to find the pixel indices at the four corners 
    # of the bounding box and you can easily populate the rest of the array.
    pix_bbox_x = []
    pix_bbox_y = []

    # first, create a coordinate grid
    x1d_short = np.arange(np.min(pix_x_cen_arr), np.max(pix_x_cen_arr)+1000.0, 1000.0)
    y1d_short = np.arange(np.min(pix_y_cen_arr), np.max(pix_y_cen_arr)+1000.0, 1000.0)

    X, Y = np.meshgrid(x1d_short, y1d_short)

    # second, find the pixel coords (and their indices) that
    # are closest to hte search corners
    len_x1d_arr = len(x1d_short)
    len_y1d_arr = len(y1d_short)
    
    bl_row_idx, bl_col_idx = closest_pixel_indices(xmin_s, ymin_s, X, Y)
    tr_row_idx, tr_col_idx = closest_pixel_indices(xmax_s, ymax_s, X, Y)
    tl_row_idx, tl_col_idx = closest_pixel_indices(xmin_s, ymax_s, X, Y)
    # the row and col indices in teh above lines are indices that will give you
    # the x and y values of the pixel center that is closest to the search corner
    # e.g. X[bl_row_idx, bl_col_idx] and Y[bl_row_idx, bl_col_idx] are the x and y
    # coords of the pixel closest to the bottom left corner of hte search area

    bl_xy_1d_idx = np.where((pix_x_cen_arr == X[bl_row_idx, bl_col_idx]) & (pix_y_cen_arr == Y[bl_row_idx, bl_col_idx]))[0]
    tr_xy_1d_idx = np.where((pix_x_cen_arr == X[tr_row_idx, tr_col_idx]) & (pix_y_cen_arr == Y[tr_row_idx, tr_col_idx]))[0]
    tl_xy_1d_idx = np.where((pix_x_cen_arr == X[tl_row_idx, tl_col_idx]) & (pix_y_cen_arr == Y[tl_row_idx, tl_col_idx]))[0]

    # uncomment the following lines to check
    # if the 1d array index assignment worked
    if mode == 'test':
        print '\n'
        print X[bl_row_idx, bl_col_idx], Y[bl_row_idx, bl_col_idx], pix_x_cen_arr[bl_xy_1d_idx], pix_y_cen_arr[bl_xy_1d_idx]
        print X[tr_row_idx, tr_col_idx], Y[tr_row_idx, tr_col_idx], pix_x_cen_arr[tr_xy_1d_idx], pix_y_cen_arr[tr_xy_1d_idx]
        print X[tl_row_idx, tl_col_idx], Y[tl_row_idx, tl_col_idx], pix_x_cen_arr[tl_xy_1d_idx], pix_y_cen_arr[tl_xy_1d_idx]

    # lastly, populate the pixel and corresponding index array
    # first populate the x and y arrays in bbox
    x_bbox_min = X[bl_row_idx, bl_col_idx]
    x_bbox_max = X[tr_row_idx, tr_col_idx]
    y_bbox_min = Y[bl_row_idx, bl_col_idx]
    y_bbox_max = Y[tr_row_idx, tr_col_idx]
    if mode == 'test':
        print "xmin and xmax values in search area:", x_bbox_min, x_bbox_max
        print "ymin and ymax values in search area:", y_bbox_min, y_bbox_max

    pix_bbox_x_short = np.arange(x_bbox_min, x_bbox_max+1000.0, 1000.0)
    pix_bbox_y_short = np.arange(y_bbox_min, y_bbox_max+1000.0, 1000.0)
    pix_bbox_x = pix_bbox_x_short
    pix_bbox_y = pix_bbox_y_short

    for u in range(len(pix_bbox_y_short) - 1):
        pix_bbox_x = np.append(pix_bbox_x, pix_bbox_x_short)

    for v in range(len(pix_bbox_x_short) - 1):
        pix_bbox_y = np.vstack((pix_bbox_y, pix_bbox_y_short))

    pix_bbox_y = pix_bbox_y.T.flatten()
    pix_bbox_y = pix_bbox_y[::-1]
    # I'm reversing the y array because the original pix_y_cen_arr is also reversed
    # i.e. it goes from max to min 
    if mode == 'test':
        print len(pix_bbox_x), len(pix_bbox_y)  # should be equal lengths

    # populate pixel index array
    pixel_indices = []

    rows, columns = get_rows_columns(pix_x_cen_arr, pix_y_cen_arr)
    rows_in_bbox, columns_in_bbox = get_rows_columns(pix_bbox_x, pix_bbox_y)
    # this will (almost?) always be square
    # the bounding box for hte circle will always be square 
    # but because I add 500 to the search area, that might 
    # cause the returned shape to be rectangular

    current_start = int(tl_xy_1d_idx)
    current_row_indices = np.arange(int(tl_xy_1d_idx), int(current_start + columns_in_bbox), 1)
    row_count = 0
    while 1:
        if row_count == rows_in_bbox:
            break

        for w in range(len(current_row_indices)):
            pixel_indices.append(current_row_indices[w])
        
        row_count += 1
        current_start += columns
        current_row_indices = np.arange(int(current_start), int(current_start + columns_in_bbox), 1)

    pixel_indices = np.asarray(pixel_indices)

    return pix_bbox_x, pix_bbox_y, pixel_indices

def crater_test(pix_x_cen_arr, pix_y_cen_arr):

    # define sample craters
    c1 = Circle(radius=2.5e5, center=(-3.4e6,0), points=128)
    c2 = Circle(radius=3.5e5, center=(-3.5e6,-0.1e6), points=128)
    c3 = Circle(radius=1e5, center=(-3.3e6,0), points=128)
    c4 = Circle(radius=2e5, center=(-2e6,-1.5e6), points=128)
    c5 = Circle(radius=1e5, center=(-2.2e6,-1.4e6), points=128)

    # plot all craters
    fig = plt.figure()
    ax = fig.add_subplot(111)

    c1x, c1y = polygon_plot_prep(c1)
    c2x, c2y = polygon_plot_prep(c2)
    c3x, c3y = polygon_plot_prep(c3)
    c4x, c4y = polygon_plot_prep(c4)
    c5x, c5y = polygon_plot_prep(c5)

    # do the crater calc
    craters_x = np.array([c1x, c2x, c3x, c4x, c5x])
    craters_y = np.array([c1y, c2y, c3y, c4y, c5y])
    all_craters = [c1,c2,c3,c4,c5]

    pix_crater_area = np.zeros(len(pix_x_cen_arr))

    for i in range(len(craters_x)):

        current_crater_x_cen = craters_x[i]
        current_crater_y_cen = craters_y[i]

        crater_poly = all_craters[i]

        pix_bbox_x, pix_bbox_y, pixel_indices =\
         get_pixels_in_bbox(crater_poly.boundingBox(), pix_x_cen_arr, pix_y_cen_arr, mode='test')

        # first check that the lengths of indices array and 
        # returned x and y array are equal and then
        # check if the x and y elements given by pixel indices
        # are indeed the elements in pix_bbox_x and pix_bbox_y
        print "Returned x, y, and index arrays are:"
        print pix_bbox_x
        print pix_bbox_y
        print pixel_indices

        if len(pix_bbox_x) == len(pix_bbox_y) == len(pixel_indices):
            print "Equal length. Now checking for equality of pixel coord values that are obtained via two different ways."
        else:
            print "Lengths:", len(pix_bbox_x), len(pix_bbox_y), len(pixel_indices)
            print "Returned arrays are not of equal length. Exiting."
            sys.exit(0)
        print np.array_equal(pix_bbox_x, pix_x_cen_arr[pixel_indices])
        print np.array_equal(pix_bbox_y, pix_y_cen_arr[pixel_indices])

        for j in range(len(pix_bbox_x)):

            current_pix_x_cen = pix_bbox_x[j]
            current_pix_y_cen = pix_bbox_y[j]

            # define a polygon using pixel corners in exactly the same way as done for the pixel fraction case
            tl_x = current_pix_x_cen - 5e2
            tr_x = current_pix_x_cen + 5e2
            bl_x = current_pix_x_cen - 5e2
            br_x = current_pix_x_cen + 5e2

            tl_y = current_pix_y_cen + 5e2
            tr_y = current_pix_y_cen + 5e2
            bl_y = current_pix_y_cen - 5e2
            br_y = current_pix_y_cen - 5e2

            tl = [tl_x, tl_y]
            tr = [tr_x, tr_y]
            bl = [bl_x, bl_y]
            br = [br_x, br_y]

            pixel_corners = [tl, tr, br, bl]  # top and bottom, left and right going clockwise

            pixel_corners = pg.Polygon(pixel_corners)

            # find the area of intersection between the pixel and crater
            inter_area = (pixel_corners & crater_poly).area()

            # find pixel index using pixel center to append to the correct array element
            pix_index = pixel_indices[j]
            pix_crater_area[pix_index] += inter_area

    pix_crater_area /= 1e6

    print np.where(pix_crater_area != 0)
    rows, columns = get_rows_columns(pix_x_cen_arr, pix_y_cen_arr)
    im = ax.imshow(pix_crater_area.reshape(rows, columns), cmap='bone')
    plt.colorbar(im, ax=ax)

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()

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
    #region_name='orientale', save=True, with_craters=False, show_rect=False)

    # define inner and outer polygons
    poly_outer = zip(vertices_x[valid_out], vertices_y[valid_out])
    poly_inner = zip(vertices_x[valid_in], vertices_y[valid_in])

    poly_outer = pg.Polygon(poly_outer)
    poly_inner = pg.Polygon(poly_inner)

    # ----------------  measure and populate pixel area fraction array  ---------------- # 
    # read in pixel slope info
    # this file also gives the x and y centers of pixels
    slope_arr = np.load(home + '/Documents/plots_codes_for_heather/slope_effects_files/3km_slope_points.npy')

    pix_x_cen_arr = slope_arr['pix_x_cen']
    pix_y_cen_arr = slope_arr['pix_y_cen']
    slope = slope_arr['slope_val']

    rows, columns = get_rows_columns(pix_x_cen_arr, pix_y_cen_arr)

    pix_centers = zip(pix_x_cen_arr, pix_y_cen_arr)
    pix_area_arr = np.zeros(len(pix_x_cen_arr))

    # define rectangles within which pixels can be skipped
    inner_rect_x = [-3.35e6, -2.55e6, -2.55e6, -3.35e6, -3.35e6]
    inner_rect_y = [-9.5e5, -9.5e5, -3.3e5, -3.3e5, -9.5e5]

    outer_rect_x = [-3.55e6, -2.32e6, -2.32e6, -3.59e6, -3.55e6]
    outer_rect_y = [-1.47e6, -1.47e6, 5e4, 5e4, -1.47e6]
    
    # loop over all pixels
    for i in range(len(pix_centers)):

        if (i % 100000) == 0.0:
            print '\r',
            print "At pixel number:",'{0:.2e}'.format(i),\
            "; time taken upto now:",'{0:.2f}'.format((time.time() - start)/60),"minutes.",
            sys.stdout.flush()

        # check if pixel center falls "well" inside the inner excluding rectangle
        if (min(inner_rect_x) + 500 < pix_x_cen_arr[i]) and (pix_x_cen_arr[i] < max(inner_rect_x) - 500) and \
        (min(inner_rect_y) + 500 < pix_y_cen_arr[i]) and (pix_y_cen_arr[i] < max(inner_rect_y) - 500):
            pix_area_arr[i] = 0.0
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

        pixel_corners = [tl, tr, br, bl]  # top and bottom, left and right going clockwise

        pixel_corners = pg.Polygon(pixel_corners)

        # find pixel intersection with outer polygon if it is outside the outer rectangle
        # find pixel intersection with inner polygon if it is inside the outer rectangle 
        # if the pixel intersects the outer rectangle then the area by default is 1

        # check if it is inside the outer rectangle 
        #if (min(outer_rect_x) + 500 < pix_x_cen_arr[i]) and (pix_x_cen_arr[i] < max(outer_rect_x) - 500) and \
        #(min(outer_rect_y) + 500 < pix_y_cen_arr[i]) and (pix_y_cen_arr[i] < max(outer_rect_y) - 500):
        #    # so now you know that it is inside the outer rectangle and outside the inner rectangle 
        #    # find intersection with inner polygon

        #    pix_area_arr[i] = (pixel_corners & poly_inner).area() / 1e6

        # check if the pixel is completely inside both polygons
        if ((pixel_corners & poly_inner).area() == 1e6) and ((pixel_corners & poly_outer).area() == 1e6):
            # if it is completely inside the inner polygon then it is not part of the
            # annulus of interest
            pix_area_arr[i] = 0.0
            continue

        # check if the pixel is completely outside both polygons
        if ((pixel_corners & poly_inner).area() == 0.0) and ((pixel_corners & poly_outer).area() == 0.0):
            pix_area_arr[i] = 0.0
            continue

        # check if the pixel is completely outside the inner polygon but completely inside the outer polygon
        if ((pixel_corners & poly_inner).area() == 0.0) and ((pixel_corners & poly_outer).area() == 1e6):
            # if it is outside the inner polygon but inside the outer one
            pix_area_arr[i] = 1.0
            continue

        # check if the pixel is completely inside the outer polygon but intersects the inner polygon
        if ((pixel_corners & poly_inner).area() < 1e6) and ((pixel_corners & poly_inner).area() != 0.0) and\
         ((pixel_corners & poly_outer).area() == 1e6):
            pix_area_arr[i] = (pixel_corners & poly_inner).area() / 1e6
            continue

        # check if the pixel is completely outside the inner polygon but intersects the outer polygon
        if ((pixel_corners & poly_outer).area() < 1e6) and ((pixel_corners & poly_outer).area() != 0.0) and\
         ((pixel_corners & poly_inner).area() == 0.0):
            pix_area_arr[i] = (pixel_corners & poly_outer).area() / 1e6
            continue

        # check to see if it is outside the outer rectangle
        #if (min(outer_rect_x) - 500 > pix_x_cen_arr[i]) or (pix_x_cen_arr[i] > max(outer_rect_x) + 500) or \
        #(min(outer_rect_y) - 500 > pix_y_cen_arr[i]) or (pix_y_cen_arr[i] > max(outer_rect_y) + 500):
        #    # so now you know that it is outside the outer rectangle and outside the inner rectangle 
        #    # find intersection with outer polygon

        #    pix_area_arr[i] = (pixel_corners & poly_outer).area() / 1e6

        #else:
        #    # this is the case only if the pixel is "on" the outer rectangle
        #    pix_area_arr[i] = 1.0

    np.save(home + '/Documents/plots_codes_for_heather/slope_effects_files/pix_area_frac.npy', pix_area_arr)
    print "\n","Pixel fractional area computation done and saved."
    print "Moving to craters now.", '\n'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(pix_area_arr.reshape(rows, columns), cmap='bone')
    plt.colorbar(im, ax=ax)

    fig.savefig(slopedir + 'pix_area_frac.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

    # ----------------  measure and populate crater pixel fraction array  ---------------- # 
    # loop over all craters
    # for each crater get its bounding box and 
    # loop over all pixels in that bounding box
    # find intersecting area for each pixel and keep a running sum
    pix_crater_area = np.zeros(len(pix_x_cen_arr))

    for i in range(len(craters_x)):

        print '\r',
        print 'Analyzing crater', i+1, "of", len(craters_x),
        sys.stdout.flush()

        current_crater_x_cen = craters_x[i]
        current_crater_y_cen = craters_y[i]

        current_crater_rad = craters_diam[i] / 2

        crater_poly = Circle(radius=current_crater_rad, center=(current_crater_x_cen,current_crater_y_cen), points=128)
        # the crater circle is approximated using a polygon of 128 points

        pix_bbox_x, pix_bbox_y, pixel_indices =\
         get_pixels_in_bbox(crater_poly.boundingBox(), pix_x_cen_arr, pix_y_cen_arr, mode='run')

        for j in range(len(pix_bbox_x)):

            current_pix_x_cen = pix_bbox_x[j]
            current_pix_y_cen = pix_bbox_y[j]

            # define a polygon using pixel corners in exactly the same way as done for the pixel fraction case
            tl_x = current_pix_x_cen - 5e2
            tr_x = current_pix_x_cen + 5e2
            bl_x = current_pix_x_cen - 5e2
            br_x = current_pix_x_cen + 5e2

            tl_y = current_pix_y_cen + 5e2
            tr_y = current_pix_y_cen + 5e2
            bl_y = current_pix_y_cen - 5e2
            br_y = current_pix_y_cen - 5e2

            tl = [tl_x, tl_y]
            tr = [tr_x, tr_y]
            bl = [bl_x, bl_y]
            br = [br_x, br_y]

            pixel_corners = [tl, tr, br, bl]  # top and bottom, left and right going clockwise

            pixel_corners = pg.Polygon(pixel_corners)

            # find the area of intersection between the pixel and crater
            inter_area = (pixel_corners & crater_poly).area()

            # find pixel index using pixel center to append to the correct array element
            pix_index = pixel_indices[j]
            pix_crater_area[pix_index] += inter_area

    pix_crater_area /= 1e6

    np.save(home + '/Documents/plots_codes_for_heather/slope_effects_files/crater_pix_frac.npy', pix_crater_area)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(pix_crater_area.reshape(rows, columns), cmap='bone')
    plt.colorbar(im, ax=ax)

    fig.savefig(slopedir + 'pix_crater_frac.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

    # ----------------  loop over all pixels and measure crater density  ---------------- # 
    # read in pix area and crater area files 
    pix_frac = np.load(home + '/Documents/plots_codes_for_heather/slope_effects_files/pix_area_frac.npy')
    crater_frac = np.load(home + '/Documents/plots_codes_for_heather/slope_effects_files/crater_pix_frac.npy')

    density = np.zeros(len(pix_centers))
    for i in range(len(pix_centers)):

        if pix_frac[i] != 0.0:
            density[i] = crater_frac[i] / pix_frac[i]
        elif pix_frac[i] == 0.0:
            density[i] = np.nan

    np.save(home + '/Documents/plots_codes_for_heather/slope_effects_files/density.npy', density.reshape(rows,columns))
    np.save(home + '/Documents/plots_codes_for_heather/slope_effects_files/slope_val.npy', slope.reshape(rows,columns))

    # plot pixel crater fraction with slope overplotted
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.log10(density), slope, 'o', color='k', markersize=1)

    fig.savefig(slopedir + 'density_slope.png', dpi=300, bbox_inches='tight')
    plt.show()

    # total run time
    print '\n'
    print "Total time taken --", (time.time() - start)/60, "minutes."
    sys.exit(0)