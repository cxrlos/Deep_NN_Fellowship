""" UTILITY FUNCTIONS FOR SPATIAL PROCESSING

Include here all the functionalities regarding rectangles, distances, angles, etc.

"""

import math
import numpy as np


def l2_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def rect_tr(r):
    return (r[0] + r[2], r[1])


def rect_tl(r):
    return (r[0], r[1])


def rect_bl(r):
    return (r[0], r[1] + r[3])


def rect_br(r):
    return (r[0] + r[2], r[1] + r[3])


def rect_center(r):
    return (int(round(r[0] + r[2]/2)), int(round(r[1] + r[3]/2)))


def rect_center_f(r):
    return (r[0] + r[2]/2, r[1] + r[3]/2)


def rect_divide_in_n_rects(rect, N, orientation='vertical'):
    if orientation == 'vertical':
        w = rect[2] / N
        return [(int(round(rect[0]+i*w)), rect[1], int(round(w)), rect[3]) for i in range(N)]
    else:
        print('incorrect orientation, not implemented yet')
        return None


def rect_to_corners_pts(r):
    p1 = (r[0], r[1])
    p2 = (r[0] + r[2], r[1])
    p3 = (r[0] + r[2], r[1] + r[3])
    p4 = (r[0], r[1] + r[3])
    return [p1, p2, p3, p4]


def rect_from_corner_pts(p1, p2):
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p1[0] - p2[0])
    h = abs(p1[1] - p2[1])
    return (x, y, w, h)


def rect_IoU(r1, r2):
    intersection = rect_intersection(r1, r2)
    if intersection is not None:
        return rect_area(intersection) / (rect_area(r1) + rect_area(r2) - rect_area(intersection))

    return 0


def rect_IoSelf(r1, r2):
    intersection = rect_intersection(r1, r2)
    
    if intersection is not None:
        # print('ioSelf', intersection, rect_area(intersection), rect_area(r1), rect_area(intersection) / rect_area(r1))
        return rect_area(intersection) / rect_area(r1)
    
    return 0


def rect_area(a):
    return a[2] * a[3]


def horizontal_rect_intersection(a, b, img_width):
    adjusted_a = (0, a[1], img_width, a[3])
    adjusted_b = (0, b[1], img_width, b[3])
    horizontal_intersection = rect_intersection(adjusted_a, adjusted_b)
    return horizontal_intersection


def horizontal_rect_intersection_rate_over_minor(a, b, img_width):
    """ Function for computing the rate of the height of the roi intersection 
        of two roi named a and b a, and the height of the roi with minimum roi 
        between a and b

    Params:
        a: a roi
        b: a roi
        img_width:  the width of the image that is being processed

    Returns:
        rate:   the rate of the width of the roi intersection of two roi
                named a and b a and the width of the roi with minimum roi 
                between a and b

    """
    horizontal_intersection = horizontal_rect_intersection(a, b, img_width)
    if horizontal_intersection is not None:
        vertical_union = a[3] if a[3] < b[3] else b[3]
        return horizontal_intersection[3] / vertical_union
    else:
        return 0


def vertical_rect_intersection(a, b, img_height):
    adjusted_a = (a[0], 0, a[2], img_height)
    adjusted_b = (b[0], 0, b[2], img_height)
    vertical_intersection = rect_intersection(adjusted_a, adjusted_b)
    if vertical_intersection is not None:
        return vertical_intersection
    else:
        return None


def vertical_rect_intersection_rate_over_minor(a, b, img_height):
    """ Function for computing the rate of the width of the roi intersection 
        of two roi named a and b a, and the width of the roi with minimum roi 
        between a and b

    Params:
        a: a roi
        b: a roi
        img_height:  the height of the image that is being processed

    Returns:
        rate:   the rate of the height of the roi intersection of two roi
                named a and b a and the height of the roi with minimum roi
                between a and b

    """
    vertical_intersection = vertical_rect_intersection(a, b, img_height)
    if vertical_intersection is not None:
        horizontal_union = a[2] if a[2] < b[2] else b[2]
        return vertical_intersection[2] / horizontal_union
    else:
        return None


def rect_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return None  # (0,0,0,0)
    return (x, y, w, h)


def rect_union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def boundingRect_f(pts):
    if len(pts) > 1:
        min_x = min([x[0] for x in pts])
        max_x = max([x[0] for x in pts])
        min_y = min([x[1] for x in pts])
        max_y = max([x[1] for x in pts])
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        return (0, 0, 0, 0)


def distance_between_rects_corner(r1, r2, function_rect_corner):
    return l2_distance(function_rect_corner(r1), function_rect_corner(r2))


def distance_between_rects_center(r1, r2):
    p1 = (r1[0] + r1[2]/2, r1[1] + r1[3]/2)
    p2 = (r2[0] + r2[2]/2, r2[1] + r2[3]/2)
    return l2_distance(p1, p2)


def angle_trunc(a):
    while a < 0.0:
        a += math.pi * 2
    return a


def angle_between_vector_and_xaxis(p):
    ''' Angle of vector p relative to X axis [-pi/2, +pi/2]
    '''
    return math.atan2(p[1], p[0])


def angle_between_points(x_orig, y_orig, x_landmark, y_landmark):
    delta_y = y_landmark - y_orig
    delta_x= x_landmark - x_orig
    return angle_trunc(math.atan2(delta_y, delta_x))


def shortest_angular_distance(a1, a2):
    a = a1 - a2
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return abs(a)


def check_line_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # find matrix determinant
    d = ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))    
    if d == 0:
        return 0
    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / d
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / d
    # if lines do intersect
    if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
        return 1
    # if lines do not intersect
    else:
        return -1


def check_line_rect_intersection(rect, x1, y1, x2, y2, only_parallel=False):
    count = 0    
    pt1x = rect[0]
    pt1y = rect[1]
    pt2x = rect[0] + rect[2]
    pt2y = rect[1]
    pt3x = rect[0] + rect[2]
    pt3y = rect[1] + rect[3]
    pt4x = rect[0]
    pt4y = rect[1] + rect[3]    
    if only_parallel:
        # Check parallel sides
        if check_line_line_intersection(x1, y1, x2, y2, pt1x, pt1y, pt2x, pt2y) > 0 and \
              check_line_line_intersection(x1, y1, x2, y2, pt3x, pt3y, pt4x, pt4y) > 0: 
            count = 2
        if check_line_line_intersection(x1, y1, x2, y2, pt2x, pt2y, pt3x, pt3y) > 0 and \
              check_line_line_intersection(x1, y1, x2, y2, pt1x, pt1y, pt4x, pt4y) > 0: 
            count = 2
    else:
        # Check the four sides of the rectangle
        if check_line_line_intersection(x1, y1, x2, y2, pt1x, pt1y, pt2x, pt2y) > 0: 
            count += 1
        if check_line_line_intersection(x1, y1, x2, y2, pt2x, pt2y, pt3x, pt3y) > 0: 
            count += 1
        if check_line_line_intersection(x1, y1, x2, y2, pt3x, pt3y, pt4x, pt4y) > 0: 
            count += 1
        if check_line_line_intersection(x1, y1, x2, y2, pt1x, pt1y, pt4x, pt4y) > 0: 
            count += 1        
    return count


def shrink_rect(r, p):
    if p >= 0.5 or p < 0:
        return r
    x_offset = round(r[2] * p)
    y_offset = round(r[3] * p)
    x = r[0] + x_offset
    y = r[1] + y_offset
    w = r[2] - 2 * x_offset
    h = r[3] - 2 * y_offset
    return (x, y, w, h)


def grow_line(l, epsilon):
    a = angle_between_points(l[0], l[1], l[2], l[3])
    length = line_length(l)/2 + epsilon
    center_pt = (round(l[0] + l[2])/2, round(l[1] + l[3])/2)
    x1 = center_pt[0] + length * np.cos(a)
    y1 = center_pt[1] + length * np.sin(a)
    x2 = center_pt[0] + length * np.cos(a + np.pi)
    y2 = center_pt[1] + length * np.sin(a + np.pi)
    return (x1, y1, x2, y2)


def line_length(line):
    return l2_distance((line[0], line[1]), (line[2], line[3]))


def position_point_to_line(pt, line):
    '''
    If point is at left of line, the function returns a negative number
    If point is at right of line, the function returns a positive number
    If point is over line, the function returns zero
    '''
    return (pt[0] - line[0]) * (line[3] - line[1]) - (pt[1] - line[1]) * (line[2] - line[0])


