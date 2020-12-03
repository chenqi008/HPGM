'''
Description: for tran the Ground Layer data to the evaluator and generator train/test data
FilePath: /LayoutGenerator_Lited/dataset/GenerateData/GenerateTrainData.py
'''
import time
import math
import cmath
import torch
import functools
import cv2 as cv
import numpy as np
import os
import sys
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '../..')))
sys.path.append(dir_path)

def visualize(boxes_coords, boxes_types, save_pic_path):
    """
    function: visualize the coord to check as follow
    box_collection: [(tensor([[boxes_coord],[boxes_coord],..]), [tensor,tensor,...])]
    example:[(tensor([[0.6235, 0.3686, 0.3373, 0.3686],
              [0.3373, 0.3686, 0.3373, 0.6941],
              [0.3373, 0.6941, 0.6235, 0.6941],
              [0.6235, 0.6941, 0.6235, 0.3686],
              [0.6235, 0.3686, 0.6235, 0.3686]]),
              [tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(0.)])]
    """
    import vutils
    boxes_types_tensor = [torch.tensor(boxes_types[t]) for t in range(len(boxes_types))]
    boxes_coords = torch.tensor(boxes_coords)
    boxes_collection = [(boxes_coords, boxes_types_tensor)]
    background = np.zeros((256, 256))
    vutils.save_bbox(background, boxes_collection, save_pic_path, normalize=True, draw_line=True)


def visualization(point_hulls, save_path):
    hull_coords = []
    hull_types = []
    for num in range(len(point_hulls)):
        hull_coord, hull_type = trans_hull_boxes_coord(point_hulls[num], 1)
        for k in range(len(hull_coord)):
            hull_coords.append(hull_coord[k])
            hull_types.append(hull_type[k])
    visualize(hull_coords, hull_types, save_path)


def draw_point(point_hulls, save_img_path):
    """
    function: to paint a image on a white image
    """
    image = np.ones((256, 256)) * 255
    for i in range(len(point_hulls)):
        image = cv.circle(image, (int(point_hulls[i][0]), int(point_hulls[i][1])), 2, (0, 0, 255))
    # point = get_center_point(point_hulls)
    # image = cv.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 255))
    cv.imwrite(save_img_path, image)


def check_contour_hull(point_hull):
    """
    function: check the hull point and correct hull points to make the point continual add x or y
    condition: the point and the next point is vertical or horizon so is x_same or y_same
    """
    def judge_contour_hull(hull):
        # function: 0 to judge the contour hull is not ok
        same_num = 0
        for h in range(hull.shape[0]):
            if h + 1 == hull.shape[0]:
                if hull[h][0] == hull[0][0] or hull[h][1] == hull[0][1]:
                    same_num = same_num + 1
            elif hull[h][0] == hull[h + 1][0] or hull[h][1] == hull[h + 1][1]:
                same_num = same_num + 1
        if same_num == hull.shape[0]:
            return False
        else:
            return True
    last_type = "decide which is the same"
    while judge_contour_hull(point_hull):
        """
        change the first point_hull since:array([[191,  60],
            [ 96, 115], [ 95, 115], [ 95,  57],
            [105,  49], [106,  49]], dtypes=int32) will make the algorithm lose influence
        """
        judge_num = 0
        for s in range(point_hull.shape[0]):
            judge_num = judge_num + 1
            compare = s + 1
            if s + 1 == point_hull.shape[0]:
                compare = 0
            if point_hull[s][0] == point_hull[compare][0]:
                last_type = "x_same"
            elif point_hull[s][1] == point_hull[compare][1]:
                last_type = "y_same"
            else:
                if last_type == "x_same":
                    # last:x_same; now:y_same
                    point_hull = np.insert(point_hull, compare, [[point_hull[compare][0], point_hull[s][1]]], 0)
                    break
                elif last_type == "y_same":
                    point_hull = np.insert(point_hull, compare, [[point_hull[s][0], point_hull[compare][1]]], 0)
                    break
        if judge_num != point_hull.shape[0]:
            last_type = "decide which is the same"
    return point_hull


def get_contour_layout(coord, first=True, contour=False):
    """
    function: get the contour of the layout
    make up the point set format: [[x1 y1] [x2 y2] ...](numpy)(np.float32)
    """
    if first:
        point_set = np.zeros((coord.shape[0] * 2, 2), dtype=np.float32)
        for a in range(coord.shape[0]):
            x1, y1, x2, y2 = coord[a]
            point_set[2 * a] = [x1 * 256, y1 * 256]
            point_set[2 * a + 1] = [x2 * 256, y2 * 256]
        # bounding_box: (x, y, w, h)
        bounding_box = cv.boundingRect(point_set)
        bounding_box = np.array([bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]], dtype=np.int)
        if contour:
            point_set = np.unique(point_set, axis=0).astype(np.int)
            hull = cv.convexHull(point_set, returnPoints=False)
            point_hull = point_set[hull]
            point_hull = np.squeeze(point_hull)
        else:
            # only the point in order can use it
            point_hull = cv.approxPolyDP(point_set, 1, True)
            point_hull = np.squeeze(point_hull)
    else:
        # bounding_box: (x, y, w, h)
        bounding_box = cv.boundingRect(coord)
        bounding_box = np.array([bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]], dtype=np.int)
        if contour:
            point_hull = coord
        else:
            hull = cv.convexHull(coord, returnPoints=False)
            point_hull = coord[hull]
            point_hull = np.squeeze(point_hull)
    bounding_box = np.expand_dims(bounding_box, axis=0)
    point_hull = check_contour_hull(point_hull)
    return point_hull, bounding_box


def split_box_room_type(coord, types):
    """
    function: split the boxes_coord to different room
    """
    item = set(types)
    boxes_coords = []
    boxes_room_type = []
    for t in item:
        boxes_coords.append(coord[types == t])
        boxes_room_type.append(t)
    return boxes_coords, boxes_room_type


def trans_hull_boxes_coord(point_hull, boxes_types):
    """
    function: trans the point_hull[[[]]] to [[]]
    """
    hull_coord = []
    hull_type = []
    point_hull = np.array(point_hull).astype(np.float32)
    for i in range(point_hull.shape[0]):
        if i == point_hull.shape[0] - 1:
            hull_coord.append([float('%.4f' % (point_hull[i][0] / 256)), float('%.4f' % (point_hull[i][1] / 256)),
                               float('%.4f' % (point_hull[0][0] / 256)),
                               float('%.4f' % (point_hull[0][1] / 256))])
        else:
            hull_coord.append([float('%.4f' % (point_hull[i][0] / 256)), float('%.4f' % (point_hull[i][1] / 256)),
                               float('%.4f' % (point_hull[i + 1][0] / 256)),
                               float('%.4f' % (point_hull[i + 1][1] / 256))])
        hull_type.append(boxes_types)
    return hull_coord, hull_type

def point_on_contour_line(contour_coord, point):
    """
    function: point is on contour_coord line or not
    """
    for i in range(len(contour_coord)):
        if i + 1 == len(contour_coord): # the last coord next point should be the first
            contour_line = [contour_coord[i], contour_coord[0]]
        else:
            contour_line = [contour_coord[i], contour_coord[i + 1]]
        valid_line = np.array(contour_line[1]-contour_line[0])
        left_line = np.array(contour_line[0]-point)
        right_line = np.array(contour_line[1]-point)
        if np.cross(left_line, valid_line) == 0 and np.cross(right_line, valid_line) == 0: # all is on line or colinear
            return True
    return False

def point_split_scale(contour_coord, bounding_box):
    """
    function: generate the random points per 1 scale in the contour_coord in order to choose random point
    """
    x, y, w, h = bounding_box[0]
    point_coords = []
    for i in range(x, x + w):
        for j in range(y, y + h):
            # +1:inside the contour
            if cv.pointPolygonTest(contour_coord, (i, j), False) == +1:
                point_coords.append([i, j])
    point_coords = np.asarray(point_coords)
    return point_coords


def get_line_intersection(line1, line2, cross_point):
    """
    function: get the cross_point of the two lines consider endpoints
    copy from https://blog.csdn.net/wcl0617/article/details/78654944
    line:[[x1,y1], [x2,y2]]
    """
    [p0_x, p0_y] = line1[0]
    [p1_x, p1_y] = line1[1]
    [p2_x, p2_y] = line2[0]
    [p3_x, p3_y] = line2[1]
    s10_x = p1_x - p0_x
    s10_y = p1_y - p0_y
    s32_x = p3_x - p2_x
    s32_y = p3_y - p2_y
    de_norm = s10_x * s32_y - s32_x * s10_y # a X b
    if de_norm == 0:  # Parallel or collinear if cross = 0
        return 0
    de_norm_positive = de_norm > 0
    s02_x = p0_x - p2_x
    s02_y = p0_y - p2_y
    s_numer = s10_x * s02_y - s10_y * s02_x # a X (a1 - b1)
    if s_numer != 0 and (s_numer < 0) == de_norm_positive:  # a X (a1 - b1) same sign with a X b
        return 0
    t_numer = s32_x * s02_y - s32_y * s02_x  # b X (a1 - b1)
    if t_numer !=0 and (t_numer < 0) == de_norm_positive:
        return 0
    if np.fabs(s_numer) > np.fabs(de_norm) or np.fabs(t_numer) > np.fabs(de_norm):
        return 0
    t = t_numer / de_norm
    cross_point[0] = p0_x + (t * s10_x)
    cross_point[1] = p0_y + (t * s10_y)
    return 1

def check_intersection_polygon(intersection_point, contour_coord, direction, num=1):
    """
    function: to judge the repeat point which to delete
    Icon: use direction 0 1 for example 2 3 as the same
      0:* *    * 1: *    * *
        * *    *    *    * *
               |    |
        * * <--*    *--> * *
    """
    bounding_box_point, same_y_point, same_x_point, same_x_index, same_y_index, delete_point = \
        intersection_point[-1], [], [], [], [], []
    for i in range(len(intersection_point)-1):
        if intersection_point[i][1] == bounding_box_point[1]:
            same_y_point.append(intersection_point[i][0])
            same_y_index.append(i)
        elif intersection_point[i][0] == bounding_box_point[0]:
            same_x_point.append(intersection_point[i][1])
            same_x_index.append(i)
    # sorted the index used the point num
    zip_x = zip(same_x_point, same_x_index)
    zip_y = zip(same_y_point, same_y_index)
    zip_x = sorted(zip_x)
    zip_y = sorted(zip_y)
    same_x_point, same_x_index = zip(*zip_x)
    same_y_point, same_y_index = zip(*zip_y)
    # judge the farthest point is in the contour or not
    [x, y, w, h] = cv.boundingRect(np.array(intersection_point).astype(np.float32))
    x, y, w, h = bounding_box_point[0], bounding_box_point[1], w-1, h-1
    point_dict = {0: [x-w, y-h], 1:[x+w, y-h], 2:[x-w, y+h], 3:[x+w, y+h]}
    # first contour have the point1 then the second contour should delete the point1
    if cv.pointPolygonTest(contour_coord, (point_dict[direction][0], point_dict[direction][1]), False) != -1:
        # in the contour
        if num == 3:
            if len(same_x_point) > 1:
                if direction in [0, 1]:
                    delete_point.append(intersection_point[same_x_index[-1]])
                    intersection_point = np.delete(intersection_point, same_x_index[0], axis=0)
                elif direction in [2, 3]:
                    delete_point.append(intersection_point[same_x_index[0]])
                    intersection_point = np.delete(intersection_point, same_x_index[-1], axis=0)
            if len(same_y_point) > 1:
                if direction in [0, 2]:
                    delete_point.append(intersection_point[same_y_index[-1]])
                    intersection_point = np.delete(intersection_point, same_y_index[0], axis=0)
                elif direction in [1, 3]:
                    delete_point.append(intersection_point[same_y_index[0]])
                    intersection_point = np.delete(intersection_point, same_y_index[-1], axis=0)
        else:
            if len(same_x_point) > 1:
                if direction in [0, 1]:
                    delete_point.append(intersection_point[same_x_index[0]])
                    intersection_point = np.delete(intersection_point, same_x_index[-1], axis=0)
                elif direction in [2, 3]:
                    delete_point.append(intersection_point[same_x_index[-1]])
                    intersection_point = np.delete(intersection_point, same_x_index[0], axis=0)
            if len(same_y_point) > 1:
                if direction in [0, 2]:
                    delete_point.append(intersection_point[same_y_index[0]])
                    intersection_point = np.delete(intersection_point, same_y_index[-1], axis=0)
                elif direction in [1, 3]:
                    delete_point.append(intersection_point[same_y_index[-1]])
                    intersection_point = np.delete(intersection_point, same_y_index[0], axis=0)
    else:
        if num == 3:
            if len(same_x_point) > 1:
                if direction in [0, 1]:
                    delete_point.append(intersection_point[same_x_index[0]])
                    intersection_point = np.delete(intersection_point, same_x_index[-1], axis=0)
                elif direction in [2, 3]:
                    delete_point.append(intersection_point[same_x_index[-1]])
                    intersection_point = np.delete(intersection_point, same_x_index[0], axis=0)
            if len(same_y_point) > 1:
                if direction in [0, 2]:
                    delete_point.append(intersection_point[same_y_index[0]])
                    intersection_point = np.delete(intersection_point, same_y_index[-1], axis=0)
                elif direction in [1, 3]:
                    delete_point.append(intersection_point[same_y_index[-1]])
                    intersection_point = np.delete(intersection_point, same_y_index[0], axis=0)
        else:
            if len(same_x_point) > 1:
                if direction in [0, 1]:
                    delete_point.append(intersection_point[same_x_index[-1]])
                    intersection_point = np.delete(intersection_point, same_x_index[0], axis=0)
                elif direction in [2, 3]:
                    delete_point.append(intersection_point[same_x_index[0]])
                    intersection_point = np.delete(intersection_point, same_x_index[-1], axis=0)
            if len(same_y_point) > 1:
                if direction in [0, 2]:
                    delete_point.append(intersection_point[same_y_index[-1]])
                    intersection_point = np.delete(intersection_point, same_y_index[0], axis=0)
                elif direction in [1, 3]:
                    delete_point.append(intersection_point[same_y_index[0]])
                    intersection_point = np.delete(intersection_point, same_y_index[-1], axis=0)
    # process the situation:
    #          * which x, y is all > 2
    #          * how to choose the delete point
    #          |
    # * * < -- *
    if len(same_x_point) > 1 and len(same_y_point) > 1:
        remove_index = []
        delete_point = np.array(delete_point).astype(np.float32)
        for i in range(len(delete_point)):
            if cv.pointPolygonTest(np.array(intersection_point).astype(np.float32), (delete_point[i][0], delete_point[i][1]), False) == -1:
            # outside the points
                remove_index.append(i)
        delete_point = np.delete(np.array(delete_point), remove_index, axis=0)
    return intersection_point, delete_point


def get_intersection_polygon(bounding_box_point, contour_coord, direction, num=1):
    """
    bounding_box_point = np.asarray([[x, y], [random_point[0], y],
                                    random_point, [x, random_point[1]]])
    contour_coord: polygon contour
    fuction: get the polygon area between bouding_box_point and contour_coord
    """
    line = [[bounding_box_point[1], bounding_box_point[2]], [bounding_box_point[2], bounding_box_point[3]]]
    intersection_box_point, intersection_index, delete_point = [], [], None
    rest_contour_coord = contour_coord.copy()
    for i in range(len(contour_coord)):
        if i + 1 == len(contour_coord): # the last coord next point should be the first
            contour_line = [contour_coord[i], contour_coord[0]]
        else:
            contour_line = [contour_coord[i], contour_coord[i + 1]]
        for j in range(len(line)):
            cross_point = np.zeros(2)
            if get_line_intersection(line[j], contour_line, cross_point):
                intersection_box_point.append(cross_point)
                intersection_index.append(i + 1)
    # because the insert range will influence the result of the array
    inter_index_arg = np.argsort(intersection_index)
    intersection_index = sorted(intersection_index)
    for i in range(len(intersection_index)):
        rest_contour_coord = np.insert(rest_contour_coord, intersection_index[i] + i,
                                       [intersection_box_point[inter_index_arg[i]][0],
                                        intersection_box_point[inter_index_arg[i]][1]], 0)
    intersection_box_point.append(bounding_box_point[2])
    if len(intersection_box_point) > 3:
        intersection_box_point, delete_point = check_intersection_polygon(intersection_box_point, contour_coord, direction, num)
    return intersection_box_point, rest_contour_coord, delete_point


def build_bounding_box_point(random_point, direction):
    """
    direction: 1:left up 2: right up 3: left bottom 4: right bottom
    """
    direction_dict = {0: [1, 1], 1: [255, 1], 2: [1, 255], 3: [255, 255]}
    [x, y] = direction_dict[direction]
    bounding_box_point = np.asarray([[x, y], [random_point[0], y],
                                     random_point, [x, random_point[1]]]).astype(np.float32)
    return bounding_box_point


def get_center_point(point_hull):
    """
    function: get the center point from the contour
    """
    contour = cv.convexHull(point_hull, returnPoints=True)
    m = cv.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return [cx, cy]


def clock_wise(point_hull):
    """
    function: according to the cross result to judge the clockwise direction
    condition: it only can be use in convex polygon
    """
    class ClockwiseTwoPoints(object):
        def __init__(self, center_point):
            self.center_point = center_point

        def compare(self, num1, num2):
            center_point = self.center_point
            a = (num1 - center_point).copy()
            b = (num2 - center_point).copy()
            if (a[0] >= 0) and (b[0] < 0):
                return 1
            if (a[0] == 0) and (b[0] == 0):
                if a[1] > b[1]:
                    return 1
                else:
                    return -1
            # cross of vector a and b
            det = np.cross(a, b)
            # det = (a[0] - center_point[0]) * (b[1] - center_point[1]) - (b[0] - center_point[0]) * (a[1] - center_point[1])
            if det < 0:  # clockwise
                return 1
            if det > 0:
                return -1
            # vector OA is collinear with OB judge distance
            d1 = a[0] * a[0] + a[1] * a[1]
            d2 = b[0] * b[0] + b[1] * b[1]
            if d1 > d2:
                return 1
            else:
                return -1
    center = get_center_point(point_hull)
    clockwise_sort = ClockwiseTwoPoints(center)
    point_hull = sorted(point_hull, key=functools.cmp_to_key(clockwise_sort.compare))
    point_hull = np.array(point_hull)
    return point_hull


def random_point_generate(contour, point_coords, seed):
    """
    Returns: random point from the point coords
    """
    pair_points = []
    direction_index = [i for i in range(4)]
    point_coords_index = [i for i in range(len(point_coords))]
    np.random.seed(seed)
    random_point_index = np.random.choice(point_coords_index, 1)[0] # choose the random point
    random_point = point_coords[random_point_index]
    # while point_on_contour_line(contour, random_point): # to avoid the situation * --> * *
    #     random_point_index = np.random.choice(point_coords_index, 1)[0]  # choose the random point
    #     random_point = point_coords[random_point_index]
    direction = np.random.choice(direction_index, 1)[0] # choose the random direction
    pair_points.append(random_point)
    pair_points.append(direction)
    return pair_points


def calculate_contour_area(polygon):
    # calculate the area
    im = np.zeros((256, 256))
    polygon_mask = cv.fillPoly(im, [polygon], 255)
    area = np.sum(np.greater(polygon_mask, 0))
    return area

def delete_redundancy_element(point_hull):
    # delete the same element
    delete_index = []
    for i in range(len(point_hull)):
        if i + 1 == len(point_hull):
            if list(point_hull[i]) == list(point_hull[0]):
                delete_index.append(i)
        elif list(point_hull[i]) == list(point_hull[i+1]):
            delete_index.append(i)
    point_hull = np.delete(point_hull, delete_index, axis=0)
    return point_hull

def random_room_generate(random_point, direction, contour_coord):
    """
    function: generate the different room one by one according to the random point
    direction: 0:left up 1: right up 2: left bottom 3: right bottom
    """
    # 1. get the common contour between bounding_box_point and contour_coord
    bounding_box_point = build_bounding_box_point(random_point, direction)
    contour_coord = np.array(contour_coord).astype(np.float32)
    num = 0
    for i in range(len(contour_coord)):
        # positive (inside), negative (outside), or zero (on an edge)
        if cv.pointPolygonTest(bounding_box_point, (contour_coord[i][0], contour_coord[i][1]), False) == +1:
            num = num + 1
    intersection_box_point, rest_contour_coord, delete_point = get_intersection_polygon(bounding_box_point, contour_coord, direction, num)
    first_contour_random_point = np.asarray(intersection_box_point).astype(np.float32)
    second_contour_random_point = np.round(rest_contour_coord).copy()
    # 2. get the random point split to two contour
    # get the first_contour_point
    remove_point, remove_point_index, edge_point = [], [], []
    for i in range(len(contour_coord)):
        # positive (inside), negative (outside), or zero (on an edge)
        if cv.pointPolygonTest(bounding_box_point, (contour_coord[i][0], contour_coord[i][1]), False) == +1:
            remove_point.append([contour_coord[i][0], contour_coord[i][1]])
            first_contour_random_point = np.vstack((first_contour_random_point, contour_coord[i]))
    first_contour_random_point = np.squeeze(cv.convexHull(first_contour_random_point, returnPoints=True)).astype(np.int32)
    first_contour_random_point = check_contour_hull(first_contour_random_point)
    # /***** new generate second contour layout by shapely *****/
    from shapely.ops import split
    from shapely.geometry import Polygon, LineString
    intersection_box_point = delete_redundancy_element(intersection_box_point)
    intersection_box_point = np.round(intersection_box_point)
    if len(intersection_box_point) != 3:
        print("error happen! please check the intersection box point")
    line = LineString([intersection_box_point[0], intersection_box_point[-1], intersection_box_point[1]])
    contour_polygon = Polygon(contour_coord)
    contour_collection = list(split(contour_polygon, line))
    if len(contour_collection) > 1:
        second_contour_random_point = np.array(contour_collection[0].exterior.coords).astype(np.int32)
        first_contour_random_point = np.array(contour_collection[1].exterior.coords).astype(np.int32)
    else:
        print("error happen Please check!")
        # assert False
        second_contour_random_point = second_contour_random_point.astype(np.int32)
        first_contour_random_point = first_contour_random_point.astype(np.int32)
    # /***** old generate second contour layout code ******/
    # # because the first point choose the one, the rest one should be delete in second point contour
    #     if delete_point is not None:
    #         for i in range(len(delete_point)):
    #             remove_point.append([delete_point[i][0], delete_point[i][1]])
    #     # get the second_contour_point
    #     for i in range(len(second_contour_random_point)):
    #         for n in range(len(remove_point)):
    #             if list(second_contour_random_point[i]) == list(remove_point[n]):
    #                 remove_point_index.append(i)
    #     second_contour_random_point = np.delete(second_contour_random_point, remove_point_index, axis=0)
    #     second_contour_random_point = np.array(second_contour_random_point).astype(np.int32)
    #     second_contour_random_point = delete_redundancy_element(second_contour_random_point)
    #     second_contour_random_point = check_contour_hull(second_contour_random_point)
    #     second_contour_random_point, _ = get_contour_layout(second_contour_random_point, first=False, contour=True)
    # 3. calculate the two contour area to decide which be the rest contour
    if calculate_contour_area(first_contour_random_point) > \
            calculate_contour_area(second_contour_random_point):
        random_room_contour = second_contour_random_point
        rest_contour = first_contour_random_point
    else:
        random_room_contour = first_contour_random_point
        rest_contour = second_contour_random_point
    # /******try to clockwise the point but fail******/
    # rest_contour = np.unique(rest_contour, axis=0)
    # random_room_contour = np.unique(random_room_contour, axis=0)
    # rest_contour = clock_wise(rest_contour)
    # random_room_contour = clock_wise(random_room_contour)
    random_room_contour = delete_redundancy_element(random_room_contour)
    rest_contour = delete_redundancy_element(rest_contour)
    return random_room_contour, rest_contour


def get_room_contour(boxes_coord, boxes_type, pair_id=1):
    """
    Args:
        boxes_coord: original first layer coords
        boxes_type: every room type
        pair_id: random rand
    """
    point_3d_hulls, point_bounding_boxes = [], []
    # get the whole contour
    contour_point_hull, contour_bounding_box = get_contour_layout(boxes_coord, first=True, contour=True)
    # # /******1. get the first layer of the room******/
    # # use the boxes_type to split the box_coord to different room coords
    boxes_room_coords, boxes_room_type = split_box_room_type(boxes_coord, boxes_type)
    point_3d_hulls.append(list(contour_point_hull))
    point_bounding_boxes.append(contour_bounding_box)
    # get the split room contour
    first_room_point_hull, first_room_point_types, first_room_bounding_box = [], [], []
    for i in range(len(boxes_room_coords)):
        room_point_hull, room_bounding_box = get_contour_layout(boxes_room_coords[i])
        first_room_point_hull.append(room_point_hull)
        first_room_point_types.append(int(boxes_room_type[i]))
        first_room_bounding_box.append(room_bounding_box)
    point_3d_hulls.append(first_room_point_hull)
    point_bounding_boxes.append(first_room_bounding_box)
    # /******2. get the second(random) layer of the room******/
    contour_coord = np.squeeze(contour_point_hull)
    bounding_box = contour_bounding_box.copy()
    # set room num as the boxes_room
    second_room_num = len(boxes_room_coords)-1
    second_room_coords, second_room_bounding_boxes, pair_points = [], [], []
    # generate the random room, to avoid the decimal point influence:
    # make the contour_coord int and the room int together
    for i in range(second_room_num):
        # print("room_index:", i)
        point_coords = point_split_scale(contour_coord, bounding_box)
        pair_point = random_point_generate(contour_coord, point_coords, seed=pair_id * second_room_num + i)
        pair_points.append(pair_point)
        random_room_contour, rest_contour = random_room_generate(pair_point[0], pair_point[1], contour_coord)
        room_point_hull, room_bounding_box = get_contour_layout(random_room_contour, first=False, contour=False)
        second_room_coords.append(room_point_hull)
        second_room_bounding_boxes.append(room_bounding_box)
        # make the rest_contour be in clockwise
        # ****make the rest_contour be in clockwise
        contour_coord, _ = get_contour_layout(rest_contour, first=False, contour=True)
        # # ***visualization for test***
        # room_path = "./result/layout_rest_{}.png".format(i)
        # visualization([contour_coord], room_path)
    second_room_coords.append(contour_coord)
    second_room_bounding_boxes.append(contour_bounding_box)
    point_3d_hulls.append(second_room_coords)
    point_bounding_boxes.append(second_room_bounding_boxes)
    # # /******3. get the third random layer of the room******/
    # contour_coord = np.squeeze(contour_point_hull)
    # bounding_box = contour_bounding_box
    # third_room_num = 4  # len(boxes_room_coords)+2
    # third_room_coords, third_room_bounding_boxes = [], []
    # # generate the random room, to avoid the decimal point influence:
    # # make the contour_coord int and the room int together
    # for i in range(third_room_num):
    #     point_coords = point_split_scale(contour_coord, contour_bounding_box)
    #     while len(point_coords) < 20:
    #         point_coords = point_split_scale(contour_coord, bounding_box)
    #     random_room_contour, rest_contour, _ = random_room_generate(point_coords, contour_coord, seed=i+second_room_num)
    #     # random the room point hull
    #     room_point_hull, room_bounding_box = get_contour_layout(random_room_contour, first=False, contour=False)
    #     third_room_coords.append(room_point_hull)
    #     third_room_bounding_boxes.append(room_bounding_box)
    #     # make the rest_contour be in clockwise
    #     contour_coord, _ = get_contour_layout(rest_contour, first=False, contour=True)
    # third_room_coords.append(contour_coord)
    # third_room_bounding_boxes.append(contour_bounding_box)
    # point_3d_hulls.append(third_room_coords)
    # point_bounding_boxes.append(third_room_bounding_boxes)
    return point_3d_hulls, first_room_point_types, point_bounding_boxes, pair_points


if __name__ == '__main__':
    def load_data_path(data_boxes_path, data_boxes_type_path):
        coord = np.round(np.loadtxt(data_boxes_path), decimals=4)
        types = np.loadtxt(data_boxes_type_path)
        return coord, types
    # id_path = os.path.join("/home/zhoujiaqiu/Code/GAN/Layout_bbox_gcn/multiLayerLayout/multiLayer_data", "test_id.txt")
    # id = np.loadtxt(id_path)
    start_time = time.time()
    absolute_path = "/home/datasets/MultiLayerDataset"
    pair_nums = 50
    for index in range(1, 401):  # 1, 401
        # if index not in id:
        #     continue
        boxes_path = os.path.join(absolute_path, "OriginData/layout_gt_coord_type/layout{}_boxes_coord.txt".format(index))
        boxes_type_path = os.path.join(absolute_path, "OriginData/layout_gt_coord_type/layout{}_boxes_type.txt".format(index))
        boxes_data_coords, boxes_data_types = load_data_path(boxes_path, boxes_type_path)
        layer_pair_points = []
        layer_pair_hulls = []
        for p in range(pair_nums):
            # print("now process the {} contour".format(p))
            point_data_hulls, point_types, bounding_boxes, pair_hull_points = get_room_contour(boxes_data_coords, boxes_data_types, p)
            layer_pair_points.append(pair_hull_points)
            layer_pair_hulls.append(point_data_hulls)
        # /************* visualize in 2d to check right or not*********/
        # save_gt_path = "./result/layout{}_gt.png".format(index)
        # visualize(boxes_data_coords, boxes_data_types, save_gt_path)
        # for l in range(len(layer_pair_hulls[0])-1):
        #     save_contour_path = "./result/layout{}_{}.png".format(index, l)
        #     print("{} layer save OK!".format(index))
        #     visualization(layer_pair_hulls[0][l+1], save_contour_path)

        # /*********** save the result to see 3d layout***********/
        # save_boxes_path = "./data/point_3d_hulls_types/layout{:0=4}_random_point_3d_hulls.npy".format(i)
        # save_boxes_type_path = "./data/point_3d_hulls_types/layout{:0=4}_type_point_3d_hulls.npy".format(i)
        # save_boxes_bounding_path = "./data/point_3d_hulls_types/layout{:0=4}_bounding_boxes_point_3d_hulls.npy".format(i)
        # np.save(save_boxes_path, np.asarray(point_3d_hulls))
        # np.save(save_boxes_type_path, np.asarray(point_types))
        # np.save(save_boxes_bounding_path, np.asarray(point_bounding_boxes))

        # /******* save train data *******/
        save_train_data_path = os.path.join(absolute_path, "GeneratorTrainData/pair_train_data_gt/layers_pair_points_{}.npy".format(index))
        save_hull_data_path = os.path.join(absolute_path, "GeneratorTrainData/pair_hull_data_gt/layers_hull_points_{}.npy".format(index))
        layer_pair_points = np.array(layer_pair_points)
        layer_pair_hulls = np.array(layer_pair_hulls)
        np.save(save_train_data_path, layer_pair_points)
        np.save(save_hull_data_path, layer_pair_hulls)
        print("save {} layout is OK!".format(index))
        print("pair_num: ", len(layer_pair_hulls))
    end_time = time.time()
    print("The total time spent %.4f s" % (end_time - start_time))
