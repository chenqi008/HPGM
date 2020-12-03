import time
import math
import cmath
import torch
import functools
import cv2 as cv
import numpy as np


def visualize(boxes_coord, boxes_type, save_pic_path):
    """
    boxes_coord:[[],[],..] boxes_type:[]
    box_collection: [(tensor([[boxes_coord],[boxes_coord],..]), [tensor,tensor,...])]
    example:[(tensor([[0.6235, 0.3686, 0.3373, 0.3686],
              [0.3373, 0.3686, 0.3373, 0.6941],
              [0.3373, 0.6941, 0.6235, 0.6941],
              [0.6235, 0.6941, 0.6235, 0.3686],
              [0.6235, 0.3686, 0.6235, 0.3686]]),
              [tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(0.)])]
    """
    # visualize the coord to check as follow
    import vutils
    boxes_type = [torch.tensor(boxes_type[i]) for i in range(len(boxes_type))]
    boxes_coord = torch.tensor(boxes_coord)
    boxes_collection = [(boxes_coord, boxes_type)]
    background = np.zeros((256, 256))
    vutils.save_bbox(background, boxes_collection, save_pic_path, normalize=True, draw_line=True)


def visualization(point_hulls, save_contour_path):
    hull_coords = []
    hull_types = []
    for n in range(len(point_hulls)):
        hull_coord, hull_type = trans_hull_boxes_coord(point_hulls[n], 1)
        for k in range(len(hull_coord)):
            hull_coords.append(hull_coord[k])
            hull_types.append(hull_type[k])
    visualize(hull_coords, hull_types, save_contour_path)


def draw_point(points_hulls, colors, save_img_path, center=False):
    """
    function: paint the point on a white image
    points_hulls: [[x,y],...]
    colors: BGR
    """
    image = np.ones((256, 256, 3)) * 255
    for k in range(len(points_hulls)):
        point_hulls = points_hulls[k]
        for i in range(len(point_hulls)):
            image = cv.circle(image, (int(point_hulls[i][0]), int(point_hulls[i][1])), 2, colors[k], -1)
    if center:
        point = get_center_point(point_hulls)
        image = cv.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255))
    cv.imwrite(save_img_path, image)


def check_contour_hull(point_hull):
    """
    function: check the hull point and correct hull points to make the point continual add x or y
    condition: the point and the next point is vertical or horizon so is x_same or y_same
    point_hull: [[x,y], ...]
    return: continual point_hull
    """
    def judge_contour_hull(point_hull):
        # function: 0 to judge the contour hull is not ok
        same_num = 0
        for i in range(point_hull.shape[0]):
            if i + 1 == point_hull.shape[0]:
                if point_hull[i][0] == point_hull[0][0] or point_hull[i][1] == point_hull[0][1]:
                    same_num = same_num + 1
            elif point_hull[i][0] == point_hull[i + 1][0] or point_hull[i][1] == point_hull[i + 1][1]:
                    same_num = same_num + 1
        if same_num == point_hull.shape[0]:
            return False
        else:
            return True
    last_type = "decide which is the same"
    while judge_contour_hull(point_hull):
        # change the first point_hull since:array([[191,  60],
        #        [ 96, 115], [ 95, 115], [ 95,  57],
        #        [105,  49], [106,  49]], dtype=int32) will make the algorithm lose influence
        for i in range(point_hull.shape[0]):
            compare = i + 1
            if i+1 == point_hull.shape[0]:
                compare = 0
            if point_hull[i][0] == point_hull[compare][0]:
                last_type = "x_same"
            elif point_hull[i][1] == point_hull[compare][1]:
                last_type = "y_same"
            else:
                if last_type == "x_same":
                    # last:x_same; now:y_same
                    point_hull = np.insert(point_hull, compare, [[point_hull[compare][0], point_hull[i][1]]], 0)
                    break
                elif last_type == "y_same":
                    point_hull = np.insert(point_hull, compare, [[point_hull[i][0], point_hull[compare][1]]], 0)
                    break
        if i + 1 != point_hull.shape[0]:
            last_type = "decide which is the same"
    return point_hull


def trans_hull_boxes_coord(point_hull, boxes_type):
    """
    function: trans the point_hull[[x,y], [x,y], ...] to [[x1, y1, x2, y2], ...]
    point_hull:
    """
    hull_coord = []
    hull_type = []
    point_hull = point_hull.astype(np.float32)
    for i in range(point_hull.shape[0]):
        if i == point_hull.shape[0] - 1:
            hull_coord.append([float('%.4f' % (point_hull[i][0] / 256)), float('%.4f' % (point_hull[i][1] / 256)),
                               float('%.4f' % (point_hull[0][0] / 256)),
                               float('%.4f' % (point_hull[0][1] / 256))])
        else:
            hull_coord.append([float('%.4f' % (point_hull[i][0] / 256)), float('%.4f' % (point_hull[i][1] / 256)),
                               float('%.4f' % (point_hull[i + 1][0] / 256)),
                               float('%.4f' % (point_hull[i + 1][1] / 256))])
        hull_type.append(boxes_type)
    return hull_coord, hull_type


def point_split_scale(contour_coord, bounding_box):
    """
    function: generate the random point per 1 scale in the contour_coord
    contour_coord: [[x,y], ...]
    bounding_box: [[x, y, w, h]]
    return: point_coords [[x,y], ...]
    """
    x, y, w, h = bounding_box[0]
    point_coords = []
    for i in range(x, x+w, 2):
        for j in range(y, y+h):
            # +1:inside the contour
            if cv.pointPolygonTest(contour_coord, (i, j), False) == +1:
                point_coords.append([i, j])
    point_coords = np.asarray(point_coords)
    return point_coords


def get_line_intersection(line1, line2, cross_point):
    """
    function: copy from https://blog.csdn.net/wcl0617/article/details/78654944
              get the cross_point of the two lines
    line1: [[x0,y0], [x1, y1]]
    line2: [[x2,y2], [x3, y3]]
    cross_point: save_path
    return: 0: no cross 1: cross
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
    if len(same_x_point) == 0 or len(same_y_point) == 0:
        return None, None
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
        if intersection_box_point is None:
            return None, None, None
    return intersection_box_point, rest_contour_coord, delete_point


def get_center_point(point_hull):
    # get the center point from the contour
    contour = cv.convexHull(point_hull, returnPoints=True)
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return [cx, cy]


def clock_wise(point_hull):
    # according to the cross result to judge the clockwise direction
    # it only can be use in convex ploygon
    class clockwise_two_point(object):
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
            # cross between OA and OB
            det = np.cross(a, b)
            # det = (a[0] - center_point[0]) * (b[1] - center_point[1]) - (b[0] - center_point[0]) * (a[1] - center_point[1])
            if det < 0:  # clockwise
                return 1
            if det > 0:
                return -1
            # OA and OB on line, judge by distance
            d1 = a[0] * a[0] + a[1] * a[1]
            d2 = b[0] * b[0] + b[1] * b[1]
            if d1 > d2:
                return 1
            else:
                return -1
    center_point = get_center_point(point_hull)
    clockwise_sort = clockwise_two_point(center_point)
    point_hull = sorted(point_hull, key=functools.cmp_to_key(clockwise_sort.compare))
    point_hull = np.array(point_hull)
    return point_hull


def get_room_contour(hull_conversion, boxes_coord, pair_id=1):
    point_hulls, point_bounding_boxes = [], []
    # get the whole contour
    contour_point_hull, contour_bounding_box = hull_conversion.get_contour_layout(boxes_coord, first=True, contour=True)
    # /****** get the random layer of the room******/
    contour_coord = np.squeeze(contour_point_hull)
    bounding_box = contour_bounding_box.copy()
    room_num = 4
    room_coords, room_bounding_boxes, pair_points = [], [], []
    """
        generate the random room, to avoid the decimal point influence:
        make the contour_coord int and the room int together
    """
    for i in range(room_num):
        point_coords = point_split_scale(contour_coord, bounding_box)
        while len(point_coords) < 20:
            point_coords = point_split_scale(contour_coord, bounding_box)
        random_room_contour, rest_contour, pair_point = hull_conversion.random_room_generate(point_coords, contour_coord, seed=pair_id*4+i)
        pair_points.append(pair_point)
        # random the room point hull
        room_point_hull, room_bounding_box = hull_conversion.get_contour_layout(random_room_contour, first=False, contour=False)
        room_coords.append(room_point_hull)
        room_bounding_boxes.append(room_bounding_box)
        contour_coord, _ = hull_conversion.get_contour_layout(rest_contour, first=False, contour=True)
    room_coords.append(contour_coord)
    room_bounding_boxes.append(contour_bounding_box)
    point_hulls.append(room_coords)
    point_bounding_boxes.append(room_bounding_boxes)
    return point_hulls, point_bounding_boxes, pair_points


class ConversionLayout(object):
    def __init__(self):
        pass

    def build_bounding_box_point(self, random_point, direction):
        # direction: 1:left up 2: right up 3: left bottom 4: right bottom
        direction_dict = {0: [1, 1], 1: [255, 1], 2: [1, 255], 3: [255, 255]}
        [x, y] = direction_dict[direction]
        bounding_box_point = np.asarray([[x, y], [random_point[0], y],
                                         random_point, [x, random_point[1]]]).astype(np.float32)
        return bounding_box_point

    @staticmethod
    def delete_redundancy_element(point_hull):
        # delete the same element
        delete_index = []
        for i in range(len(point_hull)):
            if i + 1 == len(point_hull):
                if list(point_hull[i]) == list(point_hull[0]):
                    delete_index.append(i)
            elif list(point_hull[i]) == list(point_hull[i + 1]):
                delete_index.append(i)
        point_hull = np.delete(point_hull, delete_index, axis=0)
        return point_hull

    @staticmethod
    def calculate_contour_area(polygon):
        """
        polygon: [[x,y], ...]
        return: contour_area per pixel
        """
        im = np.zeros((256, 256))
        polygon_mask = cv.fillPoly(im, [polygon], 255)
        area = np.sum(np.greater(polygon_mask, 0))
        return area

    @staticmethod
    def delete_redundancy_element(point_hull):
        # delete the same element
        delete_index = []
        for i in range(len(point_hull)):
            if i + 1 == len(point_hull):
                if list(point_hull[i]) == list(point_hull[0]):
                    delete_index.append(i)
            elif list(point_hull[i]) == list(point_hull[i + 1]):
                delete_index.append(i)
        point_hull = np.delete(point_hull, delete_index, axis=0)
        return point_hull

    def random_room_generate(self, contour_coord, random_point):
        """
        function: generate the different room one by one
        direction: 0:left up 1: right up 2: left bottom 3: right bottom
        contour_coord:
        random_point: [x, y]
        return: random_room_contour
        """
        pair_points = []
        direction = np.round(random_point[2])
        random_point = np.array(random_point[:2])
        # 1. get the common contour between bounding_box_point and contour_coord
        bounding_box_point = self.build_bounding_box_point(random_point, direction)
        contour_coord = np.array(contour_coord).astype(np.float32)
        num = 0
        for i in range(len(contour_coord)):
            # positive (inside), negative (outside), or zero (on an edge)
            if cv.pointPolygonTest(bounding_box_point, (contour_coord[i][0], contour_coord[i][1]), False) == +1:
                num = num + 1
        intersection_box_point, rest_contour_coord, delete_point = get_intersection_polygon(bounding_box_point,
                                                                                            contour_coord, direction,
                                                                                            num)
        if intersection_box_point is None or len(intersection_box_point) == 1:
            # print("the point is not in the contour!")
            return None
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
        first_contour_random_point = np.squeeze(cv.convexHull(first_contour_random_point, returnPoints=True)).astype(
            np.int32)
        if len(first_contour_random_point) < 3:
            # print("First contour random point have error!")
            return None
        first_contour_random_point = check_contour_hull(first_contour_random_point)
        # /***** new generate second contour layout by shapely *****/
        from shapely.ops import split
        from shapely.geometry import Polygon, LineString
        intersection_box_point = self.delete_redundancy_element(intersection_box_point)
        intersection_box_point = np.round(intersection_box_point)
        # if len(intersection_box_point) != 3:
        #     print("error happen! please check the intersection box point")
        line = LineString([intersection_box_point[0], intersection_box_point[-1], intersection_box_point[1]])
        contour_polygon = Polygon(contour_coord)
        contour_collection = list(split(contour_polygon, line))
        if len(contour_collection) > 1:
            second_contour_random_point = np.array(contour_collection[0].exterior.coords).astype(np.int32)
            first_contour_random_point = np.array(contour_collection[1].exterior.coords).astype(np.int32)
        else:
            # print("error happen Please check!")
            return None
            # assert False
            second_contour_random_point = second_contour_random_point.astype(np.int32)
            first_contour_random_point = first_contour_random_point.astype(np.int32)

        if self.calculate_contour_area(first_contour_random_point) > \
                self.calculate_contour_area(second_contour_random_point):
            random_room_contour = second_contour_random_point
            rest_contour = first_contour_random_point
        else:
            random_room_contour = first_contour_random_point
            rest_contour = second_contour_random_point
        random_room_contour = self.delete_redundancy_element(random_room_contour)
        rest_contour = self.delete_redundancy_element(rest_contour)
        if len(random_room_contour) < 3 or len(rest_contour) < 3:
            # print("Random point have error!")
            return None
        return [random_room_contour, rest_contour, pair_points]

    @staticmethod
    def get_contour_layout(boxes_coord, first=True, contour=False):
        """
        function: get the contour of the layout
        first: [x1, y1, x2, y2] --> [[x1 y1] [x2 y2] ...](numpy)(np.float32)
        return: point_hull, bounding_box
        """
        if first:
            point_set = np.zeros((boxes_coord.shape[0] * 2, 2), dtype=np.float32)
            for i in range(boxes_coord.shape[0]):
                x1, y1, x2, y2 = boxes_coord[i]
                point_set[2 * i] = [x1 * 256, y1 * 256]
                point_set[2 * i + 1] = [x2 * 256, y2 * 256]
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
            bounding_box = cv.boundingRect(boxes_coord)
            bounding_box = np.array([bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]], dtype=np.int)
            if contour:
                point_hull = boxes_coord
            else:
                hull = cv.convexHull(boxes_coord, returnPoints=False)
                point_hull = boxes_coord[hull]
                point_hull = np.squeeze(point_hull)
        bounding_box = np.expand_dims(bounding_box, axis=0)
        point_hull = check_contour_hull(point_hull)
        return point_hull, bounding_box

    def tran_layout_hull(self, layout_init_contour, layout):
        '''
        layout_init_contour:  [n, 3]
        layout: [num, (x,y,direction)]
        return:
        '''
        def judge_area(room_num, room_hull, init_contour):
            """
            function:calculate the score_area of the room
            """
            def calculate_surface_area(polygon):
                # use the image to calculate the area
                im = np.zeros((256, 256))
                polygon_mask = cv.fillPoly(im, [np.array(polygon, dtype=np.int)], 255)
                area = np.sum(np.greater(polygon_mask, 0))
                return area
            k_min, area_limitation = 1. / (2* room_num), 50
            area = float(calculate_surface_area(room_hull))
            total_area = float(calculate_surface_area(init_contour))
            k = total_area / area
            # print("area_ratio: ", k)
            if k > area_limitation:
                return False
        room_num = len(layout)
        room_coords = []
        contour_coord = None
        for i in range(room_num):
            if i == 0:
                contour_coord = (layout_init_contour[:, :2] * 255).int()
            random_point = [int(layout[i][0] * 255), int(layout[i][1] * 255), int(layout[i][2] * 4)]
            data = self.random_room_generate(contour_coord, random_point)
            if data is not None:
                [random_room_contour, rest_contour, pair_point] =data
            else:
                return -1
            # print(random_room_contour)
            # random the room point hull
            room_point_hull, room_bounding_box = self.get_contour_layout(random_room_contour, first=False, contour=False)
            room_coords.append(room_point_hull)
            # make the rest_contour be in clockwise
            contour_coord, _ = self.get_contour_layout(rest_contour, first=False, contour=True)
        room_coords.append(contour_coord)
        for i in range(len(room_coords)):
            k = judge_area(len(room_coords), room_coords[i], layout_init_contour*255)
            if k == False:
                return -1
        return room_coords


if __name__ == '__main__':
    start_time = time.time()
    hull_conversion = ConversionLayout()
    pair_nums = 50
    for i in range(1, 401):  # 1, 401
        data_boxes_path = "./dataset/data/first_layer_data/layout{}_boxes_coord.txt".format(i)
        boxes_coord = np.round(np.loadtxt(data_boxes_path), decimals=4)
        layer_pair_points = []
        layer_pair_hulls = []
        for n in range(pair_nums):
            point_3d_hulls, point_bounding_boxes, pair_points = get_room_contour(hull_conversion, boxes_coord, n)
            layer_pair_points.append(pair_points)
            layer_pair_hulls.append(point_3d_hulls)
        # /*************visualize in 2d to check right or not*********/
        # save_gt_path = "./result/layout{}_gt.png".format(i)
        # visualize(boxes_coord, boxes_type, save_gt_path)
        # for l in range(len(point_3d_hulls)-1):
        #     save_contour_path = "./result/layout{}_{}.png".format(i, l)
        #     visualization(point_3d_hulls[l+1], save_contour_path)
        save_train_data_path = "./multiLayer_data/pair_train_data/layers_pair_points_{}.npy".format(i)
        save_hull_data_path = "./multiLayer_data/pair_hull_data/layers_hull_points_{}.npy".format(i)
        layer_pair_points = np.array(layer_pair_points)
        layer_pair_hulls = np.array(layer_pair_hulls)
        np.save(save_train_data_path, layer_pair_points)
        np.save(save_hull_data_path, layer_pair_hulls)
        print("save {} layout is OK!".format(i))
    end_time = time.time()
    print("The total time spent %.4f s" % (end_time-start_time))
