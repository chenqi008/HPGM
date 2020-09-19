'''
Description: Generator points from the ground layer contour data
FilePath: /LayoutGenerator_Lited/dataset/GenerateData/contourTranPoint.py
'''
import time
import os
import sys
import torch
import cv2 as cv 
import numpy as np
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '../../..')))
sys.path.append(dir_path)
from miscc.utils import bbox_iou
from GenerateTrainData import get_contour_layout, split_box_room_type, \
    visualization, get_intersection_polygon, random_room_generate, \
    point_on_contour_line, build_bounding_box_point, calculate_contour_area, get_center_point, \
    check_contour_hull, draw_point

def check_point_hull_inter(point_hull, contour_coord, check_num):
    """
    function:
    1. check_num <= 3: check the point hull if it still have two or one inter point
    2. check_num >= 3: return the max inter point index
    """
    nums = []
    for i in range(len(point_hull)):
        current_room = point_hull[i]
        num = 0
        for j in range(len(current_room)):
            if cv.pointPolygonTest(contour_coord, (current_room[j][0],
                                                   current_room[j][1]), False) == +1:  # in the contour
                num = num + 1
        nums.append(num)
        if num == check_num and check_num <= 3:
            return True
    # if check_num >= 3:  # if inter point too much
    #     return nums.index(max(nums))

def judge_direction(room_contour, judge_point):
    """
    function: to judge the point direction which can generate to room_contour
    """
    direction = [0, 1, 2, 3]
    for di in direction:
        bounding_box_point = build_bounding_box_point(judge_point, di)
        num = 0
        for i in range(len(room_contour)):
            if cv.pointPolygonTest(bounding_box_point, (room_contour[i][0],
                                   room_contour[i][1]), False) == 0:  # on the contour
                num = num + 1
        if num == 3: # the direction which contain 3 will be good
            return di

if __name__ == '__main__':
    def load_data_path(data_boxes_path, data_boxes_type_path):
        coord = np.round(np.loadtxt(data_boxes_path), decimals=4)
        types = np.loadtxt(data_boxes_type_path)
        return coord, types
    start_time = time.time()
    debug = 0
    for index in range(1, 401):  # 1, 401
        absolute_path = "/home/zhoujiaqiu/Code/GAN/MultiLayerDataset"
        boxes_path = os.path.join(absolute_path, "OriginData/layout_gt_coord_type/layout{}_boxes_coord.txt".format(index))
        boxes_type_path = os.path.join(absolute_path, "OriginData/layout_gt_coord_type/layout{}_boxes_type.txt".format(index))
        boxes_data_coords, boxes_data_types = load_data_path(boxes_path, boxes_type_path)
        # get the whole contour
        contour_point_hull, contour_bounding_box = get_contour_layout(boxes_data_coords, first=True, contour=True)
        # use the boxes_type to split the box_coord to different room coords
        boxes_room_coords, boxes_room_type = split_box_room_type(boxes_data_coords, boxes_data_types)
        # get the split room contour
        first_room_point_hull, first_room_point_types, first_room_bounding_box = [], [], []
        for i in range(len(boxes_room_coords)):
            room_point_hull, room_bounding_box = get_contour_layout(boxes_room_coords[i])
            first_room_point_hull.append(room_point_hull.astype(int))
            first_room_point_types.append(int(boxes_room_type[i]))
            first_room_bounding_box.append(room_bounding_box)
        # save the result to visualize
        # save_path = os.path.join(absolute_path, "result/visual_gt/first_layers_{}.png".format(index))
        # visualization(first_room_point_hull, save_path)
        # /**************** tran the contour to the split points ****************/
        # save for the intermediate result
        save_inter_path = os.path.join(absolute_path, "result/debug/first_layers_{}.png".format(index))
        point_hull = first_room_point_hull.copy()
        pointSet, contour_hulls, direction, num, cnum = [], [], [0, 1, 2, 3], 0, 0
        while len(point_hull) != 0:
            if cnum == 0: # first time to process
                contour_coord = contour_point_hull
                new_contour = contour_point_hull
            else:
                contour_coord = new_contour
            if num >= len(point_hull):
                num = 0
            current_room = point_hull[num]
            spilt_points, inter_num, inter_max_num, del_index = [], 0, 0, 0
            for j in range(len(current_room)):
                if cv.pointPolygonTest(contour_coord, (current_room[j][0],
                    current_room[j][1]), False) == +1: # in the contour
                        inter_num = inter_num + 1
                        spilt_points.append(np.array([current_room[j][0], current_room[j][1]]))
            # 1.del the max area in the contour
            if cnum == 0: # first time to process the contour
                max_area = 0
                for i in range(len(point_hull)):
                    area = calculate_contour_area(point_hull[i])
                    if area > max_area:
                        del_index = i
                        max_area = area
                cnum = cnum + 1
                point_hull.pop(del_index)
                continue
            if inter_num == 1:
                # 2. calculate the situation where only have one point in the contour
                spilt_point, max_iou, save_d, save_contour, save_room = spilt_points[0], 0, 0, None, None
                for d in range(len(direction)):
                    room_contour, rest_contour = random_room_generate(spilt_point, direction[d], contour_coord)
                    room_point_hull, room_bounding_box = get_contour_layout(room_contour, first=False, contour=False)
                    new_contour, _ = get_contour_layout(rest_contour, first=False, contour=True)
                    [ x, y, w, h] = cv.boundingRect(current_room)
                    [ x1, y1, w1, h1] = room_bounding_box[0]
                    iou = bbox_iou(torch.tensor([[x, y, x+w, y+h]], dtype=float)/255,
                                   torch.tensor([[x1, y1, x1+w1, y1+h1]], dtype=float)/255)[0]
                    if iou > max_iou:
                        max_iou = iou
                        save_d = d
                        save_contour = new_contour
                        save_room = room_point_hull
                pointSet.append([spilt_point, direction[d]])
                point_hull.pop(num)
                new_contour = save_contour
                cnum = cnum + 1
                contour_hulls.append(save_room)
                if debug:
                    visualization([save_room],  os.path.join(absolute_path, "result/debug/first_{}_{}_{}.png".
                                                                format(inter_num, direction[d], cnum)))
                    visualization([save_contour],  os.path.join(absolute_path, "result/debug/contour_{}_{}_{}.png".
                                                                format(inter_num, direction[d], cnum)))
            elif inter_num == 2:
                # if have not 1 inter point
                if check_point_hull_inter(point_hull, contour_coord, 1) or \
                        check_point_hull_inter(point_hull, contour_coord, 0)  :
                    num = num + 1
                    continue
                # 3. approximate the contour deliver by 2 points using one point
                spilt_ious, biggest_iou, save_p, save_d, save_contour, save_room = [], 0, 0, 0, None, None
                for p in range(inter_num):
                    spilt_point = spilt_points[p]
                    for d in range(len(direction)):
                        room_contour, rest_contour = random_room_generate(spilt_point, direction[d], contour_coord)
                        room_point_hull, room_bounding_box = get_contour_layout(room_contour, first=False, contour=False)
                        new_contour, _ = get_contour_layout(rest_contour, first=False, contour=True)
                        [ x, y, w, h] = cv.boundingRect(current_room)
                        [ x1, y1, w1, h1] = room_bounding_box[0]
                        iou = bbox_iou(torch.tensor([[x, y, x + w, y + h]], dtype=float) / 255,
                                       torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=float) / 255)[0]
                        # choose the split point and direction which have the biggest iou
                        if  iou > biggest_iou:
                            biggest_iou = iou
                            save_d, save_p, save_contour, save_room = d, p, new_contour, room_contour
                new_contour = save_contour
                pointSet.append([spilt_points[save_p], direction[save_d]])
                point_hull.pop(num)
                cnum = cnum + 1
                contour_hulls.append(save_room)
                if debug:
                    visualization([save_room],  os.path.join(absolute_path, "result/debug/first_{}_{}_{}.png".
                                                                format(inter_num, direction[d], cnum)))
                    visualization([new_contour], os.path.join(absolute_path, "result/debug/contour_{}_{}_{}.png".
                                                              format(inter_num, direction[d], cnum)))
            elif inter_num == 0:
                if check_point_hull_inter(point_hull, contour_coord, 1):
                    num = num + 1
                    continue
                # 4. point is on the contour to split the room contour
                # print("now process the points which are on the contour")
                remove_point, remove_point_index, inter_points, \
                room_index, contour_index, center_distances, check_num = [], [], [], [], [], [], 0
                # judge the spilt point
                for c in range(len(current_room)):
                    distance = pow((current_room[c][0] - 127.5),2) +\
                               pow((current_room[c][1] - 127.5),2)
                    center_distances.append(distance)
                # choose the distance near to center point as the added points
                point_index = np.argsort(np.array(center_distances))
                spilt_point, save_d = current_room[point_index[0]], 0
                di = judge_direction(current_room, spilt_point)
                pointSet.append([spilt_point, di])
                # check again if the inter point is on contour
                for k in range(len(current_room)):
                    if cv.pointPolygonTest(contour_coord, (current_room[k][0],
                                                           current_room[k][1]), False) == 0:  # in the contour
                        check_num = check_num + 1
                if check_num >= 4 and cv.pointPolygonTest(contour_coord, (spilt_point[0], spilt_point[1]), False) != -1:
                # inter point have 0 should test the point is in contour or not
                    room_contour, rest_contour = random_room_generate(spilt_point, di, contour_coord)
                    room_point_hull, room_bounding_box = get_contour_layout(room_contour, first=False, contour=False)
                new_contour, _ = get_contour_layout(rest_contour, first=False, contour=True)
                point_hull.pop(num)
                cnum = cnum + 1
                contour_hulls.append(current_room)
                if debug:
                    visualization([current_room], os.path.join(absolute_path, "result/debug/first_{}_{}_{}.png".
                                                            format(inter_num, direction[d], cnum)))
                    visualization([new_contour], os.path.join(absolute_path, "result/debug/contour_{}_{}_{}.png".
                                                              format(inter_num, direction[d], cnum)))
            elif inter_num >= 3:
                # if have not 1 inter point
                if check_point_hull_inter(point_hull, contour_coord, 1) or \
                        check_point_hull_inter(point_hull, contour_coord, 0) or \
                            check_point_hull_inter(point_hull, contour_coord, 2):
                    num = num + 1
                    continue
                print("inter_num > 3!")
                # 3. approximate the contour deliver by 2 points using one point
                spilt_ious, biggest_iou, save_p, save_d, save_contour, save_room = [], 0, 0, 0, None, None
                for p in range(inter_num):
                    spilt_point = spilt_points[p]
                    for d in range(len(direction)):
                        room_contour, rest_contour = random_room_generate(spilt_point, direction[d], contour_coord)
                        room_point_hull, room_bounding_box = get_contour_layout(room_contour, first=False, contour=False)
                        new_contour, _ = get_contour_layout(rest_contour, first=False, contour=True)
                        [ x, y, w, h] = cv.boundingRect(current_room)
                        [ x1, y1, w1, h1] = room_bounding_box[0]
                        iou = bbox_iou(torch.tensor([[x, y, x + w, y + h]], dtype=float) / 255,
                                       torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=float) / 255)[0]
                        # choose the split point and direction which have the biggest iou
                        if  iou > biggest_iou:
                            biggest_iou = iou
                            save_d, save_p, save_contour, save_room = d, p, new_contour, room_contour
                new_contour = save_contour
                pointSet.append([spilt_points[save_p], direction[save_d]])
                point_hull.pop(num)
                cnum = cnum + 1
                contour_hulls.append(save_room)
                if debug:
                    visualization([save_room],  os.path.join(absolute_path, "result/debug/first_{}_{}_{}.png".
                                                                format(inter_num, direction[d], cnum)))
                    visualization([new_contour], os.path.join(absolute_path, "result/debug/contour_{}_{}_{}.png".
                                                              format(inter_num, direction[d], cnum)))
            else:
                num = num + 1
            # if len(point_hull) == 1 and inter_num >= 3:
            #     contour_hulls.append(point_hull[0])
            #     point_hull.pop(0)
        contour_hulls.append(contour_coord)
        if len(first_room_point_hull) != len(contour_hulls) or len(first_room_point_hull) != len(pointSet)+1:
            print("The origin contour hull num is {}, Processed to get {}, PointSet num {}".format(len(first_room_point_hull), len(contour_hulls), len(pointSet)))
        print("processs the layout {} OK!".format(index))
        # /************ save the process result *****************/
        # save_train_data_path = os.path.join(absolute_path, "GroundLayer/layout_gt_points/layers_gt_points_{}.npy".format(index))
        # save_hull_data_path = os.path.join(absolute_path, "GroundLayer/layout_gt_points/layers_gt_hulls_{}.npy".format(index))
        # layer_pair_points = np.array(pointSet)
        # layer_pair_hulls = np.array(contour_hulls)
        # np.save(save_train_data_path, layer_pair_points)
        # np.save(save_hull_data_path, layer_pair_hulls)
        # print("save {} layout is OK!".format(index))
        # /************ save the final contour to see the result **************/
        # visualization(contour_hulls, os.path.join(absolute_path, "result/layout_gt_hulls_v2/approx_gt_hulls_{}.png".format(index)))
        # visualization([new_contour], os.path.join(absolute_path, "result/debug/first_final_contour.png"))
