'''
Description: According to the setting rule to valid the contour is good or not and 
             
FilePath: /LayoutGenerator_Lited/dataset/Preprocess.py
'''
import os
import sys
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '../..')))
sys.path[0] = dir_path
import numpy as np
import cv2 as cv
import torch
import argparse
from miscc.getContour import ConversionLayout
from miscc.config import cfg, cfg_from_file
from main import parse_args


class IndicatorConstraint(object):
    def __init__(self):
        pass

    @staticmethod
    def compare_score(score1, score2):
        """
        function: compare the two scores
        """
        if score1 != score2:
            if score2 > score1:
                return 0
            else:
                return 1
        else:
            return 2

    @staticmethod
    def calculate_distance(point1, point2):
        """
        function: calculate the distance between two points
        """
        [x1, y1] = np.array(point1, dtype=np.float)
        [x2, y2] = np.array(point2, dtype=np.float)
        distance = np.power((y2 - y1), 2) + np.power((x2 - x1), 2)
        distance = np.sqrt(distance)
        return distance

    @staticmethod
    def room_connection(point_hull1, point_hull2):
        """
        function: judge the point_hull2 point on point_hull1 or not
                  positive (inside), negative (outside), or zero (on an edge)
        """
        for r in range(len(point_hull2)):
            if cv.pointPolygonTest(np.array(point_hull1, dtype=np.float32), (point_hull2[r][0], point_hull2[r][1]),
                                   False) != -1:
                return True
        return False

    def calculate_cost(self, room_hull, contour):
        """
        function: according the contour to get the wall line to calculate cost
        """
        contour = np.array(contour)
        total_cost = 0.0
        line_set = self.get_line_from_room(room_hull)
        for m in range(len(line_set)):
            if not self.line_is_on_contour(line_set[m], np.array(contour)):
                cost = self.calculate_distance(line_set[m][0], line_set[m][1])
                total_cost = cost + total_cost
        return total_cost

    @staticmethod
    def get_line_from_room(room_hull):
        """
        function: get the line from room_hull
        """
        line_set = []
        for h in range(len(room_hull)):
            if h + 1 == len(room_hull):
                line_set.append([room_hull[h], room_hull[0]])
            else:
                line_set.append([room_hull[h], room_hull[h + 1]])
        return line_set

    @staticmethod
    def line_is_on_contour(line, point_hull):
        """
        function: judge the line on contour or not
        """
        line[0], line[1] = np.array(line[0]), np.array(line[1])
        valid_line = (line[0] - line[1]).astype(int)
        for p in range(len(point_hull)):
            start_point = point_hull[p]
            if p + 1 == len(point_hull):
                end_point = point_hull[0]
            else:
                end_point = point_hull[p + 1]
            contour_line = (end_point - start_point).astype(int)
            on_line = (start_point - line[0]).astype(int)
            # on line or on extension line; cv.pointPolygonTest: +1(inside); -1(outside); 0(on the edge)
            if np.cross(valid_line, contour_line) == 0 and np.cross(valid_line, on_line) == 0:
                if cv.pointPolygonTest(point_hull, (line[0][0], line[0][1]), False) == 0 \
                        and cv.pointPolygonTest(point_hull, (line[1][0], line[1][1]), False) == 0:
                    return True
        return False

    def lines_aspect_ratio_judge(self, room_hull, score):
        """
        function: calculate the line aspect ratio lines: aspect ratio > 1/3
        """
        def get_distance_from_room(hull):
            line_set = self.get_line_from_room(hull)
            hull_distances = []
            for s in range(len(line_set)):
                distance = self.calculate_distance(line_set[s][0], line_set[s][1])
                hull_distances.append(distance)
            return hull_distances
        distances = get_distance_from_room(room_hull)
        for d in range(len(distances)):
            for k in range(len(distances)):
                ratio = float(distances[d]) / float(distances[k])
                if  ratio <= 1. / 3. or ratio >= 3.:
                    score = score - 1
        # ***************** version 1
        # distances = get_distance_from_room(room_hull)
        # sorted_distances = sorted(distances)
        # max_distance, min_distance = sorted_distances[-1], sorted_distances[0]
        # ratio = float(min_distance) / float(max_distance)
        # p_max, p_min, score_ratio = 3., 1./3., 0
        # if ratio > p_max or ratio < p_min:
        #     score = score - 1
        return score

    @staticmethod
    def calculate_surface_area(polygon):
        # use the image to calculate the area
        im = np.zeros((256, 256))
        polygon_mask = cv.fillPoly(im, [np.array(polygon, dtype=np.int)], 255)
        area = np.sum(np.greater(polygon_mask, 0))
        return area

    def calculate_score(self, contour, layout1_hulls, layout2_hulls, constraint_rule, data_type=None):
        """
        build the constraint to control the two layout which better
        when the hard_rules, soft_rules, cost_rule all > 0 will return 1 otherwise 0
        """
        layout1_scores, layout2_scores, hard_rules, soft_rules, cost_rule = 0, 0, 0, 0, 0
        if data_type == "tensor":
            for i in range(len(layout1_hulls)):
                layout1_hulls[i] = torch.squeeze(layout1_hulls[i])
                layout2_hulls[i] = torch.squeeze(layout2_hulls[i])
        # /*****1.hard rule*****/
        layout_contour = np.array(contour, dtype=np.int32)
        total_area = float(self.calculate_surface_area(layout_contour[:, :2]))
        area_1, area_2, score1, score2 = [], [], 0, 0
        line_constraint, surface_constraint = 1, 2
        for e in range(len(layout1_hulls)):
            # a) lines: aspect ratio > 1/3 or ratio > 3
            if line_constraint in constraint_rule:
                score1 = self.lines_aspect_ratio_judge(layout1_hulls[e], score1)
                score2 = self.lines_aspect_ratio_judge(layout2_hulls[e], score2)
        if score1 > score2:
            layout1_scores = layout1_scores + 1
        elif score2 > score1:
            layout2_scores = layout2_scores + 1
        score1, score2 = 0, 0
        for e in range(len(layout1_hulls)):
            # b) surface: room_area/all_area > 1 / (2*room_num)
            area1 = float(self.calculate_surface_area(layout1_hulls[e]))
            area2 = float(self.calculate_surface_area(layout1_hulls[e]))
            area_1.append(area1)
            area_2.append(area2)
            if surface_constraint in constraint_rule:
                if area1 / total_area > 1. / (2 * len(layout1_hulls)):
                    score1 = score1 + 1
                if area2 / total_area > 1. / (2 * len(layout1_hulls)):
                    score2 = score2 + 1
        if score1 > score2:
            layout1_scores = layout1_scores + 1
        elif score2 > score1:
            layout2_scores = layout2_scores + 1
        if line_constraint in constraint_rule and surface_constraint in constraint_rule:
            compare_value = self.compare_score(layout1_scores, layout2_scores)
            if compare_value == 1:
                hard_rules= 1
            else:
                return 0
        else:
            hard_rules = 1
        # /*****2.soft rule*****/
        # a)Connection: living room must connect to other rooms
        # select the max area as the living room
        connection_constraint, layout1_scores, layout2_scores = 3, 0, 0
        layout1_living_index = area_1.index(max(area_1))
        layout2_living_index = area_2.index(max(area_2))
        score1, score2 = 0, 0
        if connection_constraint in constraint_rule:
            for l in range(len(layout1_hulls)):
                if self.room_connection(layout1_hulls[layout1_living_index], layout1_hulls[l]):
                    score1 = score1 + 1
                if self.room_connection(layout2_hulls[layout2_living_index], layout2_hulls[l]):
                    score2 = score2 + 1
        if score1 > score2:
            layout1_scores = layout1_scores + 1
        elif score2 > score1:
            layout2_scores = layout2_scores + 1
        # b)Neat Layout: Neat will be better than bump
        # default neat bounding box: 4
        neat_constraint, node_num1, node_num2 = 4, 0, 0
        if neat_constraint in constraint_rule:
            for k in range(len(layout1_hulls)):
                node_num1 = len(layout1_hulls[k]) + node_num1
                node_num2 = len(layout2_hulls[k]) + node_num2
        if node_num1 > node_num2:
            layout1_scores = layout1_scores - 1
        elif node_num1 < node_num2:
            layout2_scores = layout2_scores - 1
        if connection_constraint in constraint_rule and neat_constraint in constraint_rule:
            compare_value = self.compare_score(layout1_scores, layout2_scores)
            if compare_value == 1:
                soft_rules = 1
            else:
                return 0
        else:
            soft_rules = 1
        # /*****3.wall cost*****/
        # calculate the two cost
        total_cost1, total_cost2, layout1_scores, layout2_scores = 0, 0, 0, 0
        wall_constraint = 5
        for r in range(len(layout1_hulls)):
            cost1 = self.calculate_cost(layout1_hulls[r], layout_contour[:, :2])
            total_cost1 = total_cost1 + cost1
        for t in range(len(layout2_hulls)):
            cost2 = self.calculate_cost(layout2_hulls[t], layout_contour[:, :2])
            total_cost2 = total_cost2 + cost2
        if wall_constraint in constraint_rule:
            compare_value = self.compare_score(total_cost2, total_cost1)
            if compare_value == 1:
                cost_rule = 1
            else:
                return 0
        else:
            cost_rule = 1
        if hard_rules == 1 and soft_rules == 1 and cost_rule ==1:
            return 1
        else:
            return 0

def indicator_data_save():
    from dataset.datasets import LayoutDataset
    # config cfg file
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    indicator_path = "/home/zhoujiaqiu/Code/GAN/LayoutGenerator/dataset/data" \
                     "/indicator_train_data/constraint_inter"
    dataset = LayoutDataset(cfg.DATA_DIR, cfg.INDICATOR_DIR, train_set=True)

    convert_layout = ConversionLayout()
    constraint_rules = [1, 2, 3, 4, 5]
    for n in range(len(dataset.pair_room_datas)):
        pair_data, init_contour, indicator_hull = dataset.pair_room_datas[n], dataset.pair_init_contours[n], \
                                                  dataset.indicator_hull[n]
        indicator_values = np.zeros((len(pair_data), 1))
        for i in range(len(pair_data)):
            # data: [1, 2500, 2, 4, 3] init_contour: [2500, 2, n, 3]
            # indicator_hull: [1, 2500, 2, 5, n, 2]
            layout1_room, layout2_room = pair_data[i][0], pair_data[i][1]
            layout1_init_contour, layout2_init_contour = init_contour[i][0], init_contour[i][1]
            layout1_hull, layout2_hull = indicator_hull[i][0], indicator_hull[i][1]
            indicator_value = indicator.calculate_score(layout1_init_contour, layout1_hull, layout2_hull,
                                                        constraint_rules)
            indicator_values[i] = indicator_value
            print("on {} layouts, the {} indicator_value save finished!".format(n, i))

            # /****** visualization ******/
            # from trainer.trainer_evaluator import visualization
            # save_path = "/home/zhoujiaqiu/Code/GAN/ours_layout_generator/dataset/visualization/{}.png".format(n)
            # visual_hull = [layout1_hull, layout2_hull]
            # visualization(visual_hull, [indicator_value, indicator_value], save_path)
        save_indicator_path = os.path.join(indicator_path, "layout_{}_indicator_value.npy".format(n))
        np.save(save_indicator_path, indicator_values)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate new data')
    parser.add_argument('--path', dest='path', type=str, help='set gpu id')
    control_args = parser.parse_args()
    return control_args

if __name__ == "__main__":
    # config cfg file
    args = parse_args()
    path = args.path
    print("save path: ", path)
    data_dir = '/home/zhoujiaqiu/Code/GAN/MultiLayerDataset'
    indicator = IndicatorConstraint()
    for i in range(1, 401):
        # if i <= 24:
        #     continue
        saved_layers_data, saved_pairs_data, saved_contours, saved_indicator_values = [], [], [], []
        layers_data_path = os.path.join(data_dir, "EvaluatorData/20_pair_train_data/layers_pair_points_%d.npy" % i)
        contour_data_path = os.path.join(data_dir, "EvaluatorData/20_pair_train_data/layers_hull_points_%d.npy" % i)
        layers_data = np.load(layers_data_path, encoding = 'latin1')
        contour = np.load(contour_data_path, encoding = 'latin1')
        # 1. find the layer data pairs
        init_contour, rooms_contour = np.array(list(contour[0][0])), contour[:, 2]
        saved_contours.append(init_contour)
        if path == "w_all":
            constraint_rules = [1, 2, 3, 4, 5]
        elif path == "wo_Cost":
            constraint_rules = [1, 2, 3, 4]
        elif path == "wo_HardRule":
            constraint_rules = [3, 4, 5]
        elif path == "wo_SoftRule":
            constraint_rules = [1, 2, 5]
        for p in range(len(layers_data)):
            num = 0
            for pn in range(len(layers_data)):
                num = num + 1
                if indicator.calculate_score(init_contour, rooms_contour[p], rooms_contour[pn], constraint_rules):
                    saved_pairs_data.append([layers_data[p], layers_data[pn]])
                    saved_layers_data.append([rooms_contour[p], rooms_contour[pn]])
                    saved_indicator_values.append(1)
                    saved_pairs_data.append([layers_data[pn], layers_data[p]])
                    saved_layers_data.append([rooms_contour[pn], rooms_contour[p]])
                    saved_indicator_values.append(0)
                    # if num > 5:
                    #     break
            # if len(saved_layers_data) > 30:
            #     break
        # 2.compare the random generate with the gt
        # gt_point_data_path = os.path.join(data_dir, "GroundLayer/layout_gt_points/layers_gt_points_%d.npy" % i)
        # gt_hull_data_path = os.path.join(data_dir, "GroundLayer/layout_gt_points/layers_gt_hulls_%d.npy" % i)
        # gt_point = np.load(gt_point_data_path)
        # gt_hull = np.load(gt_hull_data_path)
        # num = 0
        # for l in range(len(layers_data)):
        #     if indicator.calculate_score(init_contour, gt_hull, rooms_contour[l], constraint_rules):
        #         saved_pairs_data.append([gt_point, layers_data[l]])
        #         saved_layers_data.append([gt_hull, rooms_contour[l]])
        #         saved_indicator_values.append(1)
        #         saved_pairs_data.append([layers_data[l], gt_point])
        #         saved_layers_data.append([rooms_contour[l], gt_hull])
        #         saved_indicator_values.append(0)
        
        # /****** visualization ******/
        # from trainer.trainer_evaluator import visualization
        # save_path = "/home/zhoujiaqiu/Code/GAN/LayoutGenerator/dataset/visualization/{}.png".format(i)
        # visual_hull = [gt_hull, rooms_contour[l]]
        # visualization(visual_hull, [0, 1], save_path, data_type="None")

        # 3.random some noise to compare the data
        noises_datas = []
        add_noise_num = 1
        for n in range(add_noise_num):
            seed = i*400 + n * len(layers_data)
            np.random.seed(seed)
            noises = []
            # print("set seed: ", seed)
            # if n < 5: # repeat the first data
                # random the value to 0 ~ 1
            noise = np.random.rand(1, 3)
            noise[:, :2] = np.round(noise[:, :2] * 255).astype(np.int32)
            noise[:, 2] = np.round(noise[:, 2] * 3).astype(np.int32)
            noise = noise.repeat(layers_data[n].shape[0], axis=0)
            for o in range(len(noise)):
                noises.append([noise[o, :2], noise[o, 2]])
                # print("noise: ", noise)
            # else: # no repeat random the noise
            #     noise = np.random.rand(layers_data[n].shape[0], 3)
            #     noise[:, :2] = np.round(noise[:, :2] * 255).astype(np.int32)
            #     noise[:, 2] = np.round(noise[:, 2] * 3).astype(np.int32)
            #     for o in range(len(noise)):
            #         noises.append([noise[o, :2], noise[o, 2]])
            noises = np.array(noises)
            noises_datas.append(noises)
        #     # print(noises)
        #     saved_pairs_data.append([layers_data[n], noises])
        #     saved_layers_data.append([0, 0])
        #     saved_indicator_values.append(1)
        for n in range(add_noise_num):
            saved_pairs_data.append([noises_datas[n], layers_data[n]])
            saved_layers_data.append([0, 0])
            saved_indicator_values.append(0)
        print("compare {} is OK! saved num: {} totally! {} layers can be train!".format(i, len(saved_pairs_data), np.sum(saved_indicator_values)))
        # /************* save data for train ******************/
        # save_train_data_path = os.path.join(data_dir, "EvaluatorData/layers_hull_data/layers_pair_points_{}.npy".format(i))
        # save_hull_data_path = os.path.join(data_dir, "EvaluatorData/layers_hull_data/layers_hull_points_{}.npy".format(i))
        # save_init_contour_data_path = os.path.join(data_dir, "EvaluatorData/layers_hull_data/layers_init_contour_{}.npy".format(i))
        # saved_indicator_values_path = os.path.join(data_dir, "EvaluatorData/layers_hull_data/layers_indicator_values_{}.npy".format(i))

        # layer_pair_points, layer_pair_hulls, layer_init_contour, layers_indicator_values = \
        #     np.array(saved_pairs_data), np.array(saved_layers_data), np.array(saved_contours), np.array(saved_indicator_values)
        # np.save(save_train_data_path, layer_pair_points)
        # np.save(save_hull_data_path, layer_pair_hulls)
        # np.save(save_init_contour_data_path, layer_init_contour)
        # np.save(saved_indicator_values_path, layers_indicator_values)
