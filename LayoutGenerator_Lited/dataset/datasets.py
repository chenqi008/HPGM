import os
import sys
import torch
from torchvision import transforms
import numpy as np
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '../..')))
sys.path.append(dir_path)
from miscc.config import cfg

from dataset.Preprocess import IndicatorConstraint

import torch.utils.data as data


class LayoutDataset(data.Dataset):
    def __init__(self, data_dir, indicator_data_dir, dataset, train_set):
        self.transform = transforms.Compose(
            [])  # you can add to the list all the transformations you need.
        self.data_dir = data_dir
        self.gt_data_dir = cfg.GT_DATA_DIR
        self.indicator_data_dir = indicator_data_dir
        self.train_set = train_set
        if train_set:
            id_path = os.path.join(self.data_dir, "train_id.txt")
            self.id = np.loadtxt(id_path)
        else:
            id_path = os.path.join(self.data_dir, "test_id.txt")
            self.id = np.loadtxt(id_path)
        # eval
        self.pair_room_datas, self.pair_init_contours, self.indicator_hull, \
        self.pair_indicator_values, self.contour_types, self.indicator_values_data = [], [], [], [], [], []
        # gen
        self.room_datas, self.init_contours, self.gt_hull_datas, self.gt_point_datas = [], [], [], []
        self.constraint_rules = cfg.TRAIN.CONSTRAINT_RULES
        self.indicator = IndicatorConstraint()
        self.direction_dict = {0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]}
        if cfg.DATASET == 'gen':
            for i in range(len(self.id)):
                # for test: 200_pair_test_data
                # for train: pair_train_data_gt pair_hull_data_gt
                if cfg.TRAIN.FLAG:
                    layers_data_path = os.path.join(self.data_dir, "pair_train_data_gt/layers_pair_points_%d.npy" % int(self.id[i]))
                    contour_data_path = os.path.join(self.data_dir, "pair_hull_data_gt/layers_hull_points_%d.npy" % int(self.id[i]))
                else:
                    layers_data_path = os.path.join(self.data_dir, "200_pair_test_data/layers_pair_points_%d.npy" % int(self.id[i]))
                    contour_data_path = os.path.join(self.data_dir, "200_pair_test_data/layers_hull_points_%d.npy" % int(self.id[i]))
                gt_point_data_path = os.path.join(self.gt_data_dir, "layout_gt_points/layers_gt_points_%d.npy" % int(self.id[i]))
                gt_hull_data_path = os.path.join(self.gt_data_dir, "layout_gt_points/layers_gt_hulls_%d.npy" % int(self.id[i]))
                self.layers_data = np.load(layers_data_path)
                self.contour = np.load(contour_data_path)
                self.gt_hulls = np.load(gt_hull_data_path)
                gt_points = np.load(gt_point_data_path)
                room_data, init_contour = self.data_process()
                gt_points = self.data_process_gt(gt_points)
                gt_points = gt_points.repeat(room_data.size(0), 1, 1)
                self.gt_point_datas.append(gt_points)
                self.room_datas.append(room_data)
                self.gt_hull_datas.append(self.gt_hulls)
                self.init_contours.append(init_contour)
            print("room data shape:", len(room_data))
        elif cfg.DATASET == 'eval':
            for i in range(len(self.id)): 
                print("i{} id{} 's data is processing".format(i, self.id[i]))
                # load and process the data 
                layers_data_path = os.path.join(self.data_dir, "EvaluatorData/layers_hull_data/layers_pair_points_%d.npy" % int(self.id[i]))
                contour_data_path = os.path.join(self.data_dir, "EvaluatorData/layers_hull_data/layers_hull_points_%d.npy" % int(self.id[i]))
                init_contour_data_path = os.path.join(self.data_dir, "EvaluatorData/layers_hull_data/layers_init_contour_%d.npy" % int(self.id[i]))
                indicator_data_path = os.path.join(self.data_dir, "EvaluatorData/layers_hull_data/layers_indicator_values_%d.npy" % int(self.id[i]))
                contour_type_path = os.path.join(self.data_dir, "OriginData/layout_gt_coord_type/layout%d_boxes_type.txt" % int(self.id[i]))
                self.layers_data = np.load(layers_data_path)
                self.contour_type = np.loadtxt(contour_type_path)
                self.contour = np.load(contour_data_path)
                self.init_contour = np.load(init_contour_data_path)
                self.indicator_values = np.load(indicator_data_path)
                room_data1_tensors, room_data2_tensors, init_contours_tensors,\
                    pair_hull_tensors, data_type, indicator_value_tensors = self.data_process_refine()
                self.pair_init_contours.append(init_contours_tensors)
                self.indicator_hull.append(pair_hull_tensors)
                self.pair_room_datas.append([room_data1_tensors, room_data2_tensors])
                self.contour_types.append(data_type)
                self.indicator_values_data.append(indicator_value_tensors)
                print("room data shape:", len(room_data1_tensors))
        self.iterator = self.prepare_data
    
    def tran_data_tensor(self, room_data):
        # tran the room_point_data to tensors 
        room_data_tensors = None
        for n in range(len(room_data)):
            point, direction = room_data[n]
            point = point.astype(np.float64)/256
            # direction: 0:left up 1: right up 2: left bottom 3: right bottom
            # direction = float(direction)/4
            direction = self.direction_dict[direction]
            room_feature = torch.from_numpy(np.hstack((point, direction))).type(torch.float32)
            room_feature = torch.unsqueeze(room_feature, 0)
            if n == 0:
                room_data_tensors = room_feature
            else:
                room_data_tensors = torch.cat((room_data_tensors, room_feature), dim=0)
        return room_data_tensors
    
    def data_process_refine(self):
        rooms_data1_tensors, rooms_data2_tensors, init_contours_tensors, pair_hull_tensors, data_types = None, None, None, [], []
        for i in range(len(self.layers_data)):
            [room_data1, room_data2], [room_contour1, room_contour2], init_contour = self.layers_data[i], self.contour[i], self.init_contour[0]
            # 1. process init contour
            init_contour = np.array(list(init_contour))
            init_contour = torch.from_numpy(np.array(init_contour)).type(torch.float32)/256
            zeros = torch.zeros((len(init_contour), 1)).type(torch.float32)
            init_contour = torch.cat((init_contour, zeros), dim=1)
            init_contour = torch.cat((init_contour, zeros), dim=1)
            train_init_contours = torch.unsqueeze(init_contour, 0)
            # 2. process the room data
            room_data1_tensors = self.tran_data_tensor(room_data1)
            room_data2_tensors = self.tran_data_tensor(room_data2)
            train_room_data1_tensors = torch.unsqueeze(room_data1_tensors, 0)
            train_room_data2_tensors = torch.unsqueeze(room_data2_tensors, 0)
            if i == 0:
                rooms_data1_tensors = train_room_data1_tensors
                rooms_data2_tensors = train_room_data2_tensors
                init_contours_tensors = train_init_contours
            else:
                rooms_data1_tensors = torch.cat((rooms_data1_tensors, train_room_data1_tensors), dim=0)
                rooms_data2_tensors = torch.cat((rooms_data2_tensors, train_room_data2_tensors), dim=0)
                init_contours_tensors = torch.cat((init_contours_tensors, train_init_contours), dim=0)
            pair_hull_tensors.append([room_contour1, room_contour2])
        # process type for visualization
        data_types.append(self.contour_type[0])
        for i in range(len(self.contour_type)-1):
            if self.contour_type[i] == self.contour_type[i+1]:
                continue
            else:
                data_types.append(self.contour_type[i+1])
        # make indicator_values
        indicator_values = torch.from_numpy(np.array(self.indicator_values))
        return rooms_data1_tensors, rooms_data2_tensors, init_contours_tensors, pair_hull_tensors, data_types, indicator_values
        
    def data_process_gt(self, gt_points):
        gt_data_tensor = None
        for n in range(len(gt_points)):
            point, direction = gt_points[n]
            point = point.astype(np.float64)/256
            # direction: 0:left up 1: right up 2: left bottom 3: right bottom
            # direction = float(direction)/4
            direction = self.direction_dict[direction]
            room_feature = torch.from_numpy(np.hstack((point, direction))).type(torch.float32)
            room_feature = torch.unsqueeze(room_feature, 0)
            if n == 0:
                gt_data_tensor= room_feature
            else:
                gt_data_tensor = torch.cat((gt_data_tensor, room_feature), dim=0)
        return gt_data_tensor

    def data_process(self):
        rooms_data_tensors,init_contours_tensors, rooms_contour = None, None, []
        for i in range(len(self.layers_data)):
            if i == cfg.TRAIN.SAMPLE_NUM:
                break
            room_data, [init_contour, _, room_contour]= self.layers_data[i], self.contour[i]
            init_contour = np.array(list(init_contour))
            # 1. process init contour
            init_contour = torch.from_numpy(np.array(init_contour)).type(torch.float32)/256
            zeros = torch.zeros((len(init_contour), 1)).type(torch.float32)
            init_contour = torch.cat((init_contour, zeros), dim=1)
            init_contour = torch.cat((init_contour, zeros), dim=1)
            # 2. process the room data
            room_data_tensors = None
            for n in range(len(room_data)):
                point, direction = room_data[n]
                point = point.astype(np.float64)/256
                # direction: 0:left up 1: right up 2: left bottom 3: right bottom
                direction = self.direction_dict[direction]
                room_feature = torch.from_numpy(np.hstack((point, direction))).type(torch.float32)
                room_feature = torch.unsqueeze(room_feature, 0)
                if n == 0:
                    room_data_tensors = room_feature
                else:
                    room_data_tensors = torch.cat((room_data_tensors, room_feature), dim=0)
            train_room_data_tensors = torch.unsqueeze(room_data_tensors, 0)
            train_init_contours = torch.unsqueeze(init_contour, 0)
            rooms_contour.append(room_contour)
            if i == 0:
                rooms_data_tensors = train_room_data_tensors
                init_contours_tensors = train_init_contours
            else:
                rooms_data_tensors = torch.cat((rooms_data_tensors, train_room_data_tensors), dim=0)
                init_contours_tensors = torch.cat((init_contours_tensors, train_init_contours), dim=0)
        return rooms_data_tensors, init_contours_tensors
    

    def prepare_data(self, index):
        if cfg.DATASET == 'gen':
            room_data = self.room_datas[index]
            init_contour = self.init_contours[index]
            first_room_data = self.gt_point_datas[index]
            gt_room_data = self.gt_hull_datas[index]
            return room_data, init_contour, first_room_data, gt_room_data
        elif cfg.DATASET == 'eval':
            pair_data = self.pair_room_datas[index]
            init_contour = self.pair_init_contours[index]
            indicator_hull = self.indicator_hull[index]
            contour_type = self.contour_types[index]
            indicator_values = self.indicator_values_data[index]
        return pair_data, init_contour, \
               indicator_hull, contour_type, indicator_values

    def __getitem__(self, item):
        return self.iterator(item)

    def __len__(self):
        return len(self.id)

    def collate_fn(self, batch):
        # eval
        if cfg.DATASET == 'eval':
            pair_data = list()
            init_contour = list()
            indicator_hull = list()
            contour_type = list()
            indicator_value = list()
            # batch
            for b in batch:
                pair_data.append(b[0])
                init_contour.append(b[1])
                indicator_hull.append(b[2])
                contour_type.append(b[3])
                indicator_value.append(b[4])
            return pair_data, init_contour, indicator_hull, contour_type, indicator_value
        elif cfg.DATASET == 'gen':
            room_data = list()
            init_contour =list()
            first_room_data = list()
            gt_room_data = list()
            for b in batch:
                room_data.append(b[0])
                init_contour.append(b[1])
                first_room_data.append(b[2])
                gt_room_data.append(b[3])
            return room_data, init_contour, first_room_data, gt_room_data

if __name__ == '__main__':
    # /******** test the data ********/
    points = np.load("/home/zhoujiaqiu/Code/GAN/ours_bbox_gcn_parallel/multiLayerLayout/multiLayer_data"
                     "/pair_train_data/layers_pair_points_1.npy")
    print(points)
