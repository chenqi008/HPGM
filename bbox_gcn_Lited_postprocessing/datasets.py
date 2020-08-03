from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import transforms
import os.path
import random
import numpy as np
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import scipy.sparse as sp


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, transform=None, normalize_img=None):
    if cfg.IMAGE_CHANNEL == 3:
        img = Image.open(img_path).convert('RGB')
    elif cfg.IMAGE_CHANNEL == 1:
        img = Image.open(img_path).convert('L')
    if transform is not None:
        img = transform(img)
    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        re_img = transforms.Resize(imsize[i])(img)
        ret.append(normalize_img(re_img))
    return ret


def get_imgs_test(img_path, imsize, transform=None, normalize_img=None):
    if cfg.IMAGE_CHANNEL == 3:
        img = Image.open(img_path).convert('RGB')
    elif cfg.IMAGE_CHANNEL == 1:
        img = Image.open(img_path).convert('L')
    width, height = img.size

    if transform is not None:
        img = transform(img)

    ret = []
    re_img = transforms.Resize(imsize[-1])(img)
    ret.append(normalize_img(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, base_size=64, transform=None, target_transform=None, train_set=True):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.target_transform = target_transform
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2   # 64 128 256

        self.data = []
        self.data_dir = data_dir
        self.room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen', 
                             'washroom', 'study', 'closet', 'storage', 'balcony']
        self.position_classes = ['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE']
        self.filenames = self.load_filenames(data_dir)
        self.train_set = train_set
        self.furniture = cfg.FURNITURE  # False

        if cfg.TRAIN.FLAG and self.train_set:
            # load train id of the dataset
            filepath = os.path.join(data_dir, 'train_id.pickle')
            with open(filepath, 'rb') as f:
                train_id = pickle.load(f)
            train_id = sorted(train_id)
            # build training filenames
            self.filenames_train = []
            for idx in train_id:
                self.filenames_train.append(self.filenames[idx])
            self.filenames_train = sorted(self.filenames_train)

            # load graphs(adjacency matrix of a room to every room in a text)
            self.graphs = self.load_graphs(data_dir, self.filenames_train)

            # load bounding boxes(1:room box position 2:room type)
            self.bboxes = self.load_bboxes(data_dir, self.filenames_train)

            # load vectors of objects(1:room_class+size+position class 2:room type)
            self.objs_vectors = self.load_objs_vectors(data_dir, self.filenames_train)
            # build iterator
            self.iterator = self.prepair_training_pairs
        else:
            # load train id of the dataset
            filepath = os.path.join(data_dir, 'test_id.pickle')
            with open(filepath, 'rb') as f:
                test_id = pickle.load(f)
            test_id = sorted(test_id)
            # build test filenames
            self.filenames_test = []
            for idx in test_id:
                self.filenames_test.append(self.filenames[idx])
            self.filenames_test = sorted(self.filenames_test)
            # load graphs
            self.graphs = self.load_graphs(data_dir, self.filenames_test)
            # load bounding boxes
            self.bboxes = self.load_bboxes(data_dir, self.filenames_test)
            # load vectors of objects
            self.objs_vectors = self.load_objs_vectors(data_dir, self.filenames_test)
            # build iterator
            self.iterator = self.prepair_test_pairs

    def load_filenames(self, data_dir):
        filenames = []
        current_dir = os.path.join(data_dir, 'semantic-expression')
        for file in os.listdir(current_dir):
            filenames.append(os.path.splitext(file)[0])
        return sorted(filenames)

    def load_graphs(self, data_dir, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        graphs = []
        for filename in filenames:
            path = os.path.join(current_dir, '{}.txt'.format(filename))
            # get the adjacent of the rooms
            graph = self.build_graph(path)
            # probabilistic the adjacency of a room to every room
            graph = self.normalize_graph(graph)
            graphs.append(graph)
        return graphs

    # build the node and edge according each text description
    # path: e.g., path = './semantic-expression/0.txt'
    def build_graph(self, path):
        with open(path, 'rb') as f:
            desc = f.read()
            desc = eval(desc)
            desc_rooms = desc['rooms']
            desc_links = desc['links']
            # count the number of nodes (rooms)
            count_nodes = 0
            for room in desc_rooms.keys():
                num_room = desc_rooms[room]['room num']
                count_nodes += num_room

            adj = np.zeros((count_nodes, count_nodes))

            # handle feature
            counter = 0  # counter of room number
            roomnames = []  # the name of each room, e.g., bedroom1
            # build all room in roomnames
            for room in desc_rooms.keys():
                # the position in the room classes
                idx = self.room_classes.index(room)
                num_room = desc_rooms[room]['room num']
                for i in range(num_room):
                    # feature[counter][idx] = 1.0
                    roomnames.append('{}{}'.format(room, i+1))
                    counter += 1
            # handle adjacent
            for desc_link in desc_links:
                # decide the position in adj
                link = desc_link['room pair']
                x = roomnames.index(link[0])
                y = roomnames.index(link[1])
                adj[x][y] = 1.0
        return adj

    # normalize the input graph and output with the same size of input one
    def normalize_graph(self, graph):
        # feature, adj = graph[0], graph[1]
        adj = graph

        # convert to space format
        adj = sp.coo_matrix(adj, dtype = np.float32)

        # build symmetric adjacency matrix
        # the item "adj.multiply(adj.T > adj)" may be useless
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # normalization
        # diagonal line will be 1
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj

    # normalization for graph (feature and adj)
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    # Convert a scipy sparse matrix to a torch sparse tensor.
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # loador of all bounding boxes
    def load_bboxes(self, data_dir, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        bboxes = []
        for filename in filenames:
            path = os.path.join(current_dir, '{}.txt'.format(filename))
            bbox = self.build_bboxes(path)
            bboxes.append(bbox)
        return bboxes

    # build a bounding box
    def build_bboxes(self, path):
        WW = 512
        HH = 512
        with open(path, 'rb') as f:
            desc = f.read()
            desc = eval(desc)

            desc_rooms = desc['rooms']

            O = 0  # counter of room number
            for room in desc_rooms.keys():
                num_room = desc_rooms[room]['room num']
                O += num_room

            objs = torch.LongTensor(O).fill_(-1)
            boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)

            # objs = torch.LongTensor(cfg.MAX_NODES).fill_(-1)
            # boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(cfg.MAX_NODES, 1)

            # handle bounding boxes
            counter = 0
            for room in desc_rooms.keys():
                # the position in the room classes
                idx = self.room_classes.index(room)
                num_room = desc_rooms[room]['room num']
                for i in range(num_room):
                    x0 = desc_rooms[room]['boundingbox'][i]['min']['x'] / WW
                    y0 = desc_rooms[room]['boundingbox'][i]['min']['y'] / HH
                    x1 = desc_rooms[room]['boundingbox'][i]['max']['x'] / WW
                    y1 = desc_rooms[room]['boundingbox'][i]['max']['y'] / HH
                    # boxes and objs (labels)
                    boxes[counter] = torch.FloatTensor([x0, y0, x1, y1])
                    objs[counter] = idx
                    counter += 1
        return boxes, objs

    # load input vectors of objects as format (type, size, position)
    def load_objs_vectors(self, data_dir, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        objs_vectors = []
        for filename in filenames:
            path = os.path.join(current_dir, '{}.txt'.format(filename))
            objs_vector = self.build_objs_vector(path)
            objs_vectors.append(objs_vector)
        return objs_vectors

    # build the vectors of objects in an image
    def build_objs_vector(self, path):
        with open(path, 'rb') as f:
            desc = f.read()
            desc = eval(desc)

            desc_rooms = desc['rooms']

            O = 0  # counter of room number
            for room in desc_rooms.keys():
                num_room = desc_rooms[room]['room num']
                O += num_room

            # dimension of each objects (row)
            if cfg.TRAIN.USE_SIZE_AS_INPUT:
                D = len(self.room_classes) + 1 + len(self.position_classes)
            else:
                D = len(self.room_classes) + len(self.position_classes)

            objs = torch.LongTensor(O).fill_(-1)
            objs_vector = torch.FloatTensor([[0.0]]).repeat(O, D)

            # handle vector of objects (type, size, position)
            counter = 0  # counter of room number
            for room in desc_rooms.keys():
                # the position in the room classes
                idx_type = self.room_classes.index(room)
                positions = desc_rooms[room]['position']
                sizes = desc_rooms[room]['size']
                num_room = desc_rooms[room]['room num']
                for i in range(num_room):
                    # type in vector
                    objs_vector[counter][idx_type] = 1.0
                    if cfg.TRAIN.USE_SIZE_AS_INPUT:
                        # position in vector
                        position = positions[i]
                        idx_position = self.position_classes.index(position)
                        objs_vector[counter][len(self.room_classes)+1+idx_position] = 1.0
                        # size in vector
                        objs_vector[counter][len(self.room_classes)] = sizes[i]
                    else:
                        # position in vector
                        position = positions[i]
                        idx_position = self.position_classes.index(position)
                        objs_vector[counter][len(self.room_classes)+idx_position] = 1.0

                    # record type
                    objs[counter] = idx_type
                    counter += 1
        return objs_vector, objs

    def prepair_training_pairs(self, index):
        # key = self.filenames[index]
        key = self.filenames_train[index]

        data_dir = self.data_dir
        graph = self.graphs[index]
        bbox = self.bboxes[index]
        objs_vector = self.objs_vectors[index]

        # load label images
        if self.furniture == True:
            label_img_name = os.path.join(data_dir, 'label', '{}.png'.format(key))
        else:
            label_img_name = os.path.join(data_dir, 'label_withoutFA_rearrange', '{}.png'.format(key))
        label_imgs = get_imgs(label_img_name, self.imsize, self.transform, normalize_img=self.norm)

        # load mask images
        mask_img_name = os.path.join(data_dir, 'mask', '{}.png'.format(key))
        mask_imgs = get_imgs(mask_img_name, self.imsize, self.transform, normalize_img=self.norm)

        # get wrong images
        wrong_ix = random.randint(0, len(self.filenames_train) - 1)
        while index == wrong_ix:
            wrong_ix = random.randint(0, len(self.filenames_train) - 1)
        wrong_key = self.filenames_train[wrong_ix]

        # load label images
        if self.furniture == True:
            wrong_label_img_name = os.path.join(data_dir, 'label', '{}.png'.format(wrong_key))
        else:
            wrong_label_img_name = os.path.join(data_dir, 'label_withoutFA_rearrange', '{}.png'.format(wrong_key))
        wrong_label_imgs = get_imgs(wrong_label_img_name, self.imsize, self.transform, normalize_img=self.norm)

        # load mask images
        wrong_mask_img_name = os.path.join(data_dir, 'mask', '{}.png'.format(wrong_key))
        wrong_mask_imgs = get_imgs(wrong_mask_img_name, self.imsize, self.transform, normalize_img=self.norm)
        return label_imgs, mask_imgs, wrong_label_imgs, wrong_mask_imgs, graph, bbox, objs_vector, key

    def prepair_test_pairs(self, index):
        # key = self.filenames[index]
        key = self.filenames_test[index]
        data_dir = self.data_dir
        graph = self.graphs[index]
        bbox = self.bboxes[index]
        objs_vector = self.objs_vectors[index]
        # load label images
        if self.furniture == True:
            label_img_name = os.path.join(data_dir, 'label', '{}.png'.format(key))
        else:
            label_img_name = os.path.join(data_dir, 'label_withoutFA_rearrange', '{}.png'.format(key))
        label_imgs = get_imgs_test(label_img_name, self.imsize, self.transform, normalize_img=self.norm)

        # load mask images
        mask_img_name = os.path.join(data_dir, 'mask', '{}.png'.format(key))
        mask_imgs = get_imgs_test(mask_img_name, self.imsize, self.transform, normalize_img=self.norm)

        # return imgs, wrong_imgs, embedding, key  # captions
        return label_imgs, mask_imgs, graph, bbox, objs_vector, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        if cfg.TRAIN.FLAG and self.train_set:
            return len(self.filenames_train)
        else:
            return len(self.filenames_test)

    def collate_fn(self, batch):
        # training set
        if len(batch[0]) == 8:
            label_imgs = list()
            mask_imgs = list()
            wrong_label_imgs = list()
            wrong_mask_imgs = list()
            graph = list()
            bbox = list()
            objs_vector = list()
            key = list()
            # batch
            for b in batch:
                # print(b[0][-1].size())
                # assert False
                label_imgs.append(b[0][-1])
                mask_imgs.append(b[1][-1])
                wrong_label_imgs.append(b[2][-1])
                wrong_mask_imgs.append(b[3][-1])
                graph.append(b[4])
                bbox.append(b[5])
                objs_vector.append(b[6])
                key.append(b[7])
            label_imgs = torch.stack(label_imgs, dim=0)
            return label_imgs, mask_imgs, wrong_label_imgs, wrong_mask_imgs, graph, bbox, objs_vector, key

        # test set
        elif len(batch[0])==6:
            label_imgs = list()
            mask_imgs = list()
            graph = list()
            bbox = list()
            objs_vector = list()
            key = list()
            # batch
            for b in batch:
                label_imgs.append(b[0][-1])
                mask_imgs.append(b[1][-1])
                graph.append(b[2])
                bbox.append(b[3])
                objs_vector.append(b[4])
                key.append(b[5])
            label_imgs = torch.stack(label_imgs, dim=0)
            return label_imgs, mask_imgs, graph, bbox, objs_vector, key
