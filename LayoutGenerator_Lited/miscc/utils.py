import os
import errno
import random
import torch


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# def bbox_iou(box1, box2, x1y1x2y2=True):
def bbox_iou(boxes1, boxes2):
    """
    Returns the IoU of two bounding boxes
    """
    # if not x1y1x2y2:
    #     # Transform from center and width to exact coordinates
    #     b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    #     b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    #     b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    #     b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    # else:
    #     # Get the coordinates of bounding boxes
    #     b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    #     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    num_boxes = boxes1.size(0)
    iou = 0.0
    for i in range(num_boxes):
        box1 = boxes1[i]
        box2 = boxes2[i]

        # Get the coordinates of bounding boxes
        # ([left, bottom, right, top] -> [top, left, bottom, right])
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[3], box1[0], box1[1], box1[2]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[3], box2[0], box2[1], box2[2]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou += inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou, num_boxes


def bbox_refiner(boxes):
    threshold = 0.015
    num_boxes = boxes.size(0)
    for i in range(num_boxes):
        for j in range(num_boxes):
            # x_min
            if abs((boxes[i][0]-boxes[j][0]).item()) < threshold:
                boxes[i][0] = boxes[j][0]
                # boxes[i][0] -= (boxes[i][0]-boxes[j][0])
                # boxes[i][2] -= (boxes[i][0]-boxes[j][0])
            elif abs((boxes[i][0]-boxes[j][2]).item()) < threshold:
                boxes[i][0] = boxes[j][2]
                # boxes[i][0] -= (boxes[i][0]-boxes[j][2])
                # boxes[i][2] -= (boxes[i][0]-boxes[j][2])
            # y_min
            if abs((boxes[i][1]-boxes[j][1]).item()) < threshold:
                boxes[i][1] = boxes[j][1]
                # boxes[i][1] -= (boxes[i][1]-boxes[j][1])
                # boxes[i][3] -= (boxes[i][1]-boxes[j][1])
            elif abs((boxes[i][1]-boxes[j][3]).item()) < threshold:
                boxes[i][1] = boxes[j][3]
                # boxes[i][1] -= (boxes[i][1]-boxes[j][3])
                # boxes[i][3] -= (boxes[i][1]-boxes[j][3])
            # x_max
            if abs((boxes[i][2]-boxes[j][0]).item()) < threshold:
                boxes[i][2] = boxes[j][0]
                # boxes[i][0] -= (boxes[i][2]-boxes[j][0])
                # boxes[i][2] -= (boxes[i][2]-boxes[j][0])
            elif abs((boxes[i][2]-boxes[j][2]).item()) < threshold:
                boxes[i][2] = boxes[j][2]
                # boxes[i][0] -= (boxes[i][2]-boxes[j][2])
                # boxes[i][2] -= (boxes[i][2]-boxes[j][2])
            # y_max
            if abs((boxes[i][3]-boxes[j][1]).item()) < threshold:
                boxes[i][3] = boxes[j][1]
                # boxes[i][1] -= (boxes[i][3]-boxes[j][1])
                # boxes[i][3] -= (boxes[i][3]-boxes[j][1])
            elif abs((boxes[i][3]-boxes[j][3]).item()) < threshold:
                boxes[i][3] = boxes[j][3]
                # boxes[i][1] -= (boxes[i][3]-boxes[j][3])
                # boxes[i][3] -= (boxes[i][3]-boxes[j][3])
    return boxes


def generate_data_addition(data):
    label_imgs, origin_graph, origin_objs_vector, bbox, key, origin_obj = data
    # editing addition process one->one->...->all
    # generate data_process_index
    num = torch.nonzero(torch.eq(origin_obj[0], -1))[0]
    obj_type = origin_obj[0][:num]
    data_process_index = []
    living_index = [i for i, x in enumerate(obj_type) if x == 0]
    if len(living_index) == 0:
        living_index.append(0)
    data_process_index.append(int(living_index[0]))
    index_list = [i for i in range(len(obj_type))]
    index_list.pop(living_index[0])
    random.seed(1)
    for i in range(len(index_list)):
        random_index = random.sample(index_list, 1)
        data_process_index.append(int(random_index[0]))
        index_list.remove(int(random_index[0]))
    # data_process_index = [5, 6, 1, 3, 2, 0, 4]
    gen_obj_data = []
    gen_graph_data = []
    gen_type_data = []
    # get the graph connection relationship from the origin graph
    graph = origin_graph[0][:num, :]
    graph_connect = []
    for i in range(len(graph)):
        nonzeros_list = torch.nonzero(graph[i])
        relationship = []
        for i in range(len(nonzeros_list)):
            relationship.append(int(nonzeros_list[i][0]))
        graph_connect.append(relationship)
    for i in range(len(data_process_index)):
        obj_vector = origin_objs_vector[0]
        if i == 0:
            gen_obj_vector = obj_vector[torch.arange(obj_vector.size(0)) == data_process_index[i]]
            gen_obj_vector = gen_obj_vector.unsqueeze(0)
            gen_obj_data.append(gen_obj_vector)
            gen_type = [origin_obj[0][data_process_index[i]]]
            gen_type_data.append(gen_type)
        else:
            gen_obj_old_data = gen_obj_data[i-1].squeeze(0)
            obj_data = obj_vector[torch.arange(obj_vector.size(0)) == data_process_index[i]]
            gen_obj_vector = torch.cat((gen_obj_old_data, obj_data), dim=0)
            gen_obj_vector = gen_obj_vector.unsqueeze(0)
            gen_obj_data.append(gen_obj_vector)
            # build new or it will the same
            gen_type_old_data = list(gen_type_data[i-1])
            gen_type_old_data.append(origin_obj[0][data_process_index[i]])
            gen_type_data.append(gen_type_old_data)
        layout_index = data_process_index[:(i+1)]
        gen_graph = torch.zeros((i+1, i+1))
        # process the line one by one
        for k in range(len(layout_index)):
            for j in range(len(data_process_index)):
                if data_process_index[j] in graph_connect[layout_index[k]] and data_process_index[j] in layout_index:
                    gen_graph[k][j] = 1.0000
            gen_graph[k] = gen_graph[k]/len(torch.nonzero(gen_graph[k]))
        gen_graph = gen_graph.unsqueeze(0)
        gen_graph_data.append(gen_graph)
    return gen_obj_data, gen_graph_data, gen_type_data


def generate_data_reduction(data):
    label_imgs, origin_graph, origin_objs_vector, bbox, key, origin_obj = data
    # editing reduction process 6->1->...->zero
    data_process_index = [5, 4, 0, 2, 3, 1, 6]
    gen_obj_data = []
    gen_graph_data = []
    gen_type_data = []
    for i in range(len(data_process_index)):
        obj_vector = origin_objs_vector[0]
        if i == 0:
            gen_obj_vector = obj_vector[torch.arange(obj_vector.size(0)) == data_process_index[i]]
            gen_obj_vector = gen_obj_vector.unsqueeze(0)
            gen_obj_data.append(gen_obj_vector)
            gen_type = [origin_obj[0][data_process_index[i]]]
            gen_type_data.append(gen_type)
        else:
            gen_obj_old_data = gen_obj_data[i-1].squeeze(0)
            obj_data = obj_vector[torch.arange(obj_vector.size(0)) == data_process_index[i]]
            gen_obj_vector = torch.cat((gen_obj_old_data, obj_data), dim=0)
            gen_obj_vector = gen_obj_vector.unsqueeze(0)
            gen_obj_data.append(gen_obj_vector)
            # build new or it will the same
            gen_type_old_data = list(gen_type_data[i-1])
            gen_type_old_data.append(origin_obj[0][data_process_index[i]])
            gen_type_data.append(gen_type_old_data)
        if i == 0:
            gen_graph = torch.Tensor([[1.0000]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 1:
            gen_graph = torch.Tensor([[0.5000, 0.5000],
                                      [0.5000, 0.5000]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 2:
            gen_graph = torch.Tensor([[0.3333, 0.3333, 0.3333],
                         [0.3333, 0.3333, 0.3333],
                         [0.3333, 0.3333, 0.3333]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 3:
            gen_graph = torch.Tensor([[0.2500, 0.2500, 0.2500, 0.2500],
                         [0.3333, 0.3333, 0.3333, 0.0000],
                         [0.2500, 0.2500, 0.2500, 0.2500],
                         [0.3333, 0.0000, 0.3333, 0.3333]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 4:
            gen_graph = torch.Tensor([[0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000],
                         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
                         [0.2500, 0.0000, 0.2500, 0.2500, 0.2500],
                         [0.3333, 0.0000, 0.0000, 0.3333, 0.3333]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 5:
            gen_graph = torch.Tensor([[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
                         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000],
                         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000],
                         [0.2500, 0.0000, 0.2500, 0.2500, 0.2500, 0.0000],
                         [0.2500, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500],
                         [0.3333, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
        elif i == 6:
            gen_graph = torch.Tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429],
                         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000],
                         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
                         [0.2500, 0.0000, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000],
                         [0.2500, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500, 0.0000],
                         [0.2500, 0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500],
                         [0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333]])
            gen_graph = gen_graph.unsqueeze(0)
            gen_graph_data.append(gen_graph)
    return gen_obj_data, gen_graph_data, gen_type_data




