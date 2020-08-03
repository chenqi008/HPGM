import os
import errno
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





