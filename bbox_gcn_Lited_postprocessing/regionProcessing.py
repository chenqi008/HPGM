# encoding: utf-8
import json
import os
import cv2
import shapely.ops
import math
import numpy as np
import datetime
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Point, Polygon

image_size = 256


class RegionProcessor():
    def __init__(self, json_data=None, coord_data=None):
        self.json_data = json_data
        self.coord_data = coord_data

    # get lines of boundary from room's box.
    # line: x1, y1, x2, y2, make sure x1<=x2 and y1<=y2
    # processed_room: width, height, location
    def get_lines_from_json(self):
        lines = []
        if not self.json_data:
            [bbox_pred, bbox_type, room_class] = self.coord_data
            processed_rooms = dict()
            rooms_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(len(bbox_type)):
                rooms_counter[bbox_type[i]] += 1
                key = "%s%s" % (room_class[bbox_type[i]], rooms_counter[bbox_type[i]])
                min_x = int(float(bbox_pred[i][0]) * image_size)
                min_y = int(float(bbox_pred[i][1]) * image_size)
                max_x = int(float(bbox_pred[i][2]) * image_size)
                max_y = int(float(bbox_pred[i][3]) * image_size)
                lines.append([min_x, min_y, max_x, min_y])
                lines.append([min_x, max_y, max_x, max_y])
                lines.append([max_x, min_y, max_x, max_y])
                lines.append([min_x, min_y, min_x, max_y])
                width = (max_x - min_x + 1) / 2.0
                height = (max_y - min_y + 1) / 2.0
                location = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]
                processed_room = {'location': location, 'width': width, 'height': height}
                processed_rooms[key] = processed_room
        else:
            rooms = self.json_data["rooms"]
            processed_rooms = dict()
            for key in rooms:
                room = rooms[key]
                min_x = int(float(room["min_x"]) * image_size)
                min_y = int(float(room["min_y"]) * image_size)
                max_x = int(float(room["max_x"]) * image_size)
                max_y = int(float(room["max_y"]) * image_size)
                lines.append([min_x, min_y, max_x, min_y])
                lines.append([min_x, max_y, max_x, max_y])
                lines.append([max_x, min_y, max_x, max_y])
                lines.append([min_x, min_y, min_x, max_y])
                width = (max_x - min_x + 1)/2.0
                height = (max_y - min_y + 1)/2.0
                location = [(min_x + max_x)/2.0, (min_y + max_y)/2.0]
                processed_room = {'location': location, 'width': width, 'height': height}
                processed_rooms[key] = processed_room
        return lines, processed_rooms

    # merge lines based on the longer line. method can be replaced by interpolate
    def merge(self, line1, line2):
        merge_min_y = min(line1[1], line2[1])
        merge_max_y = max(line1[3], line2[3])
        merge_min_x = min(line1[0], line2[0])
        merge_max_x = max(line1[2], line2[2])
        # vertical line
        if line1[0] == line1[2]:
            length1 = line1[3] - line1[1]
            length2 = line2[3] - line2[1]
            if length1 > length2:
                merge_line = [line1[0], merge_min_y, line1[2], merge_max_y]
            else:
                merge_line = [line2[0], merge_min_y, line2[2], merge_max_y]
        # horizontal line
        else:
            length1 = line1[2] - line1[0]
            length2 = line2[2] - line2[0]
            if length1 > length2:
                merge_line = [merge_min_x, line1[1], merge_max_x, line1[3]]
            else:
                merge_line = [merge_min_x, line2[1], merge_max_x, line2[3]]
        return merge_line

    # merge lines when close enough
    # parallel_distance_threshold: distance threshold in same direction of lines
    # perpendicular_distance_threshold: distance threshold in perpendicular direction of lines
    def merge_lines(self, lines, parallel_distance_threshold, perpendicular_distance_threshold, max_iteration):
        iteration = 0
        while iteration < max_iteration:
            iteration += 1
            i = 0
            while i < len(lines):
                j = i + 1
                while j < len(lines):
                    flag = False
                    line1 = lines[i]
                    line2 = lines[j]
                    merge_min_y = min(line1[1], line2[1])
                    merge_max_y = max(line1[3], line2[3])
                    merge_min_x = min(line1[0], line2[0])
                    merge_max_x = max(line1[2], line2[2])
                    merge_y_distance = merge_max_y - merge_min_y
                    merge_x_distance = merge_max_x - merge_min_x
                    # vertical line
                    if line1[0] == line1[2] and line2[0] == line2[2]:
                        if merge_max_x - merge_min_x < perpendicular_distance_threshold and \
                                line1[3] - line1[1] + line2[3] - line2[1] + parallel_distance_threshold \
                                > merge_y_distance:
                            lines[i] = self.merge(line1, line2)
                            lines.pop(j)
                            flag = True
                    # horizontal line
                    elif line1[1] == line1[3] and line2[1] == line2[3]:
                        if merge_max_y - merge_min_y < perpendicular_distance_threshold and \
                                line1[2] - line1[0] + line2[2] - line2[0] + parallel_distance_threshold \
                                > merge_x_distance:
                            lines[i] = self.merge(line1, line2)
                            lines.pop(j)
                            flag = True
                    if flag is not True:
                        j = j + 1
                i = i + 1
        return lines

    # extend lines to intersection within distance threshold to generate closed polygon
    def extend_lines_to_intersection(self, lines, distance_threshold):
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines):
                line1 = lines[i]
                line2 = lines[j]
                # line1 is vertical to line2
                if (line1[0] == line1[2]) != (line2[0] == line2[2]):
                    # if line1 is vertical
                    if line1[0] == line1[2]:
                        intersection = [line1[0], line2[1]]
                        i_in_extended_range_of_j = \
                            (line2[0] - distance_threshold < line1[0] < line2[2] + distance_threshold)
                        i_in_range_of_j = (line2[0] <= line1[0] <= line2[2])
                        j_in_extended_range_of_i = \
                            (line1[1] - distance_threshold < line2[1] < line1[3] + distance_threshold)
                        j_in_range_of_i = (line1[1] <= line2[1] <= line1[3])
                        # if in extended range but not in range ,then the line can be extended
                        if i_in_extended_range_of_j and j_in_extended_range_of_i:
                            if not j_in_range_of_i:
                                if intersection[1] > line1[3]:
                                    lines[i][3] = intersection[1]
                                else:
                                    lines[i][1] = intersection[1]
                            if not i_in_range_of_j:
                                if intersection[0] > line2[2]:
                                    lines[j][2] = intersection[0]
                                else:
                                    lines[j][0] = intersection[0]

                    # if line1 is horizontal
                    else:
                        intersection = [line2[0], line1[1]]
                        i_in_extended_range_of_j = \
                            (line2[1] - distance_threshold < line1[1] < line2[3] + distance_threshold)
                        i_in_range_of_j = (line2[1] <= line1[1] <= line2[3])
                        j_in_extended_range_of_i = \
                            (line1[0] - distance_threshold < line2[0] < line1[2] + distance_threshold)
                        j_in_range_of_i = (line1[0] <= line2[0] <= line1[2])
                        # if in extended range but not in range ,then the line can be extended
                        if i_in_extended_range_of_j and j_in_extended_range_of_i:
                            if not j_in_range_of_i:
                                if intersection[0] > line1[2]:
                                    lines[i][2] = intersection[0]
                                else:
                                    lines[i][0] = intersection[0]
                            if not i_in_range_of_j:
                                if intersection[1] > line2[3]:
                                    lines[j][3] = intersection[1]
                                else:
                                    lines[j][1] = intersection[1]
                j = j + 1
            i = i + 1
        return lines

    # decompose lines from intersections
    def lines_decomposition(self, lines):
        i = 0
        decomposed_line = []
        while i < len(lines):
            j = 0
            intersections = []
            line1 = lines[i]
            # if line1 is vertical
            if line1[0] == line1[2]:
                intersections.append(line1[1])
            # if line1 is horizontal
            else:
                intersections.append(line1[0])
            # get intersections of line1 with all the other lines
            while j < len(lines):
                if j != i:
                    line2 = lines[j]
                    # if line1 is vertical
                    if line1[0] == line1[2]:
                        if line1[1] <= line2[1] <= line1[3] and line2[0] <= line1[0] <= line2[2]:
                            intersections.append(line2[1])
                    else:
                        if line1[0] <= line2[0] <= line1[2] and line2[1] <= line1[1] <= line2[3]:
                            intersections.append(line2[0])
                j = j + 1
            # rebuild lines from start and end and all intersections of line1
            if line1[0] == line1[2]:
                intersections.append(line1[3])
                intersections.sort()
                for index in range(len(intersections) - 1):
                    if intersections[index] != intersections[index + 1]:  # remove duplicated points
                        decomposed_line.append([line1[0], intersections[index], line1[0], intersections[index + 1]])
            else:
                intersections.append(line1[2])
                intersections.sort()
                for index in range(len(intersections) - 1):
                    if intersections[index] != intersections[index + 1]:  # remove duplicated points
                        decomposed_line.append([intersections[index], line1[1], intersections[index + 1], line1[1]])
            i = i + 1
        return decomposed_line

    def generate_room_polygon_relation(self, polygons, processed_rooms):
        room_polygon_relation_map = dict()  # store all the polygon of a room
        polygon_room_weight_map = dict()  # store weight of a room to a polygon
        for x in range(0, 256):  # calculate weight of every position to rooms in a 256 * 256 grid
            for y in range(0, 256):
                point = Point(x, y)
                for index in range(0, len(polygons)):
                    if polygons[index].contains(point):
                        for key in processed_rooms:
                            processed_room = processed_rooms[key]
                            weight = self.weight_function(point, processed_room['location'], processed_room['width'],
                                                          processed_room['height'])
                            if index in polygon_room_weight_map.keys():
                                room_weight_map = polygon_room_weight_map[index]
                                if key in room_weight_map.keys():
                                    room_weight_map[key] = room_weight_map[key] + weight
                                else:
                                    room_weight_map[key] = weight
                                polygon_room_weight_map[index] = room_weight_map
                            else:
                                room_weight_map = dict()
                                room_weight_map[key] = weight
                                polygon_room_weight_map[index] = room_weight_map
                        break
        for index in polygon_room_weight_map:
            room_weight_map = polygon_room_weight_map[index]
            max_weight = 0
            max_weight_room = 'none'
            for room in room_weight_map:
                if max_weight < room_weight_map[room]:
                    max_weight = room_weight_map[room]
                    max_weight_room = room
            if max_weight_room != 'none':
                if max_weight_room in room_polygon_relation_map.keys():
                    room_polygon_relation = room_polygon_relation_map[max_weight_room]
                    room_polygon_relation.append(polygons[index])
                else:
                    room_polygon_relation = [polygons[index]]
                    room_polygon_relation_map[max_weight_room] = room_polygon_relation
        return room_polygon_relation_map  # some room may not have polygons

    def weight_function(self, point, location, width, height):
        weight = 1 / width / height \
                 * math.exp(-pow((point.x - location[0]) / width, 2) - pow((point.y - location[1]) / height, 2))
        return weight

    # merge all polygons of a room. Merged result can be a Polygon or MultiPolygon with or without interior holes
    def merge_polygon(self, room_polygon_relation_map):
        merged_room_polygon_relation_map = dict()
        for room in room_polygon_relation_map:
            polygons = room_polygon_relation_map[room]
            merged_polygon = polygons[0]
            if len(polygons) > 1:
                for index in range(1, len(polygons)):
                    merged_polygon = merged_polygon.union(polygons[index])
            merged_room_polygon_relation_map[room] = merged_polygon
        return merged_room_polygon_relation_map

def get_merge_image(lines, rooms, processor, dir_path, count):
    lines = processor.merge_lines(lines, 4, 9, 6)
    print("merge process result:")
    print(lines)
    lines = processor.extend_lines_to_intersection(lines, 10)
    print("connect process result:")
    print(lines)
    lines = processor.lines_decomposition(lines)
    print("decompose process result:")
    print(lines)

    shape_lines = []
    for i in range(0, len(lines)):
        if lines[i][0] != lines[i][2] or lines[i][1] != lines[i][3]:
            shape_lines.append(((lines[i][0], lines[i][1]), (lines[i][2], lines[i][3])))
        else:
            print(" zero length line warning!")
    polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(shape_lines)
    print('polygons size:' + str(len(polygons)))
    print('dangles size:' + str(len(dangles)))
    print('cuts size:' + str(len(cuts)))
    print('invalids size:' + str(len(invalids)))

    array = np.ndarray((256, 256, 3), np.uint8)
    array[:, :, 0] = 0
    array[:, :, 1] = 0
    array[:, :, 2] = 100
    line_image = Image.fromarray(array)
    draw = ImageDraw.Draw(line_image)
    for i in range(0, len(lines)):
        draw.line(lines[i], 'cyan')
    # line_image.show()
    # Image.Image.save(line_image, "line_image.jpg")

    region_image = Image.fromarray(array)
    draw = ImageDraw.Draw(region_image)
    print('polygons result:')
    print(polygons)
    for i in range(0, len(polygons)):
        for j in range(0, len(polygons[i].exterior.coords)):
            start = polygons[i].exterior.coords[j]
            end = polygons[i].exterior.coords[(j + 1) % len(polygons[i].exterior.coords)]
            draw.line([start[0], start[1], end[0], end[1]], 'cyan')
    # region_image.show()
    # Image.Image.save(region_image, "region_image.jpg")

    start = datetime.datetime.now()
    room_polygon_relation_map = processor.generate_room_polygon_relation(polygons, rooms)
    end = datetime.datetime.now()
    print('generate room polygon relation time cost:')
    print((end - start))
    merged_room_polygon_relation_map = processor.merge_polygon(room_polygon_relation_map)
    print('merged polygons result:')
    print(merged_room_polygon_relation_map)

    merged_region_image = Image.fromarray(array)
    draw = ImageDraw.Draw(merged_region_image)
    for room in merged_room_polygon_relation_map:
        polygon = merged_room_polygon_relation_map[room]
        if isinstance(polygon, Polygon):  # polygon can be a Polygon or MultiPolygon with or without interior holes
            for i in range(0, len(polygon.exterior.coords)):
                start = polygon.exterior.coords[i]
                end = polygon.exterior.coords[(i + 1) % len(polygon.exterior.coords)]
                draw.line([start[0], start[1], end[0], end[1]], 'cyan')
        else:
            for i in range(0, len(polygon.geoms)):
                for j in range(0, len(polygon.geoms[i].exterior.coords)):
                    start = polygon.geoms[i].exterior.coords[j]
                    end = polygon.geoms[i].exterior.coords[(j + 1) % len(polygon.geoms[i].exterior.coords)]
                    draw.line([start[0], start[1], end[0], end[1]], 'cyan')
    # merged_region_image.show()
    image_path = os.path.join(dir_path, "merged_region_image_{}.jpg".format(count))
    Image.Image.save(merged_region_image, image_path)


if __name__ == '__main__':
    # json_dir = '/Users/ball/Document/smil/chenqi-CVPR拓展/Code/ours_bbox_gcn_parallel/data/Text_eval/count_000000055.json'
    json_dir = "/home/zhoujiaqiu/Code/GAN/output_bbox_gcn/layout_3stages_2020_03_15_22_46_56/Text_eval_gt/count_000000001.json"
    # json_dir = "/Users/ball/Document/smil/chenqi-CVPR拓展/Code/ours_bbox_gcn_Lited/count_000000001.json"
    from io import open
    f = open(json_dir, encoding="utf-8")
    input_json = json.load(f)
    processor = RegionProcessor(input_json)
    lines, rooms = processor.get_lines_from_json()
    lines = processor.merge_lines(lines, 4, 9, 6)
    print("merge process result:")
    print(lines)
    lines = processor.extend_lines_to_intersection(lines, 10)
    print("connect process result:")
    print(lines)
    lines = processor.lines_decomposition(lines)
    print("decompose process result:")
    print(lines)

    shape_lines = []
    for i in range(0, len(lines)):
        if lines[i][0] != lines[i][2] or lines[i][1] != lines[i][3]:
            shape_lines.append(((lines[i][0], lines[i][1]), (lines[i][2], lines[i][3])))
        else:
            print(" zero length line warning!")
    polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(shape_lines)
    print('polygons size:' + str(len(polygons)))
    print('dangles size:' + str(len(dangles)))
    print('cuts size:' + str(len(cuts)))
    print('invalids size:' + str(len(invalids)))

    array = np.ndarray((256, 256, 3), np.uint8)
    array[:, :, 0] = 0
    array[:, :, 1] = 0
    array[:, :, 2] = 100
    line_image = Image.fromarray(array)
    draw = ImageDraw.Draw(line_image)
    for i in range(0, len(lines)):
        draw.line(lines[i], 'cyan')
    # line_image.show()
    Image.Image.save(line_image, "line_image.jpg")

    region_image = Image.fromarray(array)
    draw = ImageDraw.Draw(region_image)
    print('polygons result:')
    print(polygons)
    for i in range(0, len(polygons)):
        for j in range(0, len(polygons[i].exterior.coords)):
            start = polygons[i].exterior.coords[j]
            end = polygons[i].exterior.coords[(j + 1) % len(polygons[i].exterior.coords)]
            draw.line([start[0], start[1], end[0], end[1]], 'cyan')
    # region_image.show()
    Image.Image.save(region_image, "region_image.jpg")

    start = datetime.datetime.now()
    room_polygon_relation_map = processor.generate_room_polygon_relation(polygons, rooms)
    end = datetime.datetime.now()
    print('generate room polygon relation time cost:')
    print((end - start))
    merged_room_polygon_relation_map = processor.merge_polygon(room_polygon_relation_map)
    print('merged polygons result:')
    print(merged_room_polygon_relation_map)

    merged_region_image = Image.fromarray(array)
    draw = ImageDraw.Draw(merged_region_image)
    for room in merged_room_polygon_relation_map:
        polygon = merged_room_polygon_relation_map[room]
        if isinstance(polygon, Polygon):  # polygon can be a Polygon or MultiPolygon with or without interior holes
            for i in range(0, len(polygon.exterior.coords)):
                start = polygon.exterior.coords[i]
                end = polygon.exterior.coords[(i + 1) % len(polygon.exterior.coords)]
                draw.line([start[0], start[1], end[0], end[1]], 'cyan')
        else:
            for i in range(0, len(polygon.geoms)):
                for j in range(0, len(polygon.geoms[i].exterior.coords)):
                    start = polygon.geoms[i].exterior.coords[j]
                    end = polygon.geoms[i].exterior.coords[(j + 1) % len(polygon.geoms[i].exterior.coords)]
                    draw.line([start[0], start[1], end[0], end[1]], 'cyan')
    # merged_region_image.show()
    Image.Image.save(merged_region_image, "merged_region_image.jpg")
