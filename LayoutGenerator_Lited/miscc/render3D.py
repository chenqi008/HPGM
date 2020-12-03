import os 
import sys
import open3d
import numpy as np
import cv2 as cv
from miscc.getContour import check_contour_hull, visualization
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '../..')))
sys.path.append(dir_path)

def get_center_point(point_hull):
    """
    function: get the center point from the contour
    """
    contour = cv.convexHull(point_hull, returnPoints=True)
    m = cv.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return [cx, cy]


def draw_point(point_hulls, save_img_path):
    """
    function: to paint a image on a white image
    """
    image = np.ones((256, 256)) * 255
    for i in range(len(point_hulls)):
        image = cv.circle(image, (int(point_hulls[i][0]), int(point_hulls[i][1])), 2, (0, 0, 255))
    cv.imwrite(save_img_path, image)


def generate_triangle(triangle_index, point_hull, shift_index=0):
    # function: apply the points real position and generate a surface triangle
    # condition: triangle_index must be in clockwise
    # point_position:[0, 2, 3, 1]-->[[3, 1], generate_triangle:[[0, 2, 3],
    #                                [2, 0]]                    [0, 3, 1]]
    if point_hull is None:
        triangle = np.zeros((len(triangle_index)-2, 3))
        for i in range(triangle.shape[0]):
            triangle[i] = [triangle_index[0]+shift_index, triangle_index[i+1]+shift_index, triangle_index[i+2]+shift_index]
    else:
        triangle = np.zeros((len(triangle_index) - 2, 3))
        for i in range(triangle.shape[0]):
            triangle[i] = [triangle_index[0] + shift_index, triangle_index[i + 1] + shift_index,
                           triangle_index[i + 2] + shift_index]
        # rect = cv.boundingRect(point_hull)
        # rect = (0, 0, 255, 255)
        # hull_subdiv = cv.Subdiv2D(rect)
        # for i in range(len(point_hull)):
        #     hull_subdiv.insert((point_hull[i][0], point_hull[i][1]))
        # hull_triangle = hull_subdiv.getTriangleList()
    return triangle


def clockwise(index):
    # tran [0, 1, 2, 3] --> [0, 2, 3, 1]
    for i in range(len(index)):
        temp = index[i][1]
        index[i][1] = index[i][2]
        index[i][2] = index[i][3]
        index[i][3] = temp
    return index


def generate_line_set(point_hull, bottom_height, height, max_size):
    point_num = np.array(point_hull).shape[0]
    # generate every line link
    vertical_link_1,vertical_link_2 = np.zeros((point_num, 3)), np.zeros((point_num, 3))
    # downside_link = np.zeros((point_num, 3))
    upside_link_1, upside_link_2 = np.zeros((point_num, 3)), np.zeros((point_num, 3))
    upside_back_link_1 = np.zeros((point_num, 3))
    contour_3d_hull_1, contour_3d_hull_2 = np.zeros((point_num * 3, 3)), np.zeros((point_num * 3, 3))
    for i in range(point_num):
        point = point_hull[i]
        contour_3d_hull_1[3 * i] = [point[0] / max_size, point[1] / max_size, bottom_height/max_size]
        contour_3d_hull_1[3 * i + 1] = [point[0] / max_size, point[1] / max_size, height / max_size]
        contour_3d_hull_1[3 * i + 2] = [point[0] / max_size+0.002, point[1] / max_size+0.002, height / max_size]
        vertical_link_1[i] = [3*i, 3*i+1, 3*i+2]
        upside_link_1[i] = [3*i+1, 3*i+2, (3*i+4) % (point_num*3)]
        upside_back_link_1[i] = [(3*i+4)% (point_num*3), (3*i+5)% (point_num*3), 3*i+2]
    for i in range(point_num):
        point = point_hull[i]
        contour_3d_hull_2[3 * i] = [point[0] / max_size, point[1] / max_size, bottom_height/max_size]
        contour_3d_hull_2[3 * i + 1] = [point[0] / max_size+0.002, point[1] / max_size+0.002, bottom_height / max_size]
        contour_3d_hull_2[3 * i + 2] = [point[0] / max_size, point[1] / max_size, height / max_size]
        vertical_link_2[i] = [3*i, 3*i+1, 3*i+2]
        upside_link_2[i] = [3*i, 3*i+1, (3*i+3) % (point_num*3)]
    lines_link_1 = np.vstack((vertical_link_1, upside_link_1))
    lines_link_2 = np.vstack((vertical_link_2+contour_3d_hull_1.shape[0], upside_link_2+contour_3d_hull_2.shape[0]))
    lines_link = np.vstack((lines_link_1, lines_link_2))
    lines_link = np.vstack((lines_link, upside_back_link_1))
    lines_vertices = np.vstack((contour_3d_hull_1, contour_3d_hull_2))
    return lines_vertices, lines_link

def generate_3d_contour(point_hull, height, bottom_height, shift_triangle_index, max_size, surface_type):
    """
    function: tran the point_rectangle to 3D vertices contour with height
              generate 3d point from line or a plane
    condition: point_hull:[[[x,y]],[[x,y]],...] the point must be in clockwise direction
    """
    point_num = np.array(point_hull).shape[0]
    contour_3d_hull = np.zeros((point_num * 2, 3))
    for i in range(point_num):
        point = point_hull[i]
        contour_3d_hull[2 * i] = [point[0] / max_size, point[1] / max_size, bottom_height/max_size]
        contour_3d_hull[2 * i + 1] = [point[0] / max_size, point[1] / max_size, height / max_size]
    # generate every surface triangle
    triangle_index = [i for i in range(contour_3d_hull.shape[0])]
    surface_index = [triangle_index[i:i + 4] for i in range(0, len(triangle_index), 2)]
    surface_index[-1].append(triangle_index[0])
    surface_index[-1].append(triangle_index[1])
    surface_index = clockwise(surface_index)
    # generate the up and down face
    downside_index = [i * 2 for i in range(point_num)]
    upside_index = [i * 2 + 1 for i in range(point_num)]
    # point_hull = np.insert(point_hull, 0, get_center_point(point_hull),0)
    if surface_type == "downside":
        contour_3d_triangle = generate_triangle(downside_index, point_hull, shift_index=shift_triangle_index)
    elif surface_type == "upside":
        contour_3d_triangle = generate_triangle(upside_index, point_hull, shift_index=shift_triangle_index)
    else:
        downside_3d_triangle = generate_triangle(downside_index, point_hull, shift_index=shift_triangle_index)
        upside_3d_triangle = generate_triangle(upside_index, point_hull, shift_index=shift_triangle_index)
        contour_3d_triangle = np.vstack((downside_3d_triangle, upside_3d_triangle))
    # get the surface_triangle
    for i in range(len(surface_index)):
        surface_triangle = generate_triangle(surface_index[i], None, shift_index=shift_triangle_index)
        contour_3d_triangle = np.vstack((contour_3d_triangle, surface_triangle))
    return contour_3d_hull, contour_3d_triangle


def corrected_coord(point_hull, max_size):
    for i in range(len(point_hull)):
        point_hull[i][0][1] = max_size - point_hull[i][0][1]
    corrected_point_hull = point_hull
    return corrected_point_hull


def generate_wall(point_hull, shift_triangle_index, bottom_height, height, contour_point_hull, wall_thickness=3, max_size=256):
    """
    function: trans the surface to the wall have thickness; split to 4 points to represent a plane
    condition: point_hull in 2D pic: hull.png follow clock_wise
    """
    point_num = len(point_hull)
    wall_point_hull = np.zeros((point_num, 4, 2))
    remove_index = []
    # generate the 2D wall_point_hull
    for i in range(point_num):
        point = point_hull[i]
        if i+1 == point_num:
            adjacent_point = point_hull[0]
        else:
            adjacent_point = point_hull[i+1]
        wall_point_hull[i][0] = [point[0], point[1]]
        wall_point_hull[i][1] = [adjacent_point[0], adjacent_point[1]]
        wall_line = [wall_point_hull[i][0], wall_point_hull[i][1]]
        # judge which line belongs to contour
        # ****** if want show wall please hide this **********
        if line_is_on_contour(wall_line, contour_point_hull):
            remove_index.append(i)
        # judge the vertical line or the horizon line
        if point[1] == adjacent_point[1]:
            wall_point_hull[i][2] = [adjacent_point[0], adjacent_point[1] - wall_thickness]
            wall_point_hull[i][3] = [point[0], point[1] - wall_thickness]
        elif point[0] == adjacent_point[0]:
            wall_point_hull[i][2] = [adjacent_point[0] - wall_thickness, adjacent_point[1]]
            wall_point_hull[i][3] = [point[0] - wall_thickness, point[1]]
    # remove the index which belongs to the contour
    wall_point_hull = np.delete(wall_point_hull, remove_index, axis=0)
    base_triangle_index = shift_triangle_index
    # build the wall triangle
    wall_3d_hulls = None
    wall_3d_triangles = None
    for i in range(len(wall_point_hull)):
        wall_3d_hull, wall_3d_triangle = generate_3d_contour(wall_point_hull[i], height, bottom_height, shift_triangle_index, max_size, "upside")
        if i == 0:
            wall_3d_hulls = wall_3d_hull
            wall_3d_triangles = wall_3d_triangle
        else:
            wall_3d_hulls = np.vstack((wall_3d_hulls, wall_3d_hull))
            wall_3d_triangles = np.vstack((wall_3d_triangles, wall_3d_triangle))
        shift_triangle_index = base_triangle_index + len(wall_3d_hulls)
    return wall_3d_hulls, wall_3d_triangles


def line_is_on_contour(line, point_hull):
    valid_line = (line[0] - line[1]).astype(int)
    for i in range(len(point_hull)):
        start_point = point_hull[i]
        if i+1 == len(point_hull):
            end_point = point_hull[0]
        else:
            end_point = point_hull[i+1]
        contour_line = (end_point - start_point).astype(int)
        on_line = (start_point - line[0]).astype(int)
        # on line or on extension line; cv.pointPolygonTest:+1(inside);-1(outside);0(on the edge)
        if np.cross(valid_line, contour_line) == 0 and np.cross(valid_line, on_line) == 0:
            if cv.pointPolygonTest(point_hull, (line[0][0], line[0][1]), False) == 0\
            and cv.pointPolygonTest(point_hull, (line[1][0], line[1][1]), False) == 0:
                return True
    return False

def get_contour(hull):
    new_hull = []
    for i in range(len(hull)):
        for l in range(len(hull[i])):
            new_hull.append([hull[i][l][0], hull[i][l][1]])
    new_hull = np.array(new_hull)
    contour_hull = cv.convexHull(new_hull, returnPoints=True)
    contour_hull = np.squeeze(contour_hull)
    contour_hull = check_contour_hull(contour_hull)
    return contour_hull

def render_3d_contour(init_hull, first_hull, Next_hull, save_mesh_path):
    height_scale = 40  # layout height 28  40 4layers:30
    max_size, wall_thick = 256, 1 
    Color = np.array([[84, 139, 84], [0, 100, 0], [0, 0, 128], [85, 26, 139], [255, 0, 255],
                      [165, 42, 42], [139, 134, 130], [205, 198, 115], [139, 58, 58],
                      [255, 255, 255], [0, 0, 0], [30, 144, 255], [135, 206, 235], [255, 255, 0]])
    layouts = open3d.geometry.TriangleMesh()

    point_3d_hulls = []
    point_3d_hulls.append(first_hull)
    for i in range(len(Next_hull)):
        point_3d_hulls.append(Next_hull[i]) # 1
    layout_layer = len(point_3d_hulls)
    for l in range(layout_layer):
        bottom_height = height_scale * l
        height = height_scale * (l+1)
        # 1. generate the Bottom layer face layout
        Bottom_layout = open3d.geometry.TriangleMesh()
        contour_point_hull = get_contour(point_3d_hulls[l])
        for h in range(len(point_3d_hulls[l])):
            Bottom_3d_hulls, Bottom_3d_triangle = generate_3d_contour(point_3d_hulls[0][h],  bottom_height+1, bottom_height, 0, max_size, "Both")
            Bottom_layout.vertices = open3d.utility.Vector3dVector(Bottom_3d_hulls)
            Bottom_layout.triangles = open3d.utility.Vector3iVector(Bottom_3d_triangle)
            layouts = layouts + Bottom_layout
        # 2. generate the 3D layout wall
        for n in range(len(point_3d_hulls[l])):
            point_hull = np.squeeze(point_3d_hulls[l][n]).astype(int)
            wall_hulls, wall_triangles = generate_wall(point_hull, 0, bottom_height, height, contour_point_hull, wall_thickness=wall_thick)
            wall = open3d.geometry.TriangleMesh()
            if wall_hulls is not None:
                wall.vertices = open3d.utility.Vector3dVector(wall_hulls)
                wall.triangles = open3d.utility.Vector3iVector(wall_triangles)
            layouts = layouts + wall
    layout_lines = open3d.geometry.TriangleMesh()
    layout_hulls, layout_triangles = generate_line_set(contour_point_hull, 0, height, max_size)
    layout_lines.vertices = open3d.utility.Vector3dVector(layout_hulls)
    layout_lines.triangles = open3d.utility.Vector3iVector(layout_triangles)
    layout_lines.paint_uniform_color([0, 0, 0])
    # save 
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    layouts = layouts #+ axis_pcd
    layouts = layouts.remove_duplicated_vertices()
    layouts = layouts.remove_duplicated_triangles()
    layouts = layouts.compute_vertex_normals()
    layouts = layouts.compute_triangle_normals()
    print("hello")
    open3d.io.write_triangle_mesh(save_mesh_path, layouts+layout_lines)
    # open3d.visualization.draw_geometries([layouts]+lines, mesh_show_back_face=True)


if __name__ == "__main__":
    def load_data_path():
        # /***load the layout coord and type***/
        abosulte_path = "/Users/ball/Public/Code/Project/Layout/Layout_bbox_gcn/multiLayerLayout/data"
        # data_hulls_path = os.path.join(abosulte_path, "{}/layout_score_2.2.npy".format(data_dir))
        # point_3d_hulls = np.load(data_hulls_path)
        num = 48
        init_hull_path = os.path.join(abosulte_path, "{}/layout_init_hulls_{}.npy".format(data_dir, num))
        # /***for cover experiment******/
        first_hull_path = os.path.join(abosulte_path, "{}/layout_gt_hulls_{}.npy".format(data_dir, num))
        # second_hull_path = os.path.join(abosulte_path, "{}/random_cover_second.npy".format(data_dir))
        # third_hull_path = os.path.join(abosulte_path, "{}/cover_third_layout.npy".format(data_dir))
        first_hull = np.load(first_hull_path)
        # second_hull = np.load(second_hull_path)
        # third_hull = np.load(third_hull_path)
        # /*****for ablation experiment******/
        w_all_hull_path = os.path.join(abosulte_path, "{}/layout_score_hulls_{}.npy".format(data_dir, num))
        w_all_hull = np.load(w_all_hull_path)
        init_hull = np.load(init_hull_path)
        # return first_hull, second_hull, init_hull
        return first_hull, w_all_hull, init_hull
    data_dir = "supp_visual/visual_4"
    # save_name = "visual_w_all_whole.ply"
    # data_dir = "cover_visual"
    # save_name = "cover_good_whole.ply"
    height_scale = 30  # layout height 28  40 4layers:30
    max_size = 256 # 256 --> 1
    wall_thick = 1
    # for cover
    # first_hull, second_hull, init_hull = load_data_path()
    # init_hull = np.round(init_hull*256)
    # visualization([init_hull], "./contour.png")
    # assert False
    first_hull, w_all_hull, init_hull = load_data_path()
    point_3d_hulls = []
    point_3d_hulls.append(first_hull)
    point_3d_hulls.append(w_all_hull[2]) # 1
    point_3d_hulls.append(w_all_hull[6]) # 3
    point_3d_hulls.append(w_all_hull[3])

    # point_3d_hulls.append(second_hull[5])
    # point_3d_hulls.append(second_hull[4])
    # point_3d_hulls.append(point_3d_hulls[2])
    # point_3d_hulls.append(point_3d_hulls[5])
    Color = np.array([[84, 139, 84], [0, 100, 0], [0, 0, 128], [85, 26, 139], [255, 0, 255],
                      [165, 42, 42], [139, 134, 130], [205, 198, 115], [139, 58, 58],
                      [255, 255, 255], [0, 0, 0], [30, 144, 255], [135, 206, 235], [255, 255, 0]])
    # Color = np.array([[128, 128, 128], [128, 128, 128], [128, 128, 128], [128, 128, 128],[128, 128, 128]])
    # example:point_hull = [[[195.7632, 165.6576]], [[52.1984, 165.6576]], [[52.1984, 107.4176]],
    #                       [[74.2912, 107.4176]], [[74.2912,  87.3472]], [[195.7632,  87.3472]]]
    layouts = open3d.geometry.TriangleMesh()
    layout_layer = 4
    # point_3d_hulls = point_3d_hulls[0]
    # contour_point_hull = np.round(init_hull*256) #np.array(list(point_3d_hulls[0])).astype(int)
    lines = []
    for l in range(layout_layer):
        bottom_height = height_scale * l
        height = height_scale * (l+1)
        # 1. generate the Bottom layer face layout
        Bottom_layout = open3d.geometry.TriangleMesh()
        contour_point_hull = get_contour(point_3d_hulls[l])
        ### for some special process
        if l == 1:
            for h in range(len(point_3d_hulls[l+1])):
                Bottom_3d_hulls, Bottom_3d_triangle = generate_3d_contour(point_3d_hulls[l+1][h],  bottom_height+1, bottom_height, 0, max_size, "Both")
                Bottom_layout.vertices = open3d.utility.Vector3dVector(Bottom_3d_hulls)
                Bottom_layout.triangles = open3d.utility.Vector3iVector(Bottom_3d_triangle)
                layouts = layouts + Bottom_layout
        for h in range(len(point_3d_hulls[l])):
            Bottom_3d_hulls, Bottom_3d_triangle = generate_3d_contour(point_3d_hulls[l][h],  bottom_height+1, bottom_height, 0, max_size, "Both")
            Bottom_layout.vertices = open3d.utility.Vector3dVector(Bottom_3d_hulls)
            Bottom_layout.triangles = open3d.utility.Vector3iVector(Bottom_3d_triangle)
            layouts = layouts + Bottom_layout
        # Bottom_layout.paint_uniform_color([0.65, 0.65, 0.65])

        # # # 2. generate the 3D layout wall
        for n in range(len(point_3d_hulls[l])):
            point_hull = np.squeeze(point_3d_hulls[l][n]).astype(int)
            wall_hulls, wall_triangles = generate_wall(point_hull, 0, bottom_height, height, contour_point_hull, wall_thickness=wall_thick)
            wall = open3d.geometry.TriangleMesh()
            if wall_hulls is not None:
                wall.vertices = open3d.utility.Vector3dVector(wall_hulls)
                wall.triangles = open3d.utility.Vector3iVector(wall_triangles)
            # wall.paint_uniform_color([0.65, 0.65, 0.65])
            layouts = layouts + wall
        # # 3. draw the whole contour line set
        lines_pcd = open3d.geometry.LineSet()
        lines_vertices, lines_link = generate_line_set(contour_point_hull, bottom_height, height)
        colors = [[0, 0, 0] for i in range(len(lines_link))]
        lines_pcd.lines = open3d.utility.Vector2iVector(lines_link)
        lines_pcd.points = open3d.utility.Vector3dVector(lines_vertices)
        lines_pcd.colors = open3d.utility.Vector3dVector(colors)
        lines.append(lines_pcd)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    layouts = layouts #+ axis_pcd
    layouts = layouts.remove_duplicated_vertices()
    layouts = layouts.remove_duplicated_triangles()
    # layouts = layouts.subdivide_midpoint(number_of_iterations=1)
    # layouts = layouts.filter_smooth_laplacian(number_of_iterations=1)
    layouts = layouts.compute_vertex_normals()
    layouts = layouts.compute_triangle_normals()
    print("hello")
    # experiment_path = "/Users/ball/Public/Code/Project/Layout/Layout_bbox_gcn/multiLayerLayout/experiment_result/{}".format(data_dir)
    # save_mesh_path = os.path.join(experiment_path, save_name)
    # open3d.io.write_triangle_mesh(save_mesh_path, layouts)
    open3d.visualization.draw_geometries([layouts]+lines, mesh_show_back_face=True)
