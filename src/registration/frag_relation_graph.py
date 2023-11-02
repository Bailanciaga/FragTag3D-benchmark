import open3d as o3d
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
# folder_path = '../../data/TUWien/brick'
import logging
import numpy as np
from plyfile import PlyData, PlyParseError
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(message)s')


def analyze_ply(file_path):
    try:
        ply_data = PlyData.read(file_path)
        vertex_data = np.array(ply_data['vertex'].data.tolist())
        face_data = ply_data['face'].data

        vertex_count = len(vertex_data)
        face_count = len(face_data)

        color_data = vertex_data[:, 3:6]  # Assume color data is the last three columns
        non_white_vertex_indices = np.any(color_data != [255, 255, 255], axis=1)
        non_white_vertex_count = np.sum(non_white_vertex_indices)

        non_white_face_count = 0
        color_vertex_dict = {}
        color_face_dict = {}

        for face in face_data:
            vertex_indices = face[0]
            face_colors = color_data[vertex_indices]
            non_white_in_face = np.any(face_colors != [255, 255, 255], axis=1)
            if np.any(non_white_in_face):
                non_white_face_count += 1
                for color in face_colors[non_white_in_face]:
                    color_tuple = tuple(color)
                    color_vertex_dict[color_tuple] = color_vertex_dict.get(color_tuple, 0) + 1
                    color_face_dict[color_tuple] = color_face_dict.get(color_tuple, 0) + 1

        max_vertex_count = max(color_vertex_dict.values(), default=0)
        min_vertex_count = min(color_vertex_dict.values(), default=0)
        max_face_count = max(color_face_dict.values(), default=0)
        min_face_count = min(color_face_dict.values(), default=0)

        result = {
            'vertex_count': vertex_count,
            'face_count': face_count,
            'non_white_vertex_count': non_white_vertex_count,
            'non_white_face_count': non_white_face_count,
            'max_vertex_count': max_vertex_count,
            'min_vertex_count': min_vertex_count,
            'max_face_count': max_face_count,
            'min_face_count': min_face_count,
            'color_vertex_count': color_vertex_dict,
            'color_face_count': color_face_dict
        }

        return result

    except Exception as e:
        logging.error(f'Failed to analyze PLY file {file_path}: {e}')
        return None  # 返回 None 或其他适当的值，以指示函数未能成功完成


def read_ply_color(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    colors = set()

    for color in pcd.colors:
        r, g, b = [int(c * 255) if not np.isnan(c) else 0 for c in color]
        if (r, g, b) != (255, 255, 255):  # Exclude white color
            colors.add((r, g, b))

    return colors


def create_graph(folder_path):
    graph = nx.Graph()

    ply_colors = {}
    analyze_result = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(folder_path, filename)
            ply_colors[filename] = read_ply_color(filepath)
            result = analyze_ply(filepath)
            analyze_result[filename] = result

    ply_files = {os.path.basename(f): folder_path + '/' + f for f in ply_colors.keys()}
    all_file_combinations = list(itertools.combinations(ply_files.keys(), 2))

    edge_colors = []
    # Create an empty dictionary to store the color of each edge
    edge_color_map = {}

    for file1, file2 in all_file_combinations:
        common_colors = ply_colors[file1].intersection(ply_colors[file2])

        if common_colors:
            chosen_color = list(common_colors)[0]
            sorted_edge = tuple(sorted((file1.split('.')[0], file2.split('.')[0])))
            graph.add_edge(*sorted_edge)
            edge_color_map[sorted_edge] = chosen_color

    # Create a list of edge colors in the order of graph.edges()
    edge_colors = [edge_color_map.get(tuple(sorted(edge)), 'gray') for edge in graph.edges()]
    edge_count = graph.number_of_edges()

    # Draw the graph
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx_nodes(graph, pos)
    # nx.draw_networkx_labels(graph, pos)
    # nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=4)
    # plt.show()
    return graph, ply_files, edge_color_map, edge_count, analyze_result
# if __name__ == '__main__':
#     create_graph(folder_path)