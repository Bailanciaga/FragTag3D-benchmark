import open3d as o3d
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
# folder_path = '../../data/TUWien/brick'
def read_ply_color(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    colors = set()

    for color in pcd.colors:
        r, g, b = [int(c * 255) for c in color]
        if (r, g, b) != (255, 255, 255):  # Exclude white color
            colors.add((r, g, b))

    return colors

def create_graph(folder_path):
    graph = nx.Graph()

    ply_colors = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(folder_path, filename)
            ply_colors[filename] = read_ply_color(filepath)

    ply_files = {os.path.basename(f): folder_path + f for f in ply_colors.keys()}
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


    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=4)
    plt.show()
    return graph, ply_files, edge_color_map
# if __name__ == '__main__':
#     create_graph(folder_path)