import open3d as o3d
import os
import networkx as nx
import matplotlib.pyplot as plt
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
    edge_colors = []  # For storing colors of the edges

    for filename in os.listdir(folder_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(folder_path, filename)
            ply_colors[filename] = read_ply_color(filepath)

    # ply_files = list(ply_colors.keys())
    ply_files = {os.path.basename(f): folder_path + f for f in ply_colors.keys()}
    for i, file1 in enumerate(ply_files):
        for j, file2 in enumerate(ply_files):
            if i >= j:
                continue
            common_colors = ply_colors[file1].intersection(ply_colors[file2])
            for color in common_colors:
                graph.add_edge(file1, file2, color=color)
                edge_colors.append(color)

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, edge_color=edge_colors, width=4)
    plt.show()
    return graph, ply_files
# if __name__ == '__main__':
#     create_graph(folder_path)