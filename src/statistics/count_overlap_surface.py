# Author: suhaot
# Date: 2023/10/19
# Description: count_overlap_surface
import os
# import plyfile
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.registration.frag_relation_graph import create_graph
import csv

# 定义文件夹路径
folder_path = '../../data/thingi10k/'
results = {}
# 遍历每个小文件夹
for subdir in os.listdir(folder_path):
    for dir in os.listdir(os.path.join(folder_path, subdir)):
        ply_files = [f for f in os.listdir(os.path.join(folder_path, subdir, dir)) if f.endswith('.ply')]
        ply_count = len(ply_files)
        _, _, _, overlap_count = create_graph(os.path.join(folder_path, subdir, dir))
        key = f"{subdir}/{dir}"
        results[key] = (ply_count, overlap_count)

# 定义CSV文件的路径
csv_file_path = './results.csv'

# 打开文件并创建一个CSV写入器对象
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入标题行
    writer.writerow(['Group', 'PLY Count', 'Overlap Count'])

    # 遍历字典并写入每行数据
    for key, value in results.items():
        writer.writerow([key, value[0], value[1]])