# Author: suhaot
# Date: 2023/10/19
# Description: count_overlap_surface
import os
# import plyfile
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.registration.frag_relation_graph import create_graph
import csv
from tqdm import tqdm

# 定义文件夹路径
folder_path = '../../data/FragTag3D/'
results = {}
analysis_results = {}
# 遍历每个小文件夹


def export_general_data(results, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['Key', 'Ply Count', 'Overlap Count'])
        # Write data rows
        for key, (ply_count, overlap_count) in results.items():
            writer.writerow([key, ply_count, overlap_count])


def export_analysis_data(analyze_result, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['k', 'Key', 'face_count', 'vertex_count', 'Non White Vertex Count', 'Non White Face Count', 'Max Vertex Count', 'Min Vertex Count', 'Max Face Count', 'Min Face Count'])
        # Assume analyze_result is a dictionary where each key is a string representing the directory
        for k, analysis in analyze_result.items():
            for key, result in analysis.items():
                if result is None:
                    continue
                writer.writerow([k, key, result['face_count'], result['vertex_count'], result['non_white_vertex_count'], result['non_white_face_count'],
                                 result['max_vertex_count'], result['min_vertex_count'],
                                 result['max_face_count'], result['min_face_count']])


def export_color_data(analyze_result, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['k', 'key','Color', 'Vertex Count', 'Face Count'])
        # Assume analyze_result is a dictionary where each key is a string representing the directory
        for k, analysis in analyze_result.items():
            for key, result in analysis.items():
                if result is None:
                    continue
                color_vertex_count = result.get('color_vertex_count', {})
                color_face_count = result.get('color_face_count', {})
                for color, vertex_count in color_vertex_count.items():
                    face_count = color_face_count.get(color, 0)
                    writer.writerow([k, key, color, vertex_count, face_count])


for subdir in os.listdir(folder_path):
    print('processing:' + subdir + '..............')
    for dir in os.listdir(os.path.join(folder_path, subdir)):
        print('-----------processing:' + dir + '..............')
        ply_files = [f for f in os.listdir(os.path.join(folder_path, subdir, dir)) if f.endswith('.ply')]
        ply_count = len(ply_files)
        _, _, _, overlap_count, analyze_result = create_graph(os.path.join(folder_path, subdir, dir))
        key = f"{subdir}/{dir}"
        results[key] = (ply_count, overlap_count)
        analysis_results[key] = analyze_result

print('*******************************Finished!!!!*******************************')

# Export data to CSV
export_general_data(results, 'general_data.csv')
export_analysis_data(analysis_results, 'analysis_data.csv')
export_color_data(analysis_results, 'color_data.csv')