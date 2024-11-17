import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import random
import shutil
from itertools import combinations
from shapely.geometry import LineString, Point
def pair_files(input_folder, target_file, output_folder):
    # 获取所有文件的列表
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # 检查目标文件是否存在于文件列表中
    if target_file not in files:
        print(f"Error: {target_file} does not exist in {input_folder}")
        return

    # 从文件列表中移除目标文件
    files.remove(target_file)

    # 遍历所有文件，与目标文件两两组合
    for file in files:
        # 创建新文件夹，文件夹名称为两个文件名组合
        folder_name = f"{target_file}&{file}"
        folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 将目标文件和当前文件复制到新文件夹中
        shutil.copy(os.path.join(input_folder, target_file), folder_path)
        shutil.copy(os.path.join(input_folder, file), folder_path)

def excel_to_pcd(folder_path, output_folder):
    """
    将指定文件夹中的 Excel 文件转换为 PCD 文件。

    参数：
    - folder_path: str，包含 Excel 文件的源文件夹路径
    - output_folder: str，保存 PCD 文件的目标文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中 Excel 文件
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # 遍历每个 Excel 文件并将其转换为 PCD 文件
    for excel_file in excel_files:
        # 加载 Excel 文件
        file_path = os.path.join(folder_path, excel_file)
        # 使用 openpyxl 加载 .xlsx 文件
        xls = pd.ExcelFile(file_path, engine='openpyxl')

        try:
            data = xls.parse('坐标数据', skiprows=2)  # 跳过前两行
        except ValueError:
            print(f"在文件 {excel_file} 中找不到工作表 '坐标数据'，跳过该文件")
            continue

        # 提取坐标数据 (X, Y, Z)
        try:
            points = data.iloc[:, [0, 1, 2]].values
            points = points.astype(float)  # 将数据转换为浮点数类型
        except KeyError:
            print(f"在文件 {excel_file} 中找不到所需的列，跳过该文件")
            continue

        # 创建点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # 定义输出文件名和路径
        output_filename = os.path.splitext(excel_file)[0] + '.pcd'
        output_path = os.path.join(output_folder, output_filename)

        # 保存点云为 .pcd 文件
        o3d.io.write_point_cloud(output_path, point_cloud)
        print(f"点云文件已保存为: {output_path}")

    ### 拟合曲线
def process_point_clouds(pcd_directory, rotation_angle):


    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 生成旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    # 显示图形的尺寸
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # 用于存储拟合的曲线
    curves = []

    # 读取点云文件并拟合曲线
    for idx, filename in enumerate(os.listdir(pcd_directory)):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(pcd_directory, filename)
            
            # 读取 PCD 文件
            pcd = o3d.io.read_point_cloud(pcd_path)

            # 旋转点云
            pcd.rotate(rotation_matrix, center=(0, 0, 0))
            
            points = np.asarray(pcd.points)
            
            # 提取二维点 (x, y)
            x = points[:, 0]
            y = points[:, 1]

            # 绘制原始点云
            ax.scatter(x, y, color=f'C{idx}', alpha=0.5, s=10)

            # 使用 splprep 和 splev 进行二维曲线拟合
            tck, u = splprep([x, y], s=0)  
            u_fine = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_fine, tck)
            
            # 存储拟合曲线
            curves.append((x_new, y_new))
            
            # 绘制拟合曲线
            ax.plot(x_new, y_new, label=f'{filename}', color=f'C{idx}')
           
    
    ax.set_aspect('equal')
    plt.axis('square')
    ax.legend()
    plt.show()
    
 ###拟合面积分析
def analyze_point_cloud_area(pcd_directory, rotation_angle):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
   

    # 生成旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

   

    # 增大显示图形的尺寸
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # 用于存储拟合的曲线
    curves = []
    # 用于存储所有区域的面积和
    total_area = 0

    # 读取点云文件并拟合曲线
    for idx, filename in enumerate(os.listdir(pcd_directory)):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(pcd_directory, filename)

            # 读取 PCD 文件
            pcd = o3d.io.read_point_cloud(pcd_path)

            # 旋转点云
            pcd.rotate(rotation_matrix, center=(0, 0, 0))

            points = np.asarray(pcd.points)

            # 提取二维点 (x, y)
            x = points[:, 0]
            y = points[:, 1]

            # 绘制原始点云
            ax.scatter(x, y, color=f'C{idx}', alpha=0.5, s=10)

            # 使用 splprep 和 splev 进行二维曲线拟合
            tck, u = splprep([x, y], s=0)  
            u_fine = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_fine, tck)

            # 存储拟合曲线
            curves.append(np.column_stack((x_new, y_new)))

            # 绘制拟合曲线
            ax.plot(x_new, y_new, label=f'{filename}', color=f'C{idx}')

    # 查找并处理曲线的交叉点
    if len(curves) >= 2:
        intersections = []

        # 查找所有曲线之间的交点
        for line1_points, line2_points in combinations(curves, 2):
            line1 = LineString(line1_points)
            line2 = LineString(line2_points)
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    intersections.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    intersections.extend(list(intersection.geoms))

            # 确保交点按曲线顺序排列
            intersections = sorted(intersections, key=lambda point: line1.project(point))

            # 分段计算封闭区域的面积
            for k in range(len(intersections) - 1):
                start_point = intersections[k]
                end_point = intersections[k + 1]

                # 提取线段之间的部分
                segment1_coords = []
                segment2_coords = []

                for point in line1_points:
                    if line1.project(Point(point)) >= line1.project(start_point) and line1.project(Point(point)) <= line1.project(end_point):
                        segment1_coords.append(point)

                for point in line2_points:
                    if line2.project(Point(point)) >= line2.project(start_point) and line2.project(Point(point)) <= line2.project(end_point):
                        segment2_coords.append(point)

                # 构建封闭区域并计算面积
                if len(segment1_coords) > 1 and len(segment2_coords) > 1:
                    polygon_points = np.vstack((segment1_coords, segment2_coords[::-1]))
                    if len(polygon_points) >= 3:
                        x, y = polygon_points[:, 0], polygon_points[:, 1]
                        ax.fill(x, y, color='black', alpha=0.3)
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        total_area += area
                        if area >= 0.001:
                            centroid_x = np.mean(x)
                            centroid_y = np.mean(y)
                            ax.text(centroid_x, centroid_y, f'{area:.4f}', color='black', fontsize=10)

            # 添加连接起点和终点的线段
            start_point1 = line1_points[0]
            start_point2 = line2_points[0]
            end_point1 = line1_points[-1]
            end_point2 = line2_points[-1]

            # 绘制连接起点和终点的线段
            ax.plot([start_point1[0], start_point2[0]], [start_point1[1], start_point2[1]], linestyle=' ')
            ax.plot([end_point1[0], end_point2[0]], [end_point1[1], end_point2[1]], linestyle=' ')

    if len(curves) >= 2:
        intersections = []

        # Existing code to find intersections
        # ...

        # For each pair of curves
        for line1_points, line2_points in combinations(curves, 2):
            line1 = LineString(line1_points)
            line2 = LineString(line2_points)
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    intersections.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    intersections.extend(list(intersection.geoms))

            # Ensure intersections are sorted along line1
            intersections = sorted(intersections, key=lambda point: line1.project(point))

            # Existing code to process areas between intersections
            # ...

            # Compute cumulative distances along the lines
            line1_dists = np.array([line1.project(Point(pt)) for pt in line1_points])
            line2_dists = np.array([line2.project(Point(pt)) for pt in line2_points])

            if intersections:
                # Part 1: Area from start points to the first intersection
                first_intersection = intersections[0]
                first_inter_proj_line1 = line1.project(first_intersection)
                first_inter_proj_line2 = line2.project(first_intersection)

                indices_line1 = np.where(line1_dists <= first_inter_proj_line1)[0]
                indices_line2 = np.where(line2_dists <= first_inter_proj_line2)[0]

                segment1_coords = line1_points[indices_line1]
                segment2_coords = line2_points[indices_line2]

                if len(segment1_coords) > 1 and len(segment2_coords) > 1:
                    # Combine the segments and the line connecting the starting points
                    polygon_points = np.vstack((segment1_coords, segment2_coords[::-1]))
                    if len(polygon_points) >= 3:
                        x, y = polygon_points[:, 0], polygon_points[:, 1]
                        ax.fill(x, y, color='black', alpha=0.3)
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        total_area += area
                        if area >= 0.001:
                            centroid_x = np.mean(x)
                            centroid_y = np.mean(y)
                            ax.text(centroid_x, centroid_y, f'{area:.4f}', color='black', fontsize=10)

                # Part 2: Area from last intersection to end points
                last_intersection = intersections[-1]
                last_inter_proj_line1 = line1.project(last_intersection)
                last_inter_proj_line2 = line2.project(last_intersection)

                indices_line1 = np.where(line1_dists >= last_inter_proj_line1)[0]
                indices_line2 = np.where(line2_dists >= last_inter_proj_line2)[0]

                segment1_coords = line1_points[indices_line1]
                segment2_coords = line2_points[indices_line2]

                if len(segment1_coords) > 1 and len(segment2_coords) > 1:
                    # Combine the segments and the line connecting the end points
                    polygon_points = np.vstack((segment1_coords, segment2_coords[::-1]))
                    if len(polygon_points) >= 3:
                        x, y = polygon_points[:, 0], polygon_points[:, 1]
                        ax.fill(x, y, color='black', alpha=0.3)
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        total_area += area
                        if area >= 0.001:
                            centroid_x = np.mean(x)
                            centroid_y = np.mean(y)
                            ax.text(centroid_x, centroid_y, f'{area:.4f}', color='black', fontsize=10)

    # 设置 xy 比例相同
    ax.set_aspect('equal')
    plt.axis('square')

    # 显示总面积信息
    plt.text(0.02, 0.98, f'Deformation: {total_area:.4f}', transform=ax.transAxes, fontsize=10, color='black', verticalalignment='top')

    ax.legend()
    plt.show()
"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 生成旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    # 增大显示图形的尺寸
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # 用于存储拟合的曲线
    curves = []
    # 用于存储所有区域的面积和
    total_area = 0

    # 读取点云文件并拟合曲线
    for idx, filename in enumerate(os.listdir(pcd_directory)):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(pcd_directory, filename)

            # 读取 PCD 文件
            pcd = o3d.io.read_point_cloud(pcd_path)

            # 旋转点云
            pcd.rotate(rotation_matrix, center=(0, 0, 0))

            points = np.asarray(pcd.points)

            # 提取二维点 (x, y)
            x = points[:, 0]
            y = points[:, 1]

            # 绘制原始点云
            ax.scatter(x, y, color=f'C{idx}', alpha=0.5, s=10)

            # 使用 splprep 和 splev 进行二维曲线拟合
            tck, u = splprep([x, y], s=0)
            u_fine = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_fine, tck)

            # 存储拟合曲线
            curves.append(np.column_stack((x_new, y_new)))

            # 绘制拟合曲线
            ax.plot(x_new, y_new, label=f'{filename}', color=f'C{idx}')

    # 查找并处理曲线的交叉点
    if len(curves) >= 2:
        intersections = []

        # 查找所有曲线之间的交点
        for line1_points, line2_points in combinations(curves, 2):
            line1 = LineString(line1_points)
            line2 = LineString(line2_points)
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    intersections.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    intersections.extend(list(intersection.geoms))

            # 确保交点按曲线顺序排列
            intersections = sorted(intersections, key=lambda point: line1.project(point))

            # 分段计算封闭区域的面积
            for k in range(len(intersections) - 1):
                start_point = intersections[k]
                end_point = intersections[k + 1]

                # 提取线段之间的部分
                segment1_coords = []
                segment2_coords = []

                for point in line1_points:
                    if line1.project(Point(point)) >= line1.project(start_point) and line1.project(Point(point)) <= line1.project(end_point):
                        segment1_coords.append(point)

                for point in line2_points:
                    if line2.project(Point(point)) >= line2.project(start_point) and line2.project(Point(point)) <= line2.project(end_point):
                        segment2_coords.append(point)

                # 构建封闭区域并计算面积
                if len(segment1_coords) > 1 and len(segment2_coords) > 1:
                    polygon_points = np.vstack((segment1_coords, segment2_coords[::-1]))
                    if len(polygon_points) >= 3:
                        x, y = polygon_points[:, 0], polygon_points[:, 1]
                        ax.fill(x, y, color='black', alpha=0.3)
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        total_area += area
                        if area >= 0.001:
                            centroid_x = np.mean(x)
                            centroid_y = np.mean(y)
                            ax.text(centroid_x, centroid_y, f'{area:.4f}', color='black', fontsize=10)

    # 设置 xy 比例相同
    ax.set_aspect('equal')
    plt.axis('square')

    # 显示总面积信息
    plt.text(0.02, 0.98, f'Deformation: {total_area:.4f}', transform=ax.transAxes, fontsize=10, color='black', verticalalignment='top')

    ax.legend()
    plt.show()
    """
# to 2d
def oint_clouds_to2d(input_dir, output_dir):
    def process_point_cloud(file_path, output_path, rotation_matrix=None):
        # 读取点云数据
        pcd = o3d.io.read_point_cloud(file_path)

        # 转换为numpy数组
        points = np.asarray(pcd.points)

        # 计算点云的中心
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        if rotation_matrix is None:
            # 进行PCA分析，计算主成分方向
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # 获取主平面的法向量
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

            # 目标法向量
            target_normal = np.array([0, 0, 1])

            # 计算旋转轴和旋转角度
            v = np.cross(normal_vector, target_normal)
            c = np.dot(normal_vector, target_normal)
            s = np.linalg.norm(v)

            # 构造旋转矩阵
            I = np.eye(3)
            if s > 1e-6:  # 如果旋转轴不为0，构造旋转矩阵
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = I + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
            else:
                R = I  # 如果已经对齐，则使用单位矩阵
        else:
            R = rotation_matrix

        # 对点云应用旋转变换
        rotated_points = np.dot(centered_points, R.T)

        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points + centroid)

        # 保存转换后的点云
        o3d.io.write_point_cloud(output_path, rotated_pcd)

        return R  # 返回计算得到的旋转矩阵

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rotation_matrix = None

    for idx, file_name in enumerate(os.listdir(input_dir)):
        if file_name.endswith('.pcd'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # 对第一个点云文件计算旋转矩阵，后续直接复用
            if idx == 0:
                rotation_matrix = process_point_cloud(input_path, output_path)
            else:
                process_point_cloud(input_path, output_path, rotation_matrix)

            print(f"处理完成： {file_name}")

    
    point_clouds = []
    file_names = []  # 用于存储原始的文件名以便后续使用
    for filename in os.listdir(output_dir):
        if filename.endswith(".pcd"):
            full_path = os.path.join(output_dir, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            point_clouds.append(pcd)
            file_names.append(filename)  # 存储文件名

    # 为每个点云设置随机颜色，便于可视化
    for pcd in point_clouds:
        color = [random.random(), random.random(), random.random()]
        pcd.paint_uniform_color(color)

    

    # 配准门限值
    threshold_coarse = 5.0   # 粗配准门限值
    threshold_fine = 0.1     # 精细配准门限值

    # 初始化变换矩阵列表，第一个点云的变换矩阵为单位矩阵
    transformations = [np.eye(4)]

    # 所有点云都和第一个点云进行配准
    target = point_clouds[0]
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]

        # 使用粗配准
        reg_p2p_coarse = o3d.pipelines.registration.registration_icp(
            source, target, threshold_coarse, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # 使用精细配准
        initial_transform = reg_p2p_coarse.transformation
        reg_p2p_fine = o3d.pipelines.registration.registration_icp(
            source, target, threshold_fine, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # 累积变换矩阵
        current_transform = reg_p2p_fine.transformation
        transformations.append(current_transform)

    # 应用累积变换并保存对齐后的点云，使用原始文件名
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(0, len(point_clouds)):
        point_clouds[i].transform(transformations[i])
        output_filename = os.path.join(output_dir, file_names[i])
        o3d.io.write_point_cloud(output_filename, point_clouds[i])
        #print(f"平面转换后点云保存为 {output_filename}")
    print("平面转换完成！")

   


#icp
def point_cloud_registration(input_folder, output_folder):
    # 加载输入文件夹的所有点云文件
    point_clouds = []
    file_names = []  # 用于存储原始的文件名以便后续使用
    for filename in os.listdir(input_folder):
        if filename.endswith(".pcd"):
            full_path = os.path.join(input_folder, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            point_clouds.append(pcd)
            file_names.append(filename)  # 存储文件名

    # 配准门限值
    threshold_coarse = 5.0   # 粗配准门限值
    threshold_fine = 0.1     # 精细配准门限值

    # 初始化变换矩阵列表，第一个点云的变换矩阵为单位矩阵
    transformations = [np.eye(4)]

    # 所有点云都和第一个点云进行配准
    target = point_clouds[0]
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]

        # 使用粗配准
        reg_p2p_coarse = o3d.pipelines.registration.registration_icp(
            source, target, threshold_coarse, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # 使用精细配准
        initial_transform = reg_p2p_coarse.transformation
        reg_p2p_fine = o3d.pipelines.registration.registration_icp(
            source, target, threshold_fine, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # 累积变换矩阵
        current_transform = reg_p2p_fine.transformation
        transformations.append(current_transform)

    # 应用累积变换并保存对齐后的点云，使用原始文件名
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(0, len(point_clouds)):
        point_clouds[i].transform(transformations[i])
        output_filename = os.path.join(output_folder, file_names[i])
        o3d.io.write_point_cloud(output_filename, point_clouds[i])
        print(f"对齐的点云已保存到 {output_filename}")


#删除平面外点
def delete_point_1(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 .pcd 文件的列表
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]

    # 批量处理每个点云文件
    for pcd_file in pcd_files:
        try:
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, pcd_file)
            
            # 读取点云
            pcd = o3d.io.read_point_cloud(input_path)
            
            # 检查点云的点数是否足够
            if len(pcd.points) < 3:
                print(f"文件 {pcd_file} 的点数不足以拟合平面，跳过处理。")
                continue

            # 使用 RANSAC 进行平面拟合
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            [a, b, c, d] = plane_model
            #print(f"{pcd_file} 平面方程为: {a}x + {b}y + {c}z + {d} = 0")

            # 提取属于平面内的点
            inlier_cloud = pcd.select_by_index(inliers)

            # 构建输出文件路径
            output_path = os.path.join(output_folder, f"{pcd_file}")

            # 保存处理后的点云
            o3d.io.write_point_cloud(output_path, inlier_cloud)
            print(f"处理完成：  {pcd_file}")

        except Exception as e:
            print(f"处理文件 {pcd_file} 时出错: {e}")

    print("噪点1删除成功！")

def delete_point_2(input_folder, output_folder, distance_threshold=2.5):
    """
    根据距离阈值过滤输入文件夹中的点云文件中的噪点，
    并将过滤后的点云保存到输出文件夹中。
    
    参数：
        input_folder (str): 包含输入.pcd文件的文件夹路径。
        output_folder (str): 保存过滤后.pcd文件的文件夹路径。
        distance_threshold (float): 两点之间的最大允许距离。
    """
    # 如果输出文件夹不存在，则创建它
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有.pcd文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".pcd"):
            # 读取点云文件
            pcd_path = os.path.join(input_folder, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)

            # 获取点云中的所有点
            points = np.asarray(pcd.points)
            if len(points) < 2:
                print(f"跳过（点数不足）：{filename}")
                continue

            # 逐点遍历并删除距离过大的点
            filtered_points = [points[0]]  # 从第一个点开始
            removed_points_count = 0
            for i in range(1, len(points)):
                distance = np.linalg.norm(points[i] - filtered_points[-1])
                if distance <= distance_threshold:
                    filtered_points.append(points[i])
                else:
                    removed_points_count += 1

            # 创建新的点云对象
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

            # 保存过滤后的点云
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_point_cloud(output_path, filtered_pcd)

            print(f"处理完成：{filename}, 移除的点数：{removed_points_count}")

    print("噪点2删除成功！")

#可视化函数
def visualize_multiple_pcd(folder_path):
    # 获取文件夹中的所有 .pcd 文件
    pcd_files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]
    
    if not pcd_files:
        print("文件夹中未找到任何 .pcd 文件。")
        return

    # 创建一个列表来存储点云对象
    point_clouds = []

    # 遍历文件夹并加载点云
    for pcd_file in pcd_files:
        file_path = os.path.join(folder_path, pcd_file)
        try:
            point_cloud = o3d.io.read_point_cloud(file_path)
            if point_cloud.is_empty():
                print(f"点云文件 {pcd_file} 是空的，跳过。")
                continue
            # 为每个点云分配随机颜色
            color = np.random.rand(3)
            point_cloud.paint_uniform_color(color)
            point_clouds.append((pcd_file, point_cloud))
            print(f"成功加载点云文件: {pcd_file}")
        except Exception as e:
            print(f"加载点云文件 {pcd_file} 时出错，错误: {e}")

    if not point_clouds:
        print("没有有效的点云。")
        return

    # 可视化所有点云
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name="所有点云查看器，XOY 平面视图",
                          width=800, height=800, left=50, top=50)

        for _, point_cloud in point_clouds:
            vis.add_geometry(point_cloud)

        vis.poll_events()
        vis.update_renderer()
        view_control = vis.get_view_control()
        view_control.set_front([0.0, -1.0, 0.0])  # 相机朝向 XOZ 平面
        view_control.set_up([0.0, 0.0, 1.0])    # 向上方向为 Z 轴

        vis.run()
    except Exception as e:
        print(f"渲染过程中出现错误: {e}")
    finally:
        vis.destroy_window()

    # 单独可视化每个点云
    for pcd_file, point_cloud in point_clouds:
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window(window_name=f"{pcd_file} 查看器，XOY 平面视图",
                              width=800, height=800, left=50, top=50)

            vis.add_geometry(point_cloud)

            vis.poll_events()
            vis.update_renderer()
            view_control = vis.get_view_control()
            view_control.set_front([0.0, -1.0, 0.0])  # 相机朝向 XOZ 平面
            view_control.set_up([0.0, 0.0, 1.0])    # 向上方向为 Z 轴

            vis.run()
        except Exception as e:
            print(f"渲染点云 {pcd_file} 时出现错误: {e}")
        finally:
            vis.destroy_window()
