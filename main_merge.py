import os
import numpy as np
import cv2
import re
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
interval = 5
mode = 'echo' # or CAMUS
filter_small_block_threshold = 300 // 8 # or 300 for CAMUS
segment_threshold = 101 # or 499 for CAMUS


def filter_small_block(image = None, filter_small_block_threshold = filter_small_block_threshold):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)   
    height, width = image.shape
    # min threshold
    # small block -> black
    for label in range(1, num_labels):
        # print(stats[label, cv2.CC_STAT_AREA])
        if stats[label, cv2.CC_STAT_AREA] <= filter_small_block_threshold:
            image[labels == label] = 0
    return image

def smooth_frames(frames, kernal_size = 1):
    frame_out = []
    for i in range(len(frames)):
        num_left = i - max(0,i-kernal_size)
        num_right = min(i+kernal_size+1, len(frames)) - i - 1
        real_num = min(num_left, num_right)
        current_frames = frames[max(0,i-real_num):min(i+1+real_num,len(frames))]
        frame = np.mean(current_frames, axis=0).astype(np.float32)
        #frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.uint8)
        frame_out.append(frame)
    return frame_out
def two_stage_segmentation(image_folder, image_out_folder, patient, threshold = 3000, border_area = None):
    seg_frames = []
    images_folder = os.path.join(image_folder, patient)
    #read images
    png_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    # png_files.sort(key=lambda x: int(re.sub('\D', '', x)))  # 假定文件名中包含数字，按数字排序
    png_files.sort(key=lambda x: int(re.findall(r'-?\d+', x)[0]))
    org_frames = []
    for png_file in png_files:
        file_path = os.path.join(images_folder, png_file)
        frame = cv2.imread(file_path).astype(np.uint8)
        #frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        org_frames.append(frame)
    #smooth frames
    #org_frames = smooth_frames(org_frames)
    if mode == 'CAMUS':
        border_area = remove_border_area(np.array(org_frames))

    for png_file , i in zip(png_files, range(len(png_files))):
        frame = org_frames[i]
        if mode == 'echo':
            segment_threshold = 101
            thresh_cat_frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, segment_threshold, 0)
        elif mode == 'CAMUS':
            segment_threshold = 499
            thresh_cat_frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, segment_threshold, 0)
        folder_path = '%s/%s/pure_seg' % (image_out_folder, patient)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite('%s/%s' % (folder_path, png_file), thresh_cat_frame)
        #filter small block
        thresh_cat_frame[border_area] = 255
        thresh_cat_frame = filter_small_block(thresh_cat_frame)
        folder_path = '%s/%s/seg-merge' % (image_out_folder, patient)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(f'{folder_path}/{png_file}', thresh_cat_frame)
        seg_frames.append(thresh_cat_frame)
    return org_frames, seg_frames, border_area

def remove_border_area(image_frames, threshold = 100):
    gray = np.sum(image_frames, axis=0)
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    # 应用阈值来识别黑色区域
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # 查找连通组件
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    # 获取图像的角点坐标
    height, width = gray.shape
    corner_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    component_mask = np.zeros_like(gray)
    # 检查角点是否属于同一连通组件
    component_label = None
    for x, y in corner_points:
        label = labels[y, x]
        component_mask[labels == label] = 255

    return component_mask == 255

def get_intersection_area(seg_frames, image_out_folder, patient, mode):
    if mode == 'echo':
        count_times = seg_frames.sum(axis=0)
        nums = len(seg_frames)
        #get top smallest 0.1 area as True
        intersection = count_times <= 0.2 * nums
    elif mode == 'CAMUS':
        intersection = seg_frames.sum(axis=0)
        #get top smallest 0.1 area as True
        intersection = intersection <= np.quantile(intersection, 0.01)
    
    intersection_complement_image = Image.fromarray(255 - np.uint8(intersection) * 255, mode='L')
    intersection_complement_image.save("%s/%s/intersection_complement.png" % (image_out_folder, patient))
    print("Intersection complement image saved as intersection_complement.png")

    return intersection if mode == 'CAMUS' else (count_times, intersection)

def select_k_points(image, k):
    # 提取特定连通域
    region = np.where(image, 255, 0).astype(np.uint8)

    # 应用骨架化
    skeleton = cv2.ximgproc.thinning(region)

    # 找到骨架上的所有点的坐标
    y, x = np.where(skeleton == 255)

    # 将坐标组合成点
    skeleton_points = np.column_stack((x, y))

    # 确保骨架上的点数大于或等于k
    if len(skeleton_points) >= k:
        selected_points_indices = np.linspace(0, len(skeleton_points) - 1, k, dtype=int)
        selected_points = skeleton_points[selected_points_indices]
    else:
        selected_points = skeleton_points  # 如果骨架上的点数少于k，则选择所有点

    return selected_points
def find_nearest_pixel_to_centroid(region, offsetx, offsety):
        """找到最接近质心的像素点"""
        y, x = np.where(region == 1)
        if len(x) == 0 or len(y) == 0:
            return None  # 区域内没有像素点
        centroid = (np.mean(x) + offsetx, np.mean(y) + offsety)
        nearest_pixel = min(zip(x + offsetx, y + offsety), key=lambda p: (p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        return nearest_pixel
def select_4_points(image):
        #return (x[0], y[0]) if len(x) > 0 and len(y) > 0 else None
    # 假设labels是连通域的标签矩阵，label是特定的连通域标签
    # labels, label = ...

    # 提取特定连通域
    region = np.where(image, 1, 0)
    # 找到连通域的边界框
    y, x = np.where(region == 1)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # 计算边界框中心线
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2

    # 将连通域分成四份
    quarters = [
        region[min_y:center_y, min_x:center_x],  # 左上
        region[min_y:center_y, center_x:max_x],  # 右上
        region[center_y:max_y, min_x:center_x],  # 左下
        region[center_y:max_y, center_x:max_x]   # 右下
    ]
    centers = []
    centers.append(find_nearest_pixel_to_centroid(quarters[0].copy(), min_x, min_y))  # 左上
    centers.append(find_nearest_pixel_to_centroid(quarters[1].copy(), center_x, min_y))  # 右上
    centers.append(find_nearest_pixel_to_centroid(quarters[2].copy(), min_x, center_y))  # 左下
    centers.append(find_nearest_pixel_to_centroid(quarters[3].copy(), center_x, center_y))  # 右下
    # 对每个部分找到一个代表性的中心点
    #centers = [find_center_of_mass(q.copy()) for q in quarters]
    # remove None
    centers = [center for center in centers if center is not None]
    print(f'centers: {centers}')
    return centers

def select_2_points(image, intersection_times=None):
    if mode == 'echo':
        smallest_area = intersection_times == np.min(intersection_times)
        # 提取特定连通域
        region = np.where(image, 1, 0) * smallest_area
    elif mode == 'CAMUS':
        region = np.where(image, 1, 0)
    # 找到连通域的边界框
    y, x = np.where(region == 1)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # 计算边界框中心线
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2

    # 将连通域分成四份
    quarters = [
        region[min_y:center_y, min_x:max_x],  # 上
        region[center_y:max_y, min_x:max_x],  # 下
    ]
    centers = []
    if mode == 'echo':
        centers.append(find_nearest_pixel_to_centroid(quarters[0].copy(), min_x, min_y))  # 上
    elif mode == 'CAMUS':
        centers.append(find_nearest_pixel_to_centroid(quarters[0].copy(), min_x, min_y))  # 上
        centers.append(find_nearest_pixel_to_centroid(quarters[1].copy(), min_x, center_y))  # 下
    #centers.append(find_nearest_pixel_to_centroid(quarters[1].copy(), min_x, center_y))  # 下
    # 对每个部分找到一个代表性的中心点
    #centers = [find_center_of_mass(q.copy()) for q in quarters]
    # remove None
    centers = [center for center in centers if center is not None]
    print(f'centers: {centers}')
    return centers

def erode(image, kernel_size=3, iterations=1):
    """腐蚀"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def get_center_point(image, image_out_folder, patient, intersection_times=None):
    h, w = image.shape
        
    #transfer np.array to cv2 image
    image = (image*255).astype(np.uint8)
    #image = Image.fromarray(np.uint8(image* 255), mode='L')
    if mode == 'echo':
        image = erode(image, kernel_size=3, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    mask = image > 0
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels[~mask] = 0
    # 对连通域进行排序，根据面积从大到小
    
    #get top 4 bigest area where label is not 0
    center_list = {}
    for label in range(1, num_labels):
        #if count of label is 0, continue
        if np.count_nonzero(labels == label) == 0:
            continue
        print(f'centroids[label]: {centroids[label]}')
        center_list[(centroids[label][0], centroids[label][1], label)] = (int)(np.count_nonzero(labels == label))
    center_list = sorted(center_list.items(), key=lambda x: x[1], reverse=True)
    #selected_center_list where value bigger than 300 or only 1 element
    first_center = center_list[0]
    if mode == 'echo':
        center_list = [first_center] + [center for center in center_list[1:] if center[1] > 100//4]
    elif mode == 'CAMUS':
        center_list = [first_center] + [center for center in center_list[1:] if center[1] > 800//4]
    min_center = 99999
    center_point = None
    k_center = 4
    cv2.circle(image, (int(w//2), int(0)), 1, (0, 0, 255), -1)
    for (x, y, label), _ in center_list:
        #select the right-up corner
        if abs(w//2 - x) + abs(0 - y) < min_center:
            min_center = abs(w//2 - x) + abs(0 - y)
            coords = np.argwhere(labels == label)
            #k_center = min(k_center, len(coords))
            if mode == 'echo':
                center_point = select_2_points(labels == label, intersection_times)#coords[np.random.choice(len(coords), k_center, replace=False)]
            elif mode == 'CAMUS':
                center_point = select_2_points(labels == label)#coords[np.random.choice(len(coords), k_center, replace=False)]
            center_point = np.array(center_point)
            print(f'ceter_point_shape: {center_point.shape}')
        # cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    cv2.imwrite('%s/%s/%s.png' % (image_out_folder, patient,'with_rect'), image)
    return center_point

def find_first_white_pixel_distance(point, angle, binary_image):
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    height, width = binary_image.shape
    
    # 计算角度的增量
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)  # 注意取负号，因为y轴向下是增加的方向

    # 初始化起始点坐标
    x = point[0]
    y = point[1]

    # 逐步移动起始点，直到遇到白色像素或超出图像范围
    while 0 <= x < width and 0 <= y < height:
        # 检查当前像素是否为白色（即值为1）
        if binary_image[int(y), int(x)] > 0:
            # 计算起始点到白色像素的距离
            distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            return distance, int(x), int(y)
        
        # 移动起始点到下一个像素
        x += dx
        y += dy
    # 不应该返回None，而是这个中心点到边缘的距离
    return np.sqrt((x - point[0])**2 + (y - point[1])**2), int(x), int(y)
def get_all_angle_distance(points, input_image_path, image_out_folder, patient, points_and_angles = None):
    dis = list()
    # 只需要读图，画线，得到距离吧
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    image0 = image.copy()
    if points_and_angles is None:
        points_and_angles = {}
        for point in points:
            selected_numbers = [j for j in range(0, 360) if j % 5 == 0]
            points_and_angles[point] = selected_numbers
    for point in points:
        selected_numbers = points_and_angles[point]
        sample_num = len(selected_numbers)
        for j1 in selected_numbers: 
            for j in range(j1, j1 + 5, 1):
            # 这里为什么会返回none呢
                result = find_first_white_pixel_distance(point, j, image0)
                if result is not None:
                    first_white_pixel_distance, dst_x, dst_y = result
                else:
                    print("This degree can not find a suitable edge.")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.circle(image, (dst_x, dst_y), 1, (0, 0, 255), -1)
                cv2.circle(image, point, 1, (0, 0, 255), -1)
                cv2.line(image, (dst_x, dst_y), point, (0, 0, 255))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                dis.append(first_white_pixel_distance)
    print(f'len(dis): {len(dis)}')
    split_parts = input_image_path.split('/')
    last_part = split_parts[-1]
    
    folder_path = '%s/%s/seg-with-line' % (image_out_folder, patient)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    cv2.imwrite('%s/%s/seg-with-line/%s' % (image_out_folder, patient, last_part), image)
    return dis
def sort_key(s):
            return int(s.split('.')[0].split('_')[-1])

def get_change_array(image_folder, image_out_folder, patient, num_sample = 48, mask_border = None):
    org_frames, seg_frames, border_area = two_stage_segmentation(image_folder, image_out_folder, patient, border_area = mask_border)
    
    # find mix area
    seg_frames = np.array(seg_frames)
    print(f'seg_frames.shape: {seg_frames.shape}')
    #intersection = np.ones_like(seg_frames) * 255
    if mode == 'echo':
        intersection_times, intersection = get_intersection_area(seg_frames, image_out_folder, patient, mode)
    elif mode == 'CAMUS':
        intersection = get_intersection_area(seg_frames, image_out_folder, patient, mode)
    
    # Add mask in order to find the white edge
    
    intersection = intersection * (1 - border_area)
    
    if mode == 'echo':
        cur_dst_points = get_center_point(intersection, image_out_folder, patient, intersection_times)
    else:
        cur_dst_points = get_center_point(intersection, image_out_folder, patient)

    res = []
    input_image_path = '%s/%s/seg-merge' % (image_out_folder, patient)

    file_name = os.listdir(input_image_path)
    file_name = sorted(file_name, key=sort_key)
    possible_numbers = [j for j in range(0, 360) if j % interval == 0]
    points_and_angles = {}
    cur_dst_points = [(int(point[0]), int(point[1])) for point in cur_dst_points]
    num_points = len(cur_dst_points)
    for cur_dst_point in cur_dst_points:
        selected_numbers = random.sample(possible_numbers, num_sample)
        points_and_angles[cur_dst_point] = selected_numbers
    for j in file_name:
        cur_path = os.path.join(input_image_path, j)
        dis = get_all_angle_distance(cur_dst_points, cur_path, image_out_folder, patient, points_and_angles)
        res.append(dis)
    # 连续列表作差
    original_list = res
    original_array = np.array(original_list) #(num_frames, 360)
    change_array = original_array[1:, :] - original_array[:-1, :]
    return change_array, num_sample, num_points

def make_bigger_interval(change_array, nums = 2):
    
    #make every value equals to the sum of previous nums values, shape (num_frames, 360, k)
    change_array = np.pad(change_array, ((nums-1, 0), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
    change_array_2 = np.zeros_like(change_array)
    for i in range(nums-1, change_array.shape[0]):
        change_array_2[i, :] = np.sum(change_array[i-nums+1:i+1, :], axis=0)
    return change_array_2[nums-1:, :]

def get_boarder_mask(image_folder, threshold = 30, times = 300):
    png_files = []
    max_value = None
    count = 0
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                if count >= times:
                    break
                count += 1
                img = cv2.imread(os.path.join(root, file)).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if max_value is None:
                    max_value = img
                else:
                    max_value = np.maximum(max_value, img)
    return max_value < threshold
def Get_ED_and_ES(image_folder, image_out_folder, num_sample = 48, num_points = 2):

    ultra_final = []
    totoal_patients = os.listdir(image_folder)
    totoal_patients = sorted(totoal_patients)
    #read cashed change array
    cashed_change_array = {}
    if mode == 'echo':
        boarder_mask = get_boarder_mask(image_folder)
    if os.path.exists(cashed_change_array_csv):
        with open(cashed_change_array_csv, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                cashed_change_array[row[0]] = (np.array(row[4:]).astype(np.float32), int(row[1]), int(row[2]), int(row[3]))
    for patient , i in zip(totoal_patients, range(len(totoal_patients))):
        if patient in cashed_change_array:
            change_array, num_sample, num_points, num_frames= cashed_change_array[patient]
            #set as np.array
            change_array = np.array(change_array)
        else:
            if mode == 'echo':
                change_array, num_sample, num_points = get_change_array(image_folder, image_out_folder, patient, num_sample, boarder_mask)
            else:
                change_array, num_sample, num_points = get_change_array(image_folder, image_out_folder, patient, num_sample)
            num_frames = change_array.shape[0]
            with open(cashed_change_array_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                change_array = change_array.reshape(-1)
                your_list = change_array.tolist()
                your_list.insert(0, num_frames)
                your_list.insert(0, num_points)
                your_list.insert(0, num_sample)
                your_list.insert(0, patient)
                writer.writerow(your_list)
        median_interval = 5
        possibility = 0.2 * median_interval
        change_array = np.reshape(change_array, (num_frames, num_sample * num_points, -1))
        if mode == 'echo':
            major_changed_second_index = abs(change_array) > 10
            major_changed_second_index = np.any(major_changed_second_index, axis=(0,2))
            change_array[:,major_changed_second_index,:] = 0
            change_array[abs(change_array) > 5//2] = 0
        change_array[change_array > 0] = 1
        change_array[change_array < 0] = -1
        change_array = np.sum(change_array, axis=2)
        change_array[abs(change_array) < possibility] = 0
        change_array[change_array <= -possibility] = -1
        change_array[change_array >= possibility] = 1
        change_array = np.sum(change_array, axis=1)/(np.sum(change_array != 0, axis=1) + 1e-6)
        # transfer to list
        res = change_array.tolist()
                

        print(res)
        # 你想添加到每行第一列的字符串
        string_value = patient
        ############################################## 这里是我之前保存中间结果的地方 ##############################
        # 打开一个新的CSV文件
        with open(output_list_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            length = len(res) + 1
            your_list = res  # 这里应该是你生成列表的代码

            # 在列表前加上字符串值
            
            your_list.insert(0, string_value)
            your_list.insert(1, length)
            # 将列表（现在包含字符串）写入CSV文件
            writer.writerow(your_list)
    return 1 #ultra_final

if __name__ == '__main__':
    image_folder = '/data/bzy/data/zybu/dataset/EchoNet-Dynamic/EchoNet'
    image_out_folder = '/data/bzy/MaxMin-Room-Detection-main/Seg_EchoNet_0319'
    # image_folder = '/data/bzy/CAMUS_PNG_Augted_contain_inverse'
    # image_out_folder = '/data/bzy/MaxMin-Room-Detection-main/Seg_CAMUS_0319'
    cashed_change_array_csv = 'cashed_change_array_echo_0319.csv'
    output_list_csv = 'output_list_echo_0319.csv'

    ultra_final = Get_ED_and_ES(image_folder, image_out_folder)