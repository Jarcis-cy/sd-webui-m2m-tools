import cv2
import os
import re
import shutil
from tqdm import tqdm
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import json
from moviepy.editor import concatenate_videoclips, VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import subprocess


def split_frames(project_path, movie_path, aim_fps=None):
    # Ensure the project directory exists
    os.makedirs(project_path, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(movie_path)

    # Get the original number of frames and FPS in the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # If aim_fps is not provided, use the video's original FPS
    if aim_fps is None:
        aim_fps = original_fps

    # Calculate the frame skip (round to nearest integer)
    frame_skip = round(original_fps / aim_fps)

    # Calculate the total number of frames
    total_frames = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip)

    # Calculate the number of digits needed for the frame filenames
    num_digits = max(5, len(str(total_frames)))

    # Create the "frame" directory
    frame_dir = os.path.join(project_path, "frame")
    os.makedirs(frame_dir, exist_ok=True)

    # Iterate over each frame in the video
    frame_count = 0
    saved_frames = 0
    pbar = tqdm(total=total_frames, desc="Splitting frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save every nth frame, where n is the frame_skip
        if frame_count % frame_skip == 0:
            # Generate the filename for the frame
            filename = f"{saved_frames + 1:0{num_digits}d}.png"
            frame_path = os.path.join(frame_dir, filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
            pbar.update(1)

        frame_count += 1

    pbar.close()

    # Release the video file
    cap.release()

    if not os.path.exists(os.path.join(project_path, "Out")):
        os.makedirs(os.path.join(project_path, "Out"))

    return "Done"


def detach_img(project_path, calcmode):
    out_folder = os.path.join(project_path, "Out")
    mask_folder = os.path.join(project_path, "Mask")
    human_folder = os.path.join(project_path, "Human")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    if not os.path.exists(human_folder):
        os.makedirs(human_folder)

    for file in os.listdir(out_folder):
        if file.endswith('.png'):
            # 通过正则表达式来解析文件名，提取出关键信息
            match = re.match(r'(\d+)_' + calcmode + '_(mask|output).png', file)
            if match:
                num, file_type = match.groups()

                new_filename = f'{num}.png'

                # 根据文件类型将文件复制到相应的文件夹，并重命名
                if file_type == 'mask':
                    shutil.copy(os.path.join(out_folder, file),
                                os.path.join(out_folder, mask_folder, new_filename))
                elif file_type == 'output':
                    shutil.copy(os.path.join(out_folder, file),
                                os.path.join(out_folder, human_folder, new_filename))
    return "Done"


def reconfiguration(project_path, width, width_cb, movie_path, smooth_factor=0.9):
    video_frame_folder = os.path.join(project_path, "video_frame")
    video_mask_folder = os.path.join(project_path, "video_mask")
    if not os.path.exists(video_frame_folder):
        os.makedirs(video_frame_folder)
    if not os.path.exists(video_mask_folder):
        os.makedirs(video_mask_folder)
    if not os.path.exists(os.path.join(project_path, "img2img_key")):
        os.makedirs(os.path.join(project_path, "img2img_key"))

    mask_folder = os.path.join(project_path, "Mask")
    frame_folder = os.path.join(project_path, "frame")
    if not os.path.exists(mask_folder):
        return "请执行步骤2，或者创建Mask文件夹，并将蒙版文件放置其中"
    if not os.path.exists(frame_folder):
        return "请执行步骤1，或者创建frame文件夹，并将源视频帧放置其中"
    cap = cv2.VideoCapture(movie_path)
    owidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    box_width = 0
    if width_cb:
        if width > owidth:
            return "输入的宽度请小于原视频宽度"
        # The width and height of the box
        if width % 2 != 0:
            box_width = width + 1
        else:
            box_width = width
    else:
        width = 0
        for filename in os.listdir(mask_folder):
            if filename.endswith('.png'):
                img = Image.open(os.path.join(mask_folder, filename))
                white_pixels = np.where(np.array(img))
                tmp_width = max(white_pixels[1]) - min(white_pixels[1])
                if tmp_width > width:
                    width = tmp_width
        if width % 2 != 0:
            box_width = width + 1
        else:
            box_width = width
    box_height = height

    # A dictionary to store the cropping information for each file
    crop_info = {}
    # 在循环外部初始化prev_center为0
    last_center_x = 0
    for filename in tqdm(sorted(os.listdir(mask_folder)), desc="处理蒙版"):
        if filename.endswith('.png'):
            # 加载图像
            img = Image.open(os.path.join(mask_folder, filename))
            img_array = np.array(img)

        # 计算图像的中心
        white_pixels = np.where(img_array)
        center_x = np.mean(white_pixels[1])
        center_y = np.mean(white_pixels[0])

        # 平滑中心位置的变化
        if last_center_x > 0:
            center_x = (1 - smooth_factor) * center_x + smooth_factor * last_center_x
        last_center_x = center_x
        # 从中心创建指定大小的方框
        left = int(center_x) - box_width // 2
        right = int(center_x) + box_width // 2
        if width % 2 != 0:
            right = right - 1
        # 裁切长方形区域
        img_cropped = img.crop((left, 0, right, box_height))
        # 保存裁切后的图像
        img_cropped.save(os.path.join(video_mask_folder, filename))

        # 记录裁剪信息
        crop_info[filename] = {
            'center_coordinates': (int(center_x), int(center_y)),  # 记录平滑后的整型中心坐标
            'crop_x': int(center_x),
        }

    # 保存裁剪信息为json文件
    with open(os.path.join(project_path, 'crop_info.json'), 'w') as f:
        json.dump(crop_info, f)

    def process_image(args):
        filename, crop_info = args
        img = Image.open(f'{frame_folder}/{filename}')
        yellow_box_x = crop_info.get(filename, {}).get('crop_x', 0)
        if width % 2 != 0:
            img_cropped = img.crop(
                (yellow_box_x - (box_width // 2), 0, yellow_box_x + (box_width // 2) - 1, box_height))
        else:
            img_cropped = img.crop((yellow_box_x - (box_width // 2), 0, yellow_box_x + (box_width // 2), box_height))
        img_cropped.save(f'{video_frame_folder}/{filename}')

    # 获取所有图片文件名
    filenames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]

    # 获取CPU核心数，设置线程数量为核心数的75%
    num_cores = multiprocessing.cpu_count()
    num_workers = int(num_cores * 0.75)

    args_list = [(filename, crop_info) for filename in filenames]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_image, args_list), total=len(filenames)))

    return "Done"


def calculate_person_height(img_array):
    """
    计算人物主体的高度。

    参数:
    img_array (numpy.array): 图像的numpy数组形式

    返回:
    int: 人物主体的高度
    """
    top = 0
    bottom = img_array.shape[0] - 1

    # 从上到下扫描，找到第一个白色像素的行
    for y in range(img_array.shape[0]):
        if np.any(img_array[y, :]):  # 如果这一行存在白色像素
            top = y
            break

    # 从下到上扫描，找到最后一个白色像素的行
    for y in range(img_array.shape[0] - 1, -1, -1):
        if np.any(img_array[y, :]):  # 如果这一行存在白色像素
            bottom = y
            break

    # 计算人物主体的高度
    person_height = bottom - top + 1

    return person_height


def calculate_continuous_white_ratio_binary(img_array, top_left, bottom_right, height, side='right'):
    """
    计算给定区域中连续白色像素占总白色像素的比例。

    参数:
    img_array (numpy.array): 图像的numpy数组形式
    top_left (tuple): 区域左上角的坐标
    bottom_right (tuple): 区域右下角的坐标
    height (int): 人物主体的高度
    side (str): 指定是在框的左边界还是右边界计算白色像素的连续占比，默认值为'right'

    返回:
    float: 给定区域中连续白色像素占总白色像素的比例
    """
    # 根据 side 参数选择边界的 x 坐标
    x = top_left[0] if side == 'left' else bottom_right[0] - 1

    # 提取边界的像素值
    edge_pixels = img_array[top_left[1]:bottom_right[1], x]

    # 如果总的像素为0，直接返回0
    if height == 0:
        return 0

    # 计算连续白色像素
    continuous_white_pixels = 0
    white_started = False
    for pixel in edge_pixels:
        if pixel:
            continuous_white_pixels += 1
            white_started = True
        elif white_started and continuous_white_pixels / height < 0.2:
            break

    # 计算连续白色像素占总的白色像素的比例
    continuous_white_ratio = continuous_white_pixels / height

    return continuous_white_ratio


def calculate_white_ratio_binary(img_array, top_left, bottom_right):
    """
    Calculate the ratio of 'white' pixels in a given area of an image.

    This function assumes the image is binary, with 'white' pixels being True and 'black' pixels being False.

    Parameters:
    img_array (numpy.array): The image as a numpy array
    top_left (tuple): The coordinates of the top left corner of the area
    bottom_right (tuple): The coordinates of the bottom right corner of the area

    Returns:
    float: The ratio of 'white' pixels in the area
    """
    # Extract the area from the image
    area = img_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Calculate the total number of pixels in the area
    total_pixels = area.shape[0] * area.shape[1]

    # If the total number of pixels is 0, return 0
    if total_pixels == 0:
        return 0

    # Count the number of 'white' pixels
    white_pixels = np.sum(area)

    # Calculate the ratio of 'white' pixels
    white_ratio = white_pixels / total_pixels

    return white_ratio


def get_fps(video_file):
    command = ['ffmpeg', '-i', video_file]
    proc = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    fps = None
    stderr = stderr.decode()
    lines = stderr.split('\n')
    for line in lines:
        if 'fps' in line:
            words = line.split(',')
            for word in words:
                if 'fps' in word:
                    fps = float(word.strip().split(' ')[0])
                    break
            if fps is not None:
                break
    return fps


def frames_to_video(project_path, input_folder, output_file, original_video, fps=None):
    if len(input_folder) == 0:
        input_folder = "video_frame"
    if len(output_file) == 0:
        output_file = "tmp.mp4"
    input_folder = str(os.path.join(project_path, input_folder))
    output_file = str(os.path.join(project_path, output_file))

    image_files = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    if not image_files:
        raise ValueError(f"No PNG images found in the input folder: {input_folder}")

    # Get fps from original video
    if fps is None:
        fps = get_fps(original_video)

    # Creating video from images using ffmpeg
    command = f"ffmpeg -y -r {fps} -i {input_folder}\\%05d.png -c:v libx264 -vf fps={fps} -pix_fmt yuv420p {output_file}"
    subprocess.call(command, shell=True)

    # Adding audio to the video
    audio_file = os.path.join(project_path, "tmp_audio.mp3")
    command = f"ffmpeg -y -i {original_video} -q:a 0 -map a {audio_file}"
    subprocess.call(command, shell=True)

    final_output = os.path.join(project_path, "final_output.mp4")
    command = f"ffmpeg -y -i {output_file} -i {audio_file} -shortest {final_output}"
    subprocess.call(command, shell=True)


def superposition(project_path, input_folder, frame_input_dir, exact):
    if len(input_folder) == 0:
        input_folder = "crossfade_tmp"
        input_folder = os.path.join(project_path, input_folder)
    if len(frame_input_dir) == 0 and not exact:
        frame_dir = os.path.join(project_path, "frame")
    elif len(frame_input_dir) == 0 and exact:
        frame_dir = os.path.join(project_path, "video_key")
    else:
        frame_dir = frame_input_dir
    if exact:
        super_folder = os.path.join(project_path, "img2img_key")
    else:
        super_folder = os.path.join(project_path, "refactor_frame")
    if exact:
        corp_json = os.path.join(project_path, "exact_crop_info.json")
    else:
        corp_json = os.path.join(project_path, "crop_info.json")
    if not os.path.exists(super_folder):
        os.makedirs(super_folder)
    if not os.path.exists(input_folder):
        return "请先使用ebs至少完成第七步，或者请填写正确的图片输入地址。"
    if not os.path.exists(frame_dir):
        return "请确认frame文件夹存在"
    if not os.path.exists(corp_json):
        return "请确认记录裁切信息的json文件存在"
    with open(corp_json, 'r') as f:
        crop_info = json.load(f)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.png'):
            # Load the images
            img1 = Image.open(f'{frame_dir}/{filename}')
            img2 = Image.open(f'{input_folder}/{filename}')

            # Get the cropping information
            center_coordinates = crop_info.get(filename, {}).get('center_coordinates', (0, 0))

            # Calculate the position to paste img2 onto img1
            position = (center_coordinates[0] - img2.width // 2, center_coordinates[1] - img2.height // 2)

            # Check if img2 has an alpha channel
            if img2.mode in ('RGBA', 'LA') or (img2.mode == 'P' and 'transparency' in img2.info):
                # Binary the alpha channel of img2
                alpha = img2.getchannel('A')
                binary_alpha = ImageOps.autocontrast(alpha)
                img2.putalpha(binary_alpha)

                # Paste img2 onto img1 using the binary alpha channel as the mask
                img1.paste(img2, position, img2)
            else:
                # Paste img2 onto img1 without a mask
                img1.paste(img2, position)

            # Save the result
            img1.save(f'{super_folder}/{filename}')
    return "Done"


def copy_png_files(source_folder, destination_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print("源文件夹不存在。")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    # 遍历文件夹中的每个文件
    for file in files:
        # 检查文件是否为 PNG 文件
        if file.endswith(".png"):
            # 构建源文件和目标文件的路径
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)

            # 复制文件
            shutil.copy(source_path, destination_path)

    print("文件复制完成。")


def exact_match(project_path, ebs):
    # 确保路径存在
    mask_folder = os.path.join(project_path, "video_mask")
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    frame_folder = os.path.join(project_path, "video_frame")
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    exact_mask_folder = os.path.join(project_path, "exact_mask")
    if not os.path.exists(exact_mask_folder):
        os.makedirs(exact_mask_folder)
    exact_frame_folder = os.path.join(project_path, "exact_frame")
    if not os.path.exists(exact_frame_folder):
        os.makedirs(exact_frame_folder)
    copy_png_files(os.path.join(project_path, "Mask"), mask_folder)
    copy_png_files(os.path.join(project_path, "frame"), frame_folder)
    key_folder = os.path.join(project_path, "video_key")
    if not os.path.exists(key_folder) and ebs:
        return "请先执行ebs stage 2 生成key帧"
    # 初始化存储裁剪信息的字典
    crop_info = {}

    # 遍历蒙版文件夹中的所有文件
    handle_folder = ""
    if ebs:
        exact_ime2img_key_mask = os.path.join(project_path, "exact_ime2img_key_mask")
        if not os.path.exists(exact_ime2img_key_mask):
            os.makedirs(exact_ime2img_key_mask)
        for i in os.listdir(key_folder):
            if not os.path.exists(os.path.join(exact_ime2img_key_mask, i)):
                shutil.copy(os.path.join(mask_folder, i), os.path.join(exact_ime2img_key_mask, i))
        handle_folder = exact_ime2img_key_mask
    else:
        handle_folder = mask_folder
    for filename in tqdm(os.listdir(handle_folder), desc="处理蒙版"):
        if filename.endswith('.png'):
            # 加载图像
            img = Image.open(os.path.join(mask_folder, filename))
            img_array = np.array(img)

            # 从四个方向向中心扫描，找到第一个非黑色像素点
            top = 0
            bottom = img_array.shape[0] - 1
            left = 0
            right = img_array.shape[1] - 1

            # 从上到下扫描
            for y in range(img_array.shape[0]):
                if np.any(img_array[y, :]):  # 如果这一行存在非黑色像素
                    top = y
                    break

            # 从下到上扫描
            for y in range(img_array.shape[0] - 1, -1, -1):
                if np.any(img_array[y, :]):  # 如果这一行存在非黑色像素
                    bottom = y
                    break

            # 从左到右扫描
            for x in range(img_array.shape[1]):
                if np.any(img_array[:, x]):  # 如果这一列存在非黑色像素
                    left = x
                    break

            # 从右到左扫描
            for x in range(img_array.shape[1] - 1, -1, -1):
                if np.any(img_array[:, x]):  # 如果这一列存在非黑色像素
                    right = x
                    break

            # 计算长和宽
            width = right - left + 1
            height = bottom - top + 1

            # 调整长和宽，使其能被4整除
            while width % 4 != 0:
                width += 1
            while height % 4 != 0:
                height += 1

            # 计算中心坐标
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # 裁切长方形区域
            img_cropped = img.crop(
                (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2))

            # 保存裁切后的图像
            img_cropped.save(os.path.join(exact_mask_folder, filename))

            # 裁剪对应的帧并保存
            if ebs:
                frame = Image.open(os.path.join(key_folder, filename))
            else:
                frame = Image.open(os.path.join(frame_folder, filename))

            frame_cropped = frame.crop(
                (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2))
            frame_cropped.save(os.path.join(exact_frame_folder, filename))

            # 记录裁剪信息
            crop_info[filename] = {
                'center_coordinates': (int(center_x), int(center_y)),
                'crop_x': int(center_x),
            }

    # 保存裁剪信息为json文件
    with open(os.path.join(project_path, 'exact_crop_info.json'), 'w') as f:
        json.dump(crop_info, f)

    return "Done"
