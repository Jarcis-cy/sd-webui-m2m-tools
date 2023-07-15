import cv2
import os
import re
import shutil
from tqdm import tqdm
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import json


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


def reconfiguration(project_path, width, movie_path, step):
    video_frame_folder = os.path.join(project_path, "video_frame")
    video_mask_folder = os.path.join(project_path, "video_mask")
    if not os.path.exists(video_frame_folder):
        os.makedirs(video_frame_folder)
    if not os.path.exists(video_mask_folder):
        os.makedirs(video_mask_folder)

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
    if width > owidth:
        return "输入的宽度请小于原视频宽度"
    # The width and height of the box
    box_width = width
    box_height = height

    # A dictionary to store the cropping information for each file
    crop_info = {}

    # Process the files in the 'mask' directory
    for filename in os.listdir(mask_folder):
        if filename.endswith('.png'):
            # Load the image
            img = Image.open(f'{mask_folder}/{filename}')
            img_array = np.array(img)

            # Initialize the max_white_ratio
            max_white_ratio = 0

            # The position of the max box
            max_box_x = 0

            for x in range(0, img_array.shape[1] - box_width, step):
                white_ratio = calculate_white_ratio_binary(img_array, (x, 0), (x + box_width, box_height))
                if white_ratio > max_white_ratio:
                    max_white_ratio = white_ratio
                    max_box_x = x

            # Calculate the white ratio for each column and find the column with the highest white ratio
            column_white_ratios = np.sum(img_array, axis=0) / img_array.shape[0]
            max_column_x = np.argmax(column_white_ratios[max_box_x:max_box_x + box_width]) + max_box_x

            # Adjust the position of the max box
            if max_box_x + box_width / 2 < max_column_x:
                max_box_x += min(5, max_column_x - (max_box_x + box_width / 2))
            else:
                max_box_x -= min(5, (max_box_x + box_width / 2) - max_column_x)

            # Ensure the max box doesn't go out of the image boundaries
            max_box_x = max(0, min(img_array.shape[1] - box_width, max_box_x))

            # Record the cropping information
            crop_info[filename] = max_box_x

            # Crop the image to the box and save it
            img_cropped = img.crop((max_box_x, 0, max_box_x + box_width, box_height))
            img_cropped.save(f'{video_mask_folder}/{filename}')

    # Save the cropping information as a json file
    with open(os.path.join(project_path, 'crop_info.json'), 'w') as f:
        json.dump(crop_info, f)

    # Process the files in the 'frame' directory
    for filename in os.listdir(frame_folder):
        if filename.endswith('.png'):
            # Load the image
            img = Image.open(f'{frame_folder}/{filename}')

            # Get the cropping information
            max_box_x = crop_info.get(filename, 0)

            # Crop the image to the box and save it
            img_cropped = img.crop((max_box_x, 0, max_box_x + box_width, box_height))
            img_cropped.save(f'{video_frame_folder}/{filename}')
    return "Done"


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
