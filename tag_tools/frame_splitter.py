import cv2
import os
from tqdm import tqdm
import math

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
            filename = f"{saved_frames+1:0{num_digits}d}.png"
            frame_path = os.path.join(frame_dir, filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
            pbar.update(1)

        frame_count += 1

    pbar.close()

    # Release the video file
    cap.release()

    # Create the additional directories
    for dir_name in ["Out", "Human", "Mask"]:
        os.makedirs(os.path.join(project_path, dir_name), exist_ok=True)
