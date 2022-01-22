# Multi GPU multi batch code for generating keypoints using the defined keypoint generator
import face_alignment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process
# from torch.multiprocessing import Pool, Process, set_start_method
# import torch.multiprocessing as mp
import math
from itertools import product
import argparse
from glob import glob

ngpus = torch.cuda.device_count()
ngpus = 1

# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id)) for id in range(ngpus)]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')

def drawPolyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks[i][0], landmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)

# Draw lines around landmarks corresponding to different facial regions
def drawPolylines(image, landmarks):
    drawPolyline(image, landmarks, 0, 16)           # Jaw line
    drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(image, landmarks, 27, 30)          # Nose bridge
    drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(image, landmarks, 36, 41, True)    # Left eye
    drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

# Detect landmarks for the given batch
def batch_landmarks(batches, fa):
    landmarks_detected = 0
    batch_landmarks = list()

    for current_batch in batches:
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:0')
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id))
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2)
        landmarks = fa.get_landmarks_from_batch(current_batch)
        landmarks_detected += len(landmarks)
        batch_landmarks.extend(landmarks)

    return batch_landmarks, landmarks_detected

# generate the face crop from the frames in the video 
def generate_face_crops(video_path, debug=True):
    lower_face_buffer = 0.3
    upper_face_buffer = 0.8
    # processed_folder = 'Processed'
    # os.makedirs(processed_folder, exist_ok=True)

    batch_size = 32
    resize_dim = 256

    video_stream = cv2.VideoCapture(video_path)

    image_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Frames read : {total_frames}, image height and width: {image_height}, {image_width}', flush=True)

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                print(f'Batch size reached 0')
                continue
            landmarks, landmarks_detected = batch_landmarks(batches, fa)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = batch_size // 2
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}', flush=True)
            batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
            continue

    landmark_threshold = 68 # Ignore frames where landmarks detected is not equal to landmark_threshold
    frames_ignored = 0
    frame_ignore_threshold = 10 # reject video if more than 10% of frames are bad 
    
    resized_gt = list() # resized gt image

    for i, landmark in enumerate(landmarks):
        image = frames[i]
        if (len(landmark) != landmark_threshold):
            frames_ignored += 1
            continue

        min_x, min_y, max_x, max_y = min(landmark[:, 0]), min(landmark[:, 1]), max(landmark[:, 0]), max(landmark[:, 1])

        x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
        x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
        y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
        y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

        size = max(x_right - x_left, y_down - y_top)

        sw = int((x_left + x_right) / 2 - size // 2)

        if (sw < 0):
            sw = 0
        if (sw + size > image_width):
            frames_ignored += 1
            continue

        original_cropped = image[y_top:y_down, sw:sw+size]
        resized_original = cv2.resize(original_cropped, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_gt.append(resized_original)

    print(f'Video : {video_path}, Total frames : {total_frames}, gt frames : {len(resized_gt)}', flush=True)
    image_folder_path = os.path.basename(video_path).split('.')[0]
    os.makedirs(image_folder_path, exist_ok=True)

    if debug:
        for i in range(len(resized_gt)):
            gt_filepath = os.path.join(image_folder_path, 'gt_' + str(i+1).zfill(3) + '.jpg')
            cv2.imwrite(gt_filepath, resized_gt[i])
            
if __name__ == '__main__':
    # Takes a video file as input and generates a sequence of face crops
    # video_id = 'LER70aFi2nQ'
    # dir_name = 'scarlet_video'
    # os.makedirs(dir_name, exist_ok=True)
    # yt_dlp_template = "yt-dlp -f \"best\" https://www.youtube.com/watch?v={} -o '{}/%(id)s.%(ext)s'".format(video_id, dir_name)
    # os.system(yt_dlp_template)

    dir_name = 'mead'
    video_file = 'mead_disgusted.mp4'

    video_path = os.path.join(dir_name, video_file)

    # Code for generating the face crops from the video 
    generate_face_crops(video_path)

    video_file = 'mead_happy.mp4'
    video_path = os.path.join(dir_name, video_file)

    # Code for generating the face crops from the video 
    generate_face_crops(video_path)