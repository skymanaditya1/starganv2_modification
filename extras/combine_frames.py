# Code used for combining the frames into a video
from glob import glob
import cv2
import os

def combine_frames(image_dir, frames_to_combine, video_name):
    images = glob(image_dir + '/*.jpg')
    print(images)
    total_frames = len(images)
    print(f'Total number of frames : {total_frames}')
    images = list()

    for i in range(total_frames):
        image_path = os.path.join(image_dir, 'rec_' + str(i+1).zfill(3) + '.jpg')
        images.append(image_path)

    print(images)

    # video_name = 'cropped_combined.mp4'

    if len(images) == 0:
        return

    frames = list()
    for image in images:
        print(f'Image : {image}')
        frame = cv2.imread(image)
        frames.append(frame)

    print(type(frames))
    height, width, channels = frames[0].shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release() # releases the video generated

if __name__ == '__main__':
    image_dir = 'mead_recs'
    video_name = 'mead_recs.mp4'
    combine_frames(image_dir, 20, video_name)