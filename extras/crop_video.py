# Code used for cropping the video using ffmpeg based on the start and end times of the video 
import os

if __name__ == '__main__':
    # Download the video using yt-dlp
    video_download_dir = 'videos_downloaded'
    os.makedirs(video_download_dir, exist_ok=True)
    video_id = 'v0V_zkng4go'
    # yt_dlp_template = "yt-dlp -f \"best\" https://www.youtube.com/watch?v={} -o '{}/%(id)s.%(ext)s'".format(video_id, video_download_dir)

    # print(f'Downloading video')
    # os.system(yt_dlp_template.format(video_id, video_download_dir))
    # print(f'Downloaded video')

    ffmpeg_template = 'ffmpeg -i {} -ss {} -to {} -strict -2 -c:v libx264 {}'
    start_seconds = 58
    end_seconds = 62
    input_video_file = os.path.join(video_download_dir, video_id + '.mp4')
    cropped_filename = 'cropped_frown.mp4'
    output_video_file = os.path.join(video_download_dir, cropped_filename)

    print(f'Generating cropped : {cropped_filename}')
    os.system(ffmpeg_template.format(input_video_file, start_seconds, end_seconds, output_video_file))
    print(f'Generated cropped : {cropped_filename}')

    start_seconds = 39
    end_seconds = 42
    cropped_filename = 'cropped_neutral.mp4'
    output_video_file = os.path.join(video_download_dir, cropped_filename)

    print(f'Generating cropped : {cropped_filename}')
    os.system(ffmpeg_template.format(input_video_file, start_seconds, end_seconds, output_video_file))
    print(f'Generating cropped : {cropped_filename}')