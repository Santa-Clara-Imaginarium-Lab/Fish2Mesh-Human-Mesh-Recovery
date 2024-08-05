import cv2
import os


# Function to create output directory if it doesn't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to extract frames from a video and save as images
def video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()


# Main function to process all videos in the specified folder
def process_videos(input_folder, output_folder):
    create_dir(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv",'.MOV')):  # Add more video extensions if needed
            video_path = os.path.join(input_folder, filename)
            video_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            create_dir(video_output_folder)

            video_to_frames(video_path, video_output_folder)
            print(f"Processed {filename}")


# Example usage
input_folder = '/media/imaginarium/12T/Dataset/temp_convert_headset_videos/'
output_folder = '/media/imaginarium/12T/Dataset/temp_headset_frames/'
# input_folder = '/media/imaginarium/12T/Dataset/temp_main_viedos/'
# output_folder = '/media/imaginarium/12T/Dataset/temp_main_frames/'

process_videos(input_folder, output_folder)
