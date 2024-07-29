import os
import shutil


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_last_number(folder_name):
    parts = folder_name.split('_')
    return parts[-1]


def copy_and_rename_frames(input_folder, output_folder):
    create_dir(output_folder)

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.startswith("processed_headset_"):
            last_number = get_last_number(subfolder)
            frame_files = sorted(os.listdir(subfolder_path))  # Sort to maintain frame order

            for idx, frame_file in enumerate(frame_files):
                if frame_file.endswith((".png", ".jpg", ".jpeg")):  # Add more image extensions if needed
                    frame_path = os.path.join(subfolder_path, frame_file)
                    new_frame_name = f"person_{last_number}_frame_{idx + 1}.png"
                    new_frame_path = os.path.join(output_folder, new_frame_name)
                    shutil.copy(frame_path, new_frame_path)
                    print(f"Copied {frame_file} to {new_frame_name}")


# Example usage
input_folder = '/media/imaginarium/12T/Dataset/headset_frames/'  # Folder containing subfolders like processed_headset_1, processed_headset_2, etc.
output_folder = '/media/imaginarium/12T/Dataset/headset_frames_all/'  # Folder to save renamed frames

copy_and_rename_frames(input_folder, output_folder)
