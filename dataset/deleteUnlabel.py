import os


def clean_frames_with_labels(frames_folder, labels_folder):
    # Get the list of all frames in the frames folder
    # frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.png')]
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.pkl')]


    # Get the list of all labels in the labels folder
    # label_files = [f for f in os.listdir(labels_folder) if f.endswith('.pkl')]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.png')]


    # Convert the list of labels to a set of label names without extensions
    label_names = {os.path.splitext(f)[0] for f in label_files}

    for frame_file in frame_files:
        # Get the name of the frame file without extension
        frame_name = os.path.splitext(frame_file)[0]

        # Check if the corresponding label exists
        if frame_name not in label_names:
            # If not, delete the frame
            frame_path = os.path.join(frames_folder, frame_file)
            os.remove(frame_path)
            print(f"Deleted {frame_file}")


# Example usage
# frames_folder = '/media/imaginarium/12T/Dataset/headset_frames_all'  # Folder containing the frames
# labels_folder = '/media/imaginarium/12T/Dataset/main_camera_label'  # Folder containing the labels
frames_folder = '/media/imaginarium/12T/Dataset/main_camera_label'  # Folder containing the frames
labels_folder = '/media/imaginarium/12T/Dataset/headset_frames_all'

clean_frames_with_labels(frames_folder, labels_folder)
