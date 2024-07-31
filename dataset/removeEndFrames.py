import os
import re


def delete_files_with_high_numbers(folder, prefix, threshold):
    # Get all .pkl files in the folder
    frame_files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

    for frame_file in frame_files:
        # Check if the file starts with the specified prefix
        if frame_file.startswith(prefix):
            # Extract the base name without extension
            base_name = os.path.splitext(frame_file)[0]

            # Extract the number from the base name using regex
            match = re.search(r'_(\d+)$', base_name)
            if match:
                number = int(match.group(1))

                # Check if the number is larger than the threshold
                if number > threshold:
                    # Construct full file path
                    file_path = os.path.join(folder, frame_file)

                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted {file_path}")


# Define the folder, prefix, and threshold
frames_folder = '/media/imaginarium/12T/Dataset/main_camera_label'
prefix = 'person_1_frame'
threshold = 24000

# Call the function to delete files
delete_files_with_high_numbers(frames_folder, prefix, threshold)
