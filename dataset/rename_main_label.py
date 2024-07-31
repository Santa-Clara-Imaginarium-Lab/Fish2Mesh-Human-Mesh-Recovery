import os
import shutil
import re

# Define the source and destination directories
src_dir = '/media/imaginarium/12T/Dataset/main_camera_label/'
dst_dir = '/media/imaginarium/12T/Dataset/new_name_label/'

# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)


# Function to extract the prefix and number from the filename
def extract_prefix_and_number(filename):
    match = re.match(r'^(.*_frame_)(\d+)\.pkl$', filename)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None


# Scan the directory for .pkl files and group them by their series prefix
files_by_series = {}
for filename in os.listdir(src_dir):
    if filename.endswith('.pkl'):
        prefix, number = extract_prefix_and_number(filename)
        if prefix is not None:
            if prefix not in files_by_series:
                files_by_series[prefix] = []
            files_by_series[prefix].append((filename, number))

# Process each series
for prefix, files in files_by_series.items():
    # Sort the files by the last number
    files.sort(key=lambda x: x[1])

    # Rename and copy the files to the new folder
    for idx, (filename, number) in enumerate(files, start=1):
        new_filename = f'{prefix}{idx}.pkl'
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_filename)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
            print(f'Copied {src_path} to {dst_path}')
