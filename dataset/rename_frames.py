import os
import shutil

# Define the root folder containing subfolders
root_folder = '/media/imaginarium/12T/Dataset/main_camera_frames'
destination_folder = '/media/imaginarium/12T/Dataset/SyncDataset/3rd_frames'  # Folder where the new folders with renamed images will be created


def get_image_number(filename):
    # Split the filename by underscores and extract the last part (before .png)
    base_name = os.path.splitext(filename)[0]  # Remove extension
    parts = base_name.split('_')
    try:
        return int(parts[-1])  # Convert the last part to an integer
    except ValueError:
        return float('inf')  # Return a high value if conversion fails to sort correctly


def rename_and_copy_images(folder_path, new_folder_path, person_id):
    # List all image files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]

    # Sort files based on the number extracted from filenames
    files.sort(key=get_image_number)

    # Ensure the destination folder exists
    os.makedirs(new_folder_path, exist_ok=True)

    # Rename and copy files
    FrameID = 1
    for index, file in enumerate(files):
        if index >0 and index % 1000 == 0:
            print(FrameID, ' skip to snyc 29.97FPS')
            continue
        old_path = os.path.join(folder_path, file)
        new_filename = f'person_{person_id}_frame_{FrameID}.png'
        new_path = os.path.join(new_folder_path, new_filename)
        shutil.copy(old_path, new_path)
        print(f"Copied and renamed {file} to {new_filename}")
        FrameID = FrameID + 1


def main():
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Extract the person ID from the folder name
            person_id = folder_name.split('_')[-1]

            if int(person_id) < 14:
                # Create a new folder path for the person ID
                new_folder_path = os.path.join(destination_folder, f'person_{person_id}')

                # Rename and copy images to the new folder
                rename_and_copy_images(folder_path, new_folder_path, person_id)


if __name__ == "__main__":
    main()
