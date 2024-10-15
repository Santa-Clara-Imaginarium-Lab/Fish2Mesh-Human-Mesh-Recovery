import os
import shutil


def copy_gt_files(image_folder, gt_folder, new_folder):
    # Ensure the new folder exists
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Get list of image files in the image folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        # Remove the suffix to get the base name
        base_name, _ = os.path.splitext(image_file)

        # Create the expected GT file name by adding the .pkl suffix
        gt_file = base_name + '.pkl'

        # Check if the GT file exists in the gt_folder
        gt_file_path = os.path.join(gt_folder, gt_file)
        if os.path.isfile(gt_file_path):
            # Copy the GT file to the new folder
            shutil.copy(gt_file_path, new_folder)
            print(f'Copied: {gt_file}')
        else:
            print(f'GT file not found: {gt_file}')


# Define the folders

gt_folder = '/media/imaginarium/2T/new_name_label/'
# image_folder = '/media/imaginarium/2T/ECHP/frames/'
# new_folder = '/media/imaginarium/2T/ECHP/GT/'

# image_folder = '/media/imaginarium/2T/Ego4D/frames/'
# new_folder = '/media/imaginarium/2T/Ego4D/GT/'

image_folder = '/media/imaginarium/2T/V3/test/'
new_folder = '/media/imaginarium/2T/V3/GT/'

# Call the function
copy_gt_files(image_folder, gt_folder, new_folder)
