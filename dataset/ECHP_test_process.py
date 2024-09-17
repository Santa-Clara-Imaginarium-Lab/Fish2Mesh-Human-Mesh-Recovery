import os
import random
import shutil


def get_filtered_images(input_folder, prefix):
    return [f for f in os.listdir(input_folder) if
            f.startswith(prefix) and os.path.isfile(os.path.join(input_folder, f))]


def select_random_images(image_list, num_samples):
    return random.sample(image_list, min(num_samples, len(image_list)))


def copy_files(src_folder, dest_folder, filenames):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for filename in filenames:
        shutil.copy(os.path.join(src_folder, filename), os.path.join(dest_folder, filename))


def main():
    input_folder = '/media/imaginarium/2T/headset_frames_all'
    gt_folder = '/media/imaginarium/2T/new_name_label'
    output_folder_images = '/media/imaginarium/2T/ECHP_test/frames'
    output_folder_gt = '/media/imaginarium/2T/ECHP_test/gt'
    prefix = 'env'
    num_samples = 5000

    # Get all images starting with 'env'
    all_images = get_filtered_images(input_folder, prefix)

    # Randomly select 5000 images
    selected_images = select_random_images(all_images, num_samples)

    # Copy selected images to output folder
    copy_files(input_folder, output_folder_images, selected_images)

    # Prepare GT filenames corresponding to selected images
    selected_gt_files = [f'{os.path.splitext(f)[0]}.pkl' for f in selected_images]

    # Prepare GT filenames corresponding to selected images
    # selected_gt_files = [f for f in selected_images if os.path.exists(os.path.join(gt_folder, f))]

    # Copy selected GT images to output folder
    copy_files(gt_folder, output_folder_gt, selected_gt_files)

    print(f"Selected {len(selected_images)} images and {len(selected_gt_files)} GT images.")


if __name__ == '__main__':
    main()
