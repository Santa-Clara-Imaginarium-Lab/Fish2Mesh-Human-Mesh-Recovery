import os
import random
import shutil

# Define the path to your folder
img_folder = '/media/imaginarium/12T/Dataset/headset_frames_all'

# Get a list of all image files in the folder
image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(img_folder) for f in filenames]

# Shuffle the list
random.shuffle(image_files)

# Define the split sizes
train_size = 184000
valid_size = 5773
test_size = 5000

# Split the list
train_images = image_files[:train_size]
valid_images = image_files[train_size:train_size + valid_size]
test_images = image_files[train_size + valid_size:train_size + valid_size + test_size]

# Create directories for train, valid, and test sets
output = '/media/imaginarium/12T/Dataset/'
train_dir = os.path.join(output, 'train')
valid_dir = os.path.join(output, 'valid')
test_dir = os.path.join(output, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move the files to their respective directories
for img in train_images:
    shutil.move(img, os.path.join(train_dir, os.path.basename(img)))

for img in valid_images:
    shutil.move(img, os.path.join(valid_dir, os.path.basename(img)))

for img in test_images:
    shutil.move(img, os.path.join(test_dir, os.path.basename(img)))

print("Dataset split complete.")
