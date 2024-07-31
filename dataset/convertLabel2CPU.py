import torch
import os
import pickle

def cuda_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: cuda_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [cuda_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(cuda_to_cpu(item) for item in data)
    else:
        return data


frames_folder = '/media/imaginarium/12T/Dataset/main_camera_label'  # Folder containing the frames
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.pkl')]

for frame_file in frame_files:
    file_path = os.path.join(frames_folder, frame_file)

    with open(file_path, 'rb') as file:
        GT_file = pickle.load(file)

    # Check if GT_file is empty
    if isinstance(GT_file, dict):
        if not GT_file:
            print(f"{frame_file} is an empty dictionary.")
        # else:
        #     print(f"{frame_file} is not empty.")
    elif isinstance(GT_file, list) or isinstance(GT_file, tuple):
        if not GT_file:
            print(f"{frame_file} is an empty list/tuple.")
        else:
            print(f"{frame_file} is not empty.")
    else:
        print(f"{frame_file} is neither a dictionary, list, nor tuple. It contains: {GT_file}")

    # convertData = cuda_to_cpu(GT_file)
    #
    # # Save the converted data
    # with open(file_path, 'wb') as file:
    #     pickle.dump(convertData, file)