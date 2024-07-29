import numpy as np
from PIL import Image
import os
import pickle
import multiprocessing
from tqdm import tqdm
import json
from functools import partial  # Import partial function



def process_frame(frame_info, smpl_gt_directory, output_folder, MAX_NUM, SKIP_NUM):
    frame_path, GT_file_entry = frame_info
    subject = str(GT_file_entry['subject'])
    action = str(GT_file_entry['action'])
    subaction = str(GT_file_entry['subaction'])
    frame_ID = int(GT_file_entry['image'].split('/')[-1][:-4].split('_')[-1])
    smpl_gt_file = f'Human36M_subject{subject}_SMPL_NeuralAnnot.json'

    with open(os.path.join(smpl_gt_directory, smpl_gt_file), 'r') as f:
        gt_data = json.load(f)



    GT = {}
    GT['joints_3d'] = GT_file_entry['joints_3d']
    GT['joints_2d'] = GT_file_entry['joints_2d']
    GT['image'] = GT_file_entry['image']


    try:
        frame = Image.open(frame_path)
        gt_clip_data = gt_data[action][subaction][str(frame_ID-1)]
        GT['pose'] = gt_clip_data['pose']
        GT['shape'] = gt_clip_data['shape']
        GT['trans'] = gt_clip_data['trans']
        return GT, frame
    except:
        print('missing:', frame_path)
        return None, None


if __name__ == '__main__':
    root_path = '/home/tianma/文档/datasets/human3.6m/'
    output_folder = '/home/tianma/文档/datasets/human3.6m/processData/train/'
    GT_path = root_path + 'h36m_train.pkl'
    smpl_gt_directory = '/home/tianma/文档/datasets/human3.6m/SMPL/'

    with open(GT_path, 'rb') as file:
        GT_file = pickle.load(file)

    MAX_NUM = 1
    SKIP_NUM = 0

    save_frames = []
    save_GTs = []
    current_frame_name = ''
    saveID = 0

    frame_info_list = [(root_path + 'images/' + entry['image'], entry) for entry in GT_file]

    with multiprocessing.Pool() as pool:
        results_iter = pool.imap(
            partial(process_frame, smpl_gt_directory=smpl_gt_directory, output_folder=output_folder, MAX_NUM=MAX_NUM,
                    SKIP_NUM=SKIP_NUM), frame_info_list)

        for result in tqdm(results_iter, total=len(frame_info_list)):
            GT, frame = result
            if GT is not None and frame is not None:
                if len(save_frames) < MAX_NUM:
                    save_GTs.append(GT)
                    save_frames.append(frame)
                else:
                    GT_savePath = output_folder + 'gt/H36m' + '_' + str(saveID) + '.pkl'
                    with open(os.path.join(GT_savePath), "wb") as f:
                        pickle.dump(GT, f)
                    saveID = saveID + 1

                    del save_GTs[0:SKIP_NUM]
                    del save_frames[0:SKIP_NUM]

                    save_GTs.append(GT)
                    save_frames.append(frame)
