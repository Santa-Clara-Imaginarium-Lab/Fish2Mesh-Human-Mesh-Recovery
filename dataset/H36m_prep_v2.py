import numpy as np
from PIL import Image
import cv2
import os
import glob
from scipy.io import loadmat
import pickle
import multiprocessing
from functools import partial
from transformers import CLIPProcessor, CLIPVisionModel
# from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from tqdm import tqdm
import json
import time
import sys

# Add the D3Pose directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from D3Pose.model.geometry import *
from D3Pose.model.smpl_wrapper import *

if __name__ == '__main__':
    root_path = '/home/imaginarium/Documents/dataset/human3.6M/original/'
    # output_folder = '/media/imaginarium/a0c299ce-8eea-4f25-92a4-b572d215821b/MergeDataset/'
    output_folder = '/media/imaginarium/12T/Dataset/validation80/' # validation
    GT_path = root_path + 'h36m_validation.pkl'
    smpl_gt_directory = '/home/imaginarium/Documents/dataset/human3.6M/original/SMPL/'

    with open(GT_path, 'rb') as file:
        GT_file = pickle.load(file)

    # print(GT_file)
    index = 0

    # right now just support max_num = 1 and skip_num = 1
    MAX_NUM = 80
    SKIP_NUM = 60

    save_frames = []
    # save_GTs = []
    # save_tensor = []
    save_joints_3d = []
    save_joints_2d = []
    save_image = []
    save_pose = []
    save_shape = []
    save_trans = []
    # s, ret_R, ret_t = 0,0,0
    current_frame_name = ''
    saveID = 0
    last_subject = -999

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda')

    SMPL_CONFIG = {'data_dir': '/home/imaginarium/.cache/4DHumans/data/',
                   'model_path': '/home/imaginarium/.cache/4DHumans/data//smpl',
                   'gender': 'neutral',
                   'num_body_joints': 23,
                   'joint_regressor_extra': '/home/imaginarium/.cache/4DHumans/data//SMPL_to_J19.pkl',
                   'mean_params': '/home/imaginarium/.cache/4DHumans/data//smpl_mean_params.npz'}

    # regression SMPL->joints
    smpl_model = SMPL(**SMPL_CONFIG)


    for i in tqdm(range(len(GT_file))):
        # if GT_file[i]['camera_id'] == 2:
        #     temp = GT_file[i]
        #     print('find camera 1')
        if len(save_joints_3d) > 0 and current_frame_name != GT_file[i]['image'].split('/')[0]:
            # if start with a new video, clean all previous arrays
            current_frame_name = GT_file[i]['image'].split('/')[0]
            save_frames = []
            # save_GTs = []
            # save_tensor = []
            save_joints_3d = []
            save_joints_2d = []
            save_image = []
            save_pose = []
            save_shape = []
            save_trans = []
            # s, ret_R, ret_t = 0, 0, 0
            # print('deal with ',current_frame_name)
        else:
            prevTime1 = time.time()
            current_frame_name = GT_file[i]['image'].split('/')[0]
            subject = str(GT_file[i]['subject'])
            action = str(GT_file[i]['action'])  # Convert action index to integer and back to string to remove leading zero
            subaction = str(GT_file[i]['subaction'])

            # [:-4] remove .jpg
            frame_ID = int(GT_file[i]['image'].split('/')[-1][:-4].split('_')[-1])

            if last_subject != int(subject):
                smpl_gt_file = f'Human36M_subject{subject}_SMPL_NeuralAnnot.json'
                with open(os.path.join(smpl_gt_directory, smpl_gt_file), 'r') as f:
                    gt_data = json.load(f)
                last_subject = int(subject)


            # GT_joints = GT_file[i]['joints_3d']

            GT = {}

            # GT['joints_3d'] = GT_file[i]['joints_3d']
            # GT['joints_2d'] = GT_file[i]['joints_2d']

            GT['image'] = GT_file[i]['image']
            prevTime2 = time.time()


            frame_path = root_path + 'images/' + GT_file[i]['image']
            try:
                frame = Image.open(frame_path)
                gt_clip_data = gt_data[action][subaction][str(frame_ID-1)]
                GT['pose'] = gt_clip_data['pose']
                GT['shape'] = gt_clip_data['shape']
                GT['trans'] = gt_clip_data['trans']

                # Convert the list to a PyTorch tensor
                pose_tensor = torch.tensor(GT['pose']).view(24, 3)

                GT_pose = aa_to_rotmat(pose_tensor).view(-1, 24, 3, 3)
                out_global_orient = GT_pose[:,0,:,:]
                out_body_pose = GT_pose[:, 1:,:,:]

                out_betas = torch.tensor(GT['shape']).view(1, 10)
                smpl_output = smpl_model(
                    **{'global_orient': out_global_orient, 'body_pose': out_body_pose, 'betas': out_betas},
                    pose2rot=False)
                pred_keypoints_3d = smpl_output.joints
                # # pred_vertices = smpl_output.vertices
                #
                focal_length = 5000 * torch.ones(1, 2, device='cpu', dtype=torch.float32)
                focal_length = focal_length.reshape(-1, 2)
                out_pred_cam = torch.tensor(GT['trans']).view(1, 3)
                pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=out_pred_cam,
                                                           focal_length=focal_length / 256)

                GT['joints_3d'] = pred_keypoints_3d
                GT['joints_2d'] = pred_keypoints_2d

            except:
                print('missing:' ,frame_path)
                save_frames = []
                # save_GTs = []
                # save_tensor = []
                save_joints_3d = []
                save_joints_2d = []
                save_image = []
                save_pose = []
                save_shape = []
                save_trans = []
                s, ret_R, ret_t = 0, 0, 0
                continue

            # CLIP_inputs = processor(images=frame, return_tensors="pt")
            # outputs = model(**CLIP_inputs.to('cuda'))
            # last_hidden_state = outputs.last_hidden_state
            if len(save_joints_3d) < MAX_NUM:
                # save_GTs.append(GT)
                save_frames.append(frame)
                # save_tensor.append(last_hidden_state)

                save_joints_3d.append(GT['joints_3d'])
                save_joints_2d.append(GT['joints_2d'])
                save_image.append(GT['image'])
                save_pose.append(GT['pose'])
                save_shape.append(GT['shape'])
                save_trans.append(GT['trans'])

            else:

                savePath = output_folder + 'feature_maps/H36m' + '_' + str(saveID) + '.pt'
                # output = torch.cat(save_tensor, dim=0)
                # torch.save(output.detach(), savePath)

                CLIP_inputs = processor(images=save_frames, return_tensors="pt")
                torch.save(CLIP_inputs['pixel_values'].detach(), savePath)

                GT_savePath = output_folder + 'gt/H36m' + '_' + str(saveID) + '.pkl'
                # np.save(GT_savePath,save_GTs)
                # print('saved ',str(saveID),'size:',output.shape,len(save_GTs))

                save_GTs = {}
                save_GTs['joints_3d'] = torch.stack(save_joints_3d)
                save_GTs['joints_2d'] = torch.stack(save_joints_2d)
                save_GTs['image'] = save_image
                save_GTs['pose'] = torch.tensor(np.array(save_pose))
                save_GTs['shape'] = torch.tensor(np.array(save_shape))
                save_GTs['trans'] = torch.tensor(np.array(save_trans))
                with open(os.path.join(GT_savePath),"wb") as f:
                    pickle.dump(save_GTs, f)
                saveID = saveID + 1

                prevTime3 = time.time()

                # remove the first to SKIP_NUM th  elements and add the new one
                # del save_GTs[0:SKIP_NUM]
                del save_frames[0:SKIP_NUM]
                # del save_tensor[0:SKIP_NUM]

                del save_joints_3d[0:SKIP_NUM]
                del save_joints_2d[0:SKIP_NUM]
                del save_image[0:SKIP_NUM]
                del save_pose[0:SKIP_NUM]
                del save_shape[0:SKIP_NUM]
                del save_trans[0:SKIP_NUM]

                # save_GTs.append(GT)
                # save_tensor.append(last_hidden_state)
                # save_frames.append(frame)
                prevTime4 = time.time()

                # print('load smpl json time: ',prevTime2 - prevTime1,
                #       'save pkl time: ',prevTime3 - prevTime2,
                #       'remove cachce time:', prevTime4 - prevTime3)