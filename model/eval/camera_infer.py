import cv2
import os
import argparse
import random
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
from model.HMR_Model.EgoHMR_EgoPositionEmbedding import EgoHMR_pos
from model.util.smpl_wrapper import SMPL
from model.util.geometry import *
from model.util.renderer import Renderer
import numpy as np


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test script for trained model.")

    parser.add_argument(
        "--checkpoint", type=str, default="../save/142.ckpt",
        help="Path to the saved checkpoint file."
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use CUDA if available")

    args = parser.parse_args(argv)
    return args

def visualize_keypoints(image, keypoints):
    """
    Draw keypoints on an image and display the result.

    Parameters:
        image (numpy.ndarray): The image on which to draw keypoints.
        keypoints (numpy.ndarray): An array of keypoints with shape (num_keypoints, 2).
    """
    output_image = image.copy()

    color = (0, 255, 0)  # Green
    radius = 2
    thickness = -1  # Fill the circle

    for (x, y) in keypoints:
        cv2.circle(output_image, (int(x), int(y)), radius, color, thickness)

    return output_image



def infer_model(images,test_transforms,model,renderer,device, args):



    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)



    with torch.no_grad():
        # Convert OpenCV image (NumPy array) to PIL image
        image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        pil_image = Image.fromarray(image)

        # Apply the transformations
        transformed_image = test_transforms(pil_image)
        transformed_image = transformed_image.unsqueeze(0).to(device)  # Add batch

        out = model(transformed_image)
        out_global_orient = out['out_global_orient']
        out_body_pose = out['out_body_pose']
        out_betas = out['out_betas']
        out_pred_cam = out['out_pred_cam']
        pred_keypoints_3d = out['pred_keypoints_3d']
        pred_keypoints_2d = out['pred_keypoints_2d']
        pred_vertices = out['pred_vertices']

        regression_img = renderer(pred_vertices[0].detach().cpu().numpy(),  # GT_npy['pred_vertices'][0][0]
                                  out_pred_cam[0].detach().cpu().numpy(),  # GT_cam[0].detach().cpu().numpy()
                                  torch.zeros((3, 256, 256), dtype=torch.float32),  # Images[0].cpu(),
                                  mesh_base_color=LIGHT_BLUE,
                                  scene_bg_color=(1, 1, 1),
                                  )

        # print('2d:',pred_keypoints_2d)
    return regression_img


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    # Specify which GPU to use, if you have multiple GPUs
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    SMPL_CONFIG = {'data_dir': '/home/imaginarium/.cache/4DHumans/data/',
                   'model_path': '/home/imaginarium/.cache/4DHumans/data//smpl',
                   'gender': 'neutral',
                   'num_body_joints': 23,
                   'joint_regressor_extra': '/home/imaginarium/.cache/4DHumans/data//SMPL_to_J19.pkl',
                   'mean_params': '/home/imaginarium/.cache/4DHumans/data//smpl_mean_params.npz'}
    smpl_model = SMPL(**SMPL_CONFIG).to(device)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    model = EgoHMR_pos()
    model = model.to(device)

    faceArray = np.load('./model_faces.npy')
    renderer = Renderer(faces=faceArray)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Checkpoint loaded.")

    # # Open the default camera (usually the built-in webcam)
    # cap = cv2.VideoCapture(0)

    # Path to your video file
    video_path = '/media/imaginarium/12T/Dataset/headset_videos/processed_headset_8.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 256*2
    height = 256

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video file
    out = cv2.VideoWriter('output_video8.avi', fourcc, fps,
                          (width, height))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_id = frame_id + 1
        # Check if frame was captured successfully
        if not ret:
            print("Failed to grab frame")
            break

        if frame_id < 360:
            continue

        if frame_id > 2000:
            break

        # Calculate the new height (half of the original height)
        height, width, channels = frame.shape
        print(frame_id, 'origin frame size: ', height, width, channels)
        # new_height = height // 2
        #
        # # Crop the frame to keep only the top half
        # cropped_frame = frame[:new_height, :, :]
        #
        # # Calculate the new height (half of the original height)
        # height, width, channels = cropped_frame.shape
        # new_width= height
        #
        # Resize the frame
        resized_frame = cv2.resize(frame, (256, 256))

        regression_img = infer_model(resized_frame,test_transforms,model,renderer,device, args)
        regression_img = (regression_img * 255).astype(np.uint8)

        final_img = np.concatenate([resized_frame, regression_img], axis=1)
        # Write the processed frame to the video file
        out.write(final_img)
        # cv2.imwrite('save_cam.jpg', final_img)
        # Display the frame
        cv2.imshow('Camera Feed', final_img)
        # # cv2.waitKey(3)
        # # cv2.waitKey(1)
        # #
        # # # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


