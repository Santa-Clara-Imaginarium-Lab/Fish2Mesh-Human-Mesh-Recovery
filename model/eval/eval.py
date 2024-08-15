import os
import argparse
import random
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
from model.HMR_Model.EgoHMR_EgoPositionEmbedding import EgoHMR_pos
from model.util.smpl_wrapper import SMPL
from model.util.geometry import *

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ensure_single_dimension(gt_files):
    def process_tensor(tensor):
        if tensor.ndim > 1 and tensor.shape[0] > 1:
            return tensor[0:1]
        return tensor

    def process_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                process_dict(value)
            elif isinstance(value, torch.Tensor):
                d[key] = process_tensor(value)

    process_dict(gt_files)
    return gt_files


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test script for trained model.")
    parser.add_argument(
        "-td", "--testing_Data", type=str, default="/media/imaginarium/2T/V2/test/",
        help="Path to the testing dataset."
    )
    parser.add_argument(
        "--batch-size", type=int, default=70, help="Batch size for testing (default: %(default)s)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of workers for data loading (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--checkpoint", type=str, default="../save/68.ckpt",
        help="Path to the saved checkpoint file."
    )

    args = parser.parse_args(argv)
    return args

class myDataset(Dataset):
    def __init__(self, root, transform):
        self.clipTensor = []
        for pt in os.listdir(root):
            pt_path = os.path.join(root, pt)
            self.clipTensor.append(pt_path)
        self.transform = transform

    def __getitem__(self, index):
        spatial_feature_map_path = self.clipTensor[index]
        labels_folder = "/media/imaginarium/2T/new_name_label/"
        pt_name = os.path.basename(spatial_feature_map_path)
        gt_name = pt_name.replace('.png', '.pkl')
        gt_path = os.path.join(labels_folder, gt_name)

        with torch.no_grad():
            image = Image.open(spatial_feature_map_path).convert('RGB')
            transformed_image = self.transform(image)

            with open(gt_path, 'rb') as file:
                GT_file = pickle.load(file)
                GT_file = ensure_single_dimension(GT_file)

        return transformed_image, GT_file

    def __len__(self):
        return len(self.clipTensor)

def test_epoch(test_dataloader, model, smpl_model, device):
    model.eval()
    loss = AverageMeter()
    loss_sumbeta = AverageMeter()
    loss_sumpose = AverageMeter()
    loss_3d = AverageMeter()
    loss_cam = AverageMeter()
    loss_MPJPE = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            Images, GT_npy = d
            Images = Images.to(device)

            out_net = model(Images)

            out_global_orient = out_net[:, :, 0:3 * 3].view(-1, 1, 3, 3)
            out_body_pose_global_orient = out_net[:, :, 3 * 3: 3 * 3 + 23 * 3 * 3].view(-1, 23, 3, 3)
            out_betas = out_net[:, :, 3 * 3 + 23 * 3 * 3: 10 + 3 * 3 + 23 * 3 * 3].view(-1, 10)
            out_pred_cam = out_net[:, :, 10 + 3 * 3 + 23 * 3 * 3:].view(-1, 3)

            GT_joints_3d = GT_npy['pred_keypoints_3d'].float().view(-1, 44, 3)
            GT_pose = GT_npy['pred_smpl_params']['global_orient'].float().view(-1, 1, 3, 3)
            GT_global_orient = GT_npy['pred_smpl_params']['body_pose'].float().view(-1, 23, 3, 3)
            GT_betas = GT_npy['pred_smpl_params']['betas'].float().view(-1, 10)
            GT_cam = GT_npy['pred_cam'].float().view(-1, 3)

            smpl_output = smpl_model(global_orient=out_global_orient, body_pose=out_body_pose_global_orient, betas=out_betas, pose2rot=False)
            pred_keypoints_3d = smpl_output.joints

            loss_beta = torch.nn.MSELoss(reduction='mean')
            loss_pose = torch.nn.MSELoss(reduction='mean')
            loss_global_orient = torch.nn.L1Loss(reduction='mean')
            loss_3d_joints = torch.nn.L1Loss(reduction='mean')

            out_criterion_beta = loss_beta(out_betas.to(device), GT_betas.to(device))
            out_criterion_pose = loss_pose(out_body_pose_global_orient.to(device), GT_pose.to(device))
            out_criterion_3d_joints = loss_3d_joints(pred_keypoints_3d.to(device), GT_joints_3d.to(device))
            out_criterion_global_orient = loss_global_orient(out_global_orient.to(device), GT_global_orient.to(device))

            combined_loss = (0.03 * out_criterion_pose + 0.01 * out_criterion_beta + 0.2 * (
                        out_criterion_3d_joints + out_criterion_global_orient) + 0.1 * loss_pose(out_pred_cam.to(device), GT_cam.to(device)))

            MPJPE_loss = mpjpe_cal(pred_keypoints_3d.to(device),GT_joints_3d.to(device))
            loss_MPJPE.update(MPJPE_loss)

            loss.update(combined_loss)
            loss_sumbeta.update(out_criterion_beta)
            loss_sumpose.update(out_criterion_pose)
            loss_3d.update(out_criterion_3d_joints)
            loss_cam.update(loss_pose(out_pred_cam.to(device), GT_cam.to(device)))

    print(
        f"Test: Average losses:"
        f"\tLoss: {loss.avg:.7f} |"
        f'\tbeta_Loss: {loss_sumbeta.avg:.7f} |'
        f'\tpose_Loss: {loss_sumpose.avg:.7f} |'
        f'\t3d_Loss: {loss_3d.avg:.7f} |'
        f'\tcamera_Loss: {loss_cam.avg:.7f} |'
        f'\tMPJPE_Loss: {loss_MPJPE.avg:.7f} |'
    )

def main(argv):
    args = parse_args(argv)

    # Specify which GPU to use, if you have multiple GPUs
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    print('Loading dataset...')
    test_dataset = myDataset(args.testing_Data, test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print('Dataset loaded.')

    net = EgoHMR_pos()
    net = net.to(device)

    SMPL_CONFIG = {'data_dir': '/home/imaginarium/.cache/4DHumans/data/',
                   'model_path': '/home/imaginarium/.cache/4DHumans/data//smpl',
                   'gender': 'neutral',
                   'num_body_joints': 23,
                   'joint_regressor_extra': '/home/imaginarium/.cache/4DHumans/data//SMPL_to_J19.pkl',
                   'mean_params': '/home/imaginarium/.cache/4DHumans/data//smpl_mean_params.npz'}
    smpl_model = SMPL(**SMPL_CONFIG).to(device)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        print("Checkpoint loaded.")

    test_epoch(test_dataloader, net, smpl_model, device)

if __name__ == '__main__':
    main(sys.argv[1:])
    #update
