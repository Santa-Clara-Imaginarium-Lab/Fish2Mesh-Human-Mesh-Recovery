import os
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys

from PIL import Image
from torchvision import transforms
from scipy.stats import multivariate_normal
import skimage.io
import skimage.transform
import skimage.color
import skimage
from HMR_Model.EgoHMR import EgoHMR
from HMR_Model.EgoHMR_EgoPositionEmbedding import EgoHMR_pos
import pickle
import torch


import torch.multiprocessing as mp

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



def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-td", "--testing_Data", type=str, default='/media/imaginarium/2T/V3/valid/',
        help="testing dataset"
    )

    parser.add_argument(
        "-d", "--Training_Data", type=str, default='/media/imaginarium/2T/headset_frames_all/',
        help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=1000000, type=int, help="Number of epochs (default: %(default)s)", )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers", type=int, default=32, help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size", type=int, nargs=2, default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=70, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=70, help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path", type=str, default="./save/", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument("--clip_max_norm", default=1.0, type=float,
                        help="gradient clipping max norm (default: %(default)s")

    parser.add_argument("--checkpoint",
                        default="./save/283.ckpt",  # ./train0008/10.ckpt
                        type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args



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


class myDataset(Dataset):

    def __init__(self, root, transform):
        # self.df = pd.read_csv(root)
        self.clipTensor = []

        for pt in os.listdir(root):
            pt_path = os.path.join(root, pt)
            self.clipTensor.append(pt_path)

        self.transform = transform

    def __getitem__(self, index):
        spatial_feature_map_path = self.clipTensor[index]
        labels_folder = '/media/imaginarium/2T/new_name_label/'

        split_string = spatial_feature_map_path.split('/')
        pt_name = split_string[len(split_string) - 1]
        gt_name = pt_name.replace('.png','.pkl')

        # folder_path = '/'.join(split_string[:-2])
        #
        # gt_folder_path = os.path.join(folder_path, 'gt')
        gt_path = os.path.join(labels_folder, gt_name)

        with torch.no_grad():
            # spatial_feature_map = torch.load(spatial_feature_map_path, map_location=lambda storage, loc: storage)
            # spatial_feature_map = spatial_feature_map.view(243, 200, 192)
            # spatial_feature_map.requires_grad = False

            # Open the image
            image = Image.open(spatial_feature_map_path).convert('RGB')
            # Apply transformations
            transformed_image = self.transform(image)
            # print(gt_path)


            with open(gt_path, 'rb') as file:
                GT_file = pickle.load(file)
                GT_file = ensure_single_dimension(GT_file)
            # GT_file = 0
            # GT_npy = torch.from_numpy(np.array(np.load(gt_path), dtype='f'))
            # GT_npy.requires_grad = False
            # print(GT_npy.dtype)

        # mu + sigma * random_number

        # GT_npy = random_tensor

        return transformed_image, GT_file

    def __len__(self):
        return len(self.clipTensor)


def train_one_epoch(model, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    start = time.time()
    sample_num = 0

    for i, d in enumerate(train_dataloader):

        Images, GT_npy = d
        Images = Images.to(device)

        optimizer.zero_grad()
        sample_num += Images.shape[0]

        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229
        out = model(Images)
        out_global_orient = out['out_global_orient']
        out_body_pose = out['out_body_pose']
        out_betas = out['out_betas']
        out_pred_cam = out['out_pred_cam']
        pred_keypoints_3d = out['pred_keypoints_3d']
        pred_keypoints_2d = out['pred_keypoints_2d']
        pred_vertices = out['pred_vertices']


        GT_joints_3d = GT_npy['pred_keypoints_3d'].float().view(-1, 44, 3) / 1
        GT_joints_2d = GT_npy['pred_keypoints_2d'].float().view(-1, 44, 2) / 1
        GT_global_orient = GT_npy['pred_smpl_params']['global_orient'].float().view(-1, 1, 3, 3)
        GT_pose = GT_npy['pred_smpl_params']['body_pose'].float().view(-1, 23, 3, 3)
        GT_betas = GT_npy['pred_smpl_params']['betas'].float().view(-1, 10)
        GT_cam = GT_npy['pred_cam_t'].float().view(-1, 3)
        GT_vertices = GT_npy['pred_vertices'].float().view(-1,6890,3)




        loss_beta = torch.nn.MSELoss(reduction='mean')
        loss_pose = torch.nn.MSELoss(reduction='mean')
        loss_global_orient = torch.nn.L1Loss(reduction='mean')
        loss_3d_joints = torch.nn.L1Loss(reduction='mean')
        # loss_2d_joints = torch.nn.MSELoss(reduction='mean')
        loss_2d_joints = torch.nn.L1Loss(reduction='mean')
        loss_vertices = torch.nn.L1Loss(reduction='mean')
        loss_cam = torch.nn.MSELoss(reduction='mean')

        criterion_beta = loss_beta(out_betas.to(device), GT_betas.to(device))
        criterion_pose = loss_pose(out_body_pose.to(device), GT_pose.to(device))
        criterion_3d = loss_3d_joints(pred_keypoints_3d.to(device), GT_joints_3d.to(device))
        criterion_global_orient = loss_global_orient(out_global_orient.to(device), GT_global_orient.to(device))
        criterion_2d = loss_2d_joints(pred_keypoints_2d.to(device),GT_joints_2d.to(device))
        criterion_vertices = loss_vertices(pred_vertices.to(device),GT_vertices.to(device))
        criterion_cam = loss_cam(out_pred_cam.to(device), GT_cam.to(device))


        combined_loss = (3 * criterion_pose + 1 * criterion_beta + 2 * (2 * criterion_3d + criterion_global_orient + criterion_2d)
                          + 0.001 * criterion_cam + criterion_vertices)

        combined_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 1000 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(Images)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {combined_loss.item():.5f} |'
                f'\tbeta_Loss: {criterion_beta.item():.5f} |'
                f'\tpose_Loss: {criterion_pose.item():.5f} |'
                f'\t3d_Loss: {criterion_3d.item():.5f} |'
                f'\t2d_Loss: {criterion_2d.item():.5f} |'
                f'\tv_Loss: {criterion_vertices.item():.5f} |'
                f'\tcamera_loss: {loss_pose(out_pred_cam.to(device),GT_cam.to(device)).item():.4f} |'
                f"\ttime: {enc_time:.1f}"
            )
        # if i > 300:
        #     break


def validate_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    loss_sumbeta = AverageMeter()
    loss_sumpose = AverageMeter()
    loss_3d = AverageMeter()
    loss_2d = AverageMeter()
    loss_camera = AverageMeter()
    loss_v = AverageMeter()

    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            Images, GT_npy = d
            sample_num += Images.shape[0]
            # out_net = model(Images.to(device))

            # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
            # 9 + 207 + 10 + 3 = 229

            out = model(Images.to(device))
            out_global_orient = out['out_global_orient']
            out_body_pose = out['out_body_pose']
            out_betas = out['out_betas']
            out_pred_cam = out['out_pred_cam']
            pred_keypoints_3d = out['pred_keypoints_3d']
            pred_keypoints_2d = out['pred_keypoints_2d']
            pred_vertices = out['pred_vertices']

            GT_joints_3d = GT_npy['pred_keypoints_3d'].float().view(-1, 44, 3) / 1
            GT_joints_2d = GT_npy['pred_keypoints_2d'].float().view(-1, 44, 2) / 1
            GT_global_orient = GT_npy['pred_smpl_params']['global_orient'].float().view(-1, 1, 3, 3)
            GT_pose = GT_npy['pred_smpl_params']['body_pose'].float().view(-1, 23, 3, 3)
            GT_betas = GT_npy['pred_smpl_params']['betas'].float().view(-1, 10)
            GT_cam = GT_npy['pred_cam_t'].float().view(-1, 3)
            GT_vertices = GT_npy['pred_vertices'].float().view(-1, 6890, 3)

            loss_beta = torch.nn.MSELoss(reduction='mean')
            loss_pose = torch.nn.MSELoss(reduction='mean')
            loss_global_orient = torch.nn.L1Loss(reduction='mean')
            loss_3d_joints = torch.nn.L1Loss(reduction='mean')
            # loss_2d_joints = torch.nn.MSELoss(reduction='mean')
            loss_2d_joints = torch.nn.L1Loss(reduction='mean')
            loss_vertices = torch.nn.L1Loss(reduction='mean')
            loss_cam = torch.nn.L1Loss(reduction='mean')

            criterion_beta = loss_beta(out_betas.to(device), GT_betas.to(device))
            criterion_pose = loss_pose(out_body_pose.to(device), GT_pose.to(device))
            criterion_3d = loss_3d_joints(pred_keypoints_3d.to(device), GT_joints_3d.to(device))
            criterion_global_orient = loss_global_orient(out_global_orient.to(device), GT_global_orient.to(device))
            criterion_2d = loss_2d_joints(pred_keypoints_2d.to(device), GT_joints_2d.to(device))
            criterion_vertices = loss_vertices(pred_vertices.to(device), GT_vertices.to(device))
            criterion_cam = loss_cam(out_pred_cam.to(device), GT_cam.to(device))

            combined_loss = (3 * criterion_pose + 1 * criterion_beta + 2 * (
                        2 * criterion_3d + criterion_global_orient + criterion_2d)
                             + 0.001 * criterion_cam + criterion_vertices)

            loss.update(combined_loss)
            loss_sumbeta.update(criterion_beta)
            loss_sumpose.update(criterion_pose)
            loss_3d.update(criterion_3d)
            loss_2d.update(criterion_2d)
            loss_camera.update(criterion_cam)
            loss_v.update(criterion_vertices)
            # if sample_num > 1200:
            #     break

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.5f} |"
        f'\tbeta_Loss: {loss_sumbeta.avg:.5f} |'
        f'\tpose_Loss: {loss_sumpose.avg:.5f} |'
        f'\t3d_Loss: {loss_3d.avg:.5f} |'
        f'\t2d_Loss: {loss_2d.avg:.5f} |'
        f'\tv_Loss: {loss_v.avg:.5f} |'
        f'\tcamera_Loss: {loss_camera.avg:.5f} |'
    )
    return loss.avg


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([
                        transforms.Resize((224, 224)),  # Example: resize to 224x224
                        transforms.ToTensor(),          # Convert to a tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
                        ])


    # test_transforms = transforms.Compose([Resizer()])
    print('loading datasets')
    train_dataset = myDataset(args.Training_Data, train_transforms)
    test_dataset = myDataset(args.testing_Data, train_transforms)
    print('finish loading datasets')
    device = "cuda:1" if args.cuda and torch.cuda.is_available() else "cpu"

    net = EgoHMR_pos()
    net = net.to(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        # prefetch_factor=200
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=(device == "cuda"),
    )



    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=4)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]

        net.load_state_dict(new_state_dict)

        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(net, train_dataloader, optimizer,
                        epoch,
                        args.clip_max_norm,
                        )

        loss = validate_epoch(epoch, test_dataloader, net)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best and epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                args.save_path + str(epoch) + '.ckpt'
            )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    main(sys.argv[1:])
