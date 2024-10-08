import os
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset
# from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys

from PIL import Image
# import pandas as pd
from torchvision import transforms
from scipy.stats import multivariate_normal
import skimage.io
import skimage.transform
import skimage.color
import skimage
from EgoHMR import EgoHMR
import pickle
from geometry import *
from smpl_wrapper import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def init_process(
        rank, # rank of the process
        args,
        world_size, # number of workers
        fn, # function to be run
        # backend='gloo',# good for single node
        # backend='nccl' # the best for CUDA
        backend='nccl'
    ):
    # information used for rank 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(rank, args,  world_size)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

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

SMPL_CONFIG = {'data_dir': '/home/imaginarium/.cache/4DHumans/data/',
                   'model_path': '/home/imaginarium/.cache/4DHumans/data//smpl',
                   'gender': 'neutral',
                   'num_body_joints': 23,
                   'joint_regressor_extra': '/home/imaginarium/.cache/4DHumans/data//SMPL_to_J19.pkl',
                   'mean_params': '/home/imaginarium/.cache/4DHumans/data//smpl_mean_params.npz'}


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

    # parser.add_argument(
    #     "-cd", "--contextDataset", type=str,
    #     default='D:/Tianma/dataset/Pre_process/train_tensor/',
    #     help="Training dataset"
    # )

    parser.add_argument(
        "-td", "--testing_Data", type=str, default='/media/imaginarium/12T/Dataset/validation80/feature_maps',
        help="testing dataset"
    )

    parser.add_argument(
        "-d", "--Training_Data", type=str, default='/media/imaginarium/12T/Dataset/train80/feature_maps',
        help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=1000000, type=int, help="Number of epochs (default: %(default)s)", )

    # you need this argument in your scripts for DDP to work
    parser.add_argument('--local-rank', type=int, default=-1, help='Local process rank')

    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers", type=int, default=1, help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size", type=int, nargs=2, default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1, help="Test batch size (default: %(default)s)",
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
                        default="./save/7.ckpt",  # ./train0008/10.ckpt
                        type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=256, max_side=256):
        # image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 256 - rows
        pad_h = 256 - cols

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # annots *= scale

        return torch.from_numpy(new_image), scale


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

        split_string = spatial_feature_map_path.split('/')
        pt_name = split_string[len(split_string) - 1]
        gt_name = pt_name.replace('.pt','.pkl')

        folder_path = '/'.join(split_string[:-2])

        gt_folder_path = os.path.join(folder_path, 'gt')
        gt_path = os.path.join(gt_folder_path, gt_name)

        with torch.no_grad():
            spatial_feature_map = torch.load(spatial_feature_map_path, map_location=lambda storage, loc: storage)
            # spatial_feature_map = spatial_feature_map.view(243, 200, 192)
            spatial_feature_map.requires_grad = False

            with open(gt_path, 'rb') as file:
                GT_file = pickle.load(file)
            # GT_npy = torch.from_numpy(np.array(np.load(gt_path), dtype='f'))
            # GT_npy.requires_grad = False
            # print(GT_npy.dtype)

        # mu + sigma * random_number

        # GT_npy = random_tensor

        return spatial_feature_map, GT_file

    def __len__(self):
        return len(self.clipTensor)


def train_one_epoch(model, train_dataloader, optimizer, epoch, clip_max_norm, device):
    smpl_model = SMPL(**SMPL_CONFIG).to(device)
    model.train()
    # device = next(model.parameters()).device
    # device = torch.device(f"cuda:{dist.get_rank()}")
    start = time.time()
    # accu_num = torch.zeros(1).to(device)
    sample_num = 0
    train_dataloader.sampler.set_epoch(epoch)

    for i, d in enumerate(train_dataloader):

        Images, GT_npy = d

        # random_tensor = GT_npy + 0 * torch.randn_like(GT_npy) * optimizer.param_groups[0]['lr']

        # Set the first row to zero
        # random_tensor[:, 0] = 0
        # random_tensor[:, 0, :] = 0

        # srcGT = random_tensor

        Images = Images.to(device)
        # srcGT = srcGT.to(device)
        # GT_npy = GT_npy.to(device)
        optimizer.zero_grad()
        sample_num += Images.shape[0]

        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229
        out_net = model(Images)
        out_global_orient = out_net[:,:, 0:3*3].view(-1,1,3,3)
        # includes out_global_orient
        out_body_pose_global_orient = out_net[:,:, 0 : 3*3 + 23 * 3 * 3].view(-1, 24, 3, 3)

        out_body_pose = out_net[:, :, 3 * 3: 3 * 3 + 23 * 3 * 3].view(-1, 23, 3, 3)
        out_betas = out_net[:,:, 3*3 + 23 * 3 * 3: 10 + 3 * 3 + 23 * 3 * 3].view(-1, 10)
        out_pred_cam = out_net[:,:, 10 + 3 * 3 + 23 * 3 * 3: ].view(-1, 3)

        GT_joints_3d = GT_npy['joints_3d'].float().view(-1, 44, 3) / 1
        GT_joints_2d = GT_npy['joints_2d'].float().view(-1, 44, 2) / 1
        GT_pose = GT_npy['pose'].float().view(-1, 3)
        GT_betas = GT_npy['shape'].float().view(-1, 10)
        GT_cam = GT_npy['trans'].float().view(-1, 3)

        # convert (-1,3) -> (-1, 24, 3, 3)
        GT_pose = aa_to_rotmat(GT_pose).view(-1, 24, 3, 3)

        # regression SMPL->joints
        smpl_output = smpl_model(**{'global_orient':out_global_orient,'body_pose':out_body_pose,'betas':out_betas}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        # pred_vertices = smpl_output.vertices

        # focal_length = 5000 * torch.ones(Images.shape[0], 2, device=device, dtype=torch.float32)
        # focal_length = focal_length.reshape(-1, 2)
        # pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=out_pred_cam, focal_length=focal_length / 256)


        loss_beta = torch.nn.MSELoss(reduction='mean')
        loss_pose = torch.nn.MSELoss(reduction='mean')
        # loss_3d_joints = torch.nn.MSELoss(reduction='mean')
        loss_3d_joints = torch.nn.L1Loss(reduction='mean')
        # loss_2d_joints = torch.nn.MSELoss(reduction='mean')
        # loss_2d_joints = torch.nn.L1Loss(reduction='mean')

        out_criterion_beta = loss_beta(out_betas.to(device), GT_betas.to(device))
        out_criterion_pose = loss_pose(out_body_pose_global_orient.to(device), GT_pose.to(device))
        out_criterion_3d_joints = loss_3d_joints(pred_keypoints_3d.to(device), GT_joints_3d.to(device))
        # out_criterion_2d_joints = loss_2d_joints(pred_keypoints_2d.to(device),GT_joints_2d.to(device))


        combined_loss = (0.03 * out_criterion_pose + 0.01 * out_criterion_beta + 0.2 * out_criterion_3d_joints
                          + 0.1 * loss_pose(out_pred_cam.to(device),GT_cam.to(device)))
        # combined_loss = combined_loss.float()

        # combined_loss = the * out_criterion_pose_first

        combined_loss.backward()
        # average gradient as DDP doesn't do it correctly
        # average_gradients(model)

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 2000 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(Images)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {combined_loss.item():.7f} |'
                f'\tbeta_Loss: {out_criterion_beta.item():.7f} |'
                f'\tpose_Loss: {out_criterion_pose.item():.7f} |'
                f'\t3d_Loss: {out_criterion_3d_joints.item():.7f} |'
                # f'\t2d_Loss: {out_criterion_2d_joints.item():.7f} |'
                f'\tcamera_loss: {loss_pose(out_pred_cam.to(device),GT_cam.to(device)).item():.4f} |'
                f"\ttime: {enc_time:.1f}"
            )
        # if i > 300:
        #     break


def validate_epoch(epoch, test_dataloader, model, device):
    model.eval()
    smpl_model = SMPL(**SMPL_CONFIG).to(device)
    # device = next(model.parameters()).device
    # device = torch.device(f"cuda:{dist.get_rank()}")

    loss = AverageMeter()
    loss_sumbeta = AverageMeter()
    loss_sumpose = AverageMeter()
    loss_3d = AverageMeter()
    loss_2d = AverageMeter()
    loss_cam = AverageMeter()
    # loss_function = torch.nn.MSELoss(reduction='mean')
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0
    test_dataloader.sampler.set_epoch(epoch)

    with torch.no_grad():
        for d in test_dataloader:
            Images, GT_npy = d
            sample_num += Images.shape[0]
            out_net = model(Images.to(device))

            # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
            # 9 + 207 + 10 + 3 = 229

            out_global_orient = out_net[:, :, 0:3 * 3].view(-1, 1, 3, 3)
            # includes out_global_orient
            out_body_pose_global_orient = out_net[:, :, 0: 3 * 3 + 23 * 3 * 3].view(-1, 24, 3, 3)

            out_body_pose = out_net[:, :, 3 * 3: 3 * 3 + 23 * 3 * 3].view(-1, 23, 3, 3)
            out_betas = out_net[:, :, 3 * 3 + 23 * 3 * 3: 10 + 3 * 3 + 23 * 3 * 3].view(-1, 10)
            out_pred_cam = out_net[:, :, 10 + 3 * 3 + 23 * 3 * 3:].view(-1, 3)

            GT_joints_3d = GT_npy['joints_3d'].float().view(-1, 44, 3) / 1
            GT_joints_2d = GT_npy['joints_2d'].float().view(-1, 44, 2) / 1
            GT_pose = GT_npy['pose'].float().view(-1, 3)
            GT_betas = GT_npy['shape'].float().view(-1, 10)
            GT_cam = GT_npy['trans'].float().view(-1, 3)

            # convert (-1,3) -> (-1, 24, 3, 3)
            GT_pose = aa_to_rotmat(GT_pose).view(-1, 24, 3, 3)

            # regression SMPL->joints
            # smpl_model = SMPL(**SMPL_CONFIG).to(device)
            smpl_output = smpl_model(
                **{'global_orient': out_global_orient, 'body_pose': out_body_pose, 'betas': out_betas}, pose2rot=False)
            pred_keypoints_3d = smpl_output.joints
            # pred_vertices = smpl_output.vertices

            # focal_length = 5000 * torch.ones(Images.shape[0], 2, device=device, dtype=torch.float32)
            # focal_length = focal_length.reshape(-1, 2)
            # pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=out_pred_cam,
            #                                            focal_length=focal_length / 256)

            loss_beta = torch.nn.MSELoss(reduction='mean')
            loss_pose = torch.nn.MSELoss(reduction='mean')
            # loss_3d_joints = torch.nn.MSELoss(reduction='mean')
            loss_3d_joints = torch.nn.L1Loss(reduction='mean')
            # loss_2d_joints = torch.nn.MSELoss(reduction='mean')
            # loss_2d_joints = torch.nn.L1Loss(reduction='mean')

            out_criterion_beta = loss_beta(out_betas.to(device), GT_betas.to(device))
            out_criterion_pose = loss_pose(out_body_pose_global_orient.to(device), GT_pose.to(device))
            out_criterion_3d_joints = loss_3d_joints(pred_keypoints_3d.to(device), GT_joints_3d.to(device))
            # out_criterion_2d_joints = loss_2d_joints(pred_keypoints_2d.to(device), GT_joints_2d.to(device))

            combined_loss = (0.03 * out_criterion_pose + 0.01 * out_criterion_beta + 0.2 * out_criterion_3d_joints
                              + 0.1 * loss_pose(out_pred_cam.to(device), GT_cam.to(device)))

            loss.update(combined_loss)
            loss_sumbeta.update(out_criterion_beta)
            loss_sumpose.update(out_criterion_pose)
            loss_3d.update(out_criterion_3d_joints)
            # loss_2d.update(out_criterion_2d_joints)
            loss_cam.update(loss_pose(out_pred_cam.to(device), GT_cam.to(device)))
            # if sample_num > 1200:
            #     break

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.7f} |"
        f'\tbeta_Loss: {loss_sumbeta.avg:.7f} |'
        f'\tpose_Loss: {loss_sumpose.avg:.7f} |'
        f'\t3d_Loss: {loss_3d.avg:.7f} |'
        # f'\t2d_Loss: {loss_2d.avg:.7f} |'
        f'\tcamera_Loss: {loss_cam.avg:.7f} |'
    )
    return loss.avg


def main(rank, argv, world_size):


    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([])


    # test_transforms = transforms.Compose([Resizer()])
    print('loading datasets')
    train_dataset = myDataset(args.Training_Data, train_transforms)
    test_dataset = myDataset(args.testing_Data, train_transforms)
    print('finish loading datasets')
    # device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # torch.cuda.set_device(rank)
    # torch.cuda.empty_cache()
    # device = torch.device(f"cuda:{rank}")
    # world_size = dist.get_world_size()
    # # Initialize distributed training
    # if args.local_rank != -1:
    #     dist.init_process_group(backend='nccl',
    #                             init_method='env://',
    #                             world_size=world_size,
    #                             rank=rank)

    net = EgoHMR()
    net = net.to(rank)
    net = DDP(
        net,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True
    )

    # initialize the DistributedSampler
    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size)
    test_sampler = DistributedSampler(test_dataset,num_replicas=world_size)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=(device == "cuda"),
        sampler=train_sampler
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=test_sampler
        # pin_memory=(device == "cuda"),
    )



    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=4)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]
        # # Use a barrier() to make sure that process 1 loads the model after process
        # # 0 saves it.
        # dist.barrier()
        # # configure map_location properly
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # # ddp_model.load_state_dict(
        # #     torch.load(CHECKPOINT_PATH, map_location=map_location))

        net.load_state_dict(new_state_dict)

        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # let all processes sync up before starting with a new epoch of training
        # dist.barrier()

        train_one_epoch(net, train_dataloader, optimizer,
                        epoch,
                        args.clip_max_norm,
                        rank
                        )



        loss = validate_epoch(epoch, test_dataloader, net, rank)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best and epoch % 1 == 0 and rank == 0:
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
    n_gpus = torch.cuda.device_count()
    print('find the GPUs:', n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    mp.spawn(main,
             args=(sys.argv[1:], world_size,),
             nprocs=world_size,
             join=True)
    # world_size = 2
    # processes = []
    # mp.set_start_method("spawn")
    # for rank in range(world_size):
    #     p = mp.Process(target=init_process, args=(rank, sys.argv[1:], world_size, main))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    # # main(sys.argv[1:])
