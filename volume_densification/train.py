import os
import sys
import argparse
import cv2
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.cpp_extension import load

sys.path.append("../")

from config import *
from dataset import NuScenesDataset
from utils.occvisualizer import OccVisualizer
from model.autoencoder import OccAutoencoder

occ_cuda = load("occ_cuda", sources=[os.path.join(ROOT, "libcpp/occ/occ.cpp"), os.path.join(ROOT, "libcpp/occ/occ.cu")], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])

class Trainer:
    def __init__(self, device, outdir, nusc_root, nusc_version, lr = 4e-4):
        # set up output
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        # set up data
        self.batch_size = 1 # should fix to 1
        self.device = device
        self.dataset = NuScenesDataset(nusc_root=nusc_root, nusc_version=nusc_version, istrain=True, verbose=True)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # configs
        self.grid = np.array([1, GRID_H, GRID_L, GRID_W]) # [T, H(Z), L(Y), W(X)]
        self.scale = GRID_SCALE
        self.offset = torch.from_numpy(np.array([self.grid[3], self.grid[2], self.grid[1]]) * self.scale / 2).to(device)
        # model
        self.model = OccAutoencoder(self.grid[1]).to(device)
        # training settings
        self.lr = lr
        self.lr_decay_step = 3
        self.lr_decay_rate = 0.6
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.lr_decay_step,
                                                         gamma=self.lr_decay_rate)

        # visualizer (optional)
        self.occviz = OccVisualizer(occ_cuda)
        self.viz_cam_idx = 0

    def tranform_pcld(self, pcld, transformation):
        B, N, _ = pcld.size()
        pcld = torch.concat([pcld[..., 0:3], torch.ones_like(pcld[..., 0].view(B, N, 1))], dim=2) # [B, N, 4]
        pcld_trans = torch.transpose(torch.bmm(transformation, torch.transpose(pcld, 1, 2)), 1, 2)
        pcld_trans = pcld_trans[..., 0:3] / pcld_trans[..., 3].view(B, N, 1)
        return pcld_trans

    def __call__(self, n_epoch=5):
        min_error = 100.0
        for epoch in range(n_epoch):
            print(f"Start training epoch {epoch} ...")
            error_list = []
            self.model.train()
            for i, data in enumerate(self.loader):
                # calculate pose
                cur_lidar_pose = data["lidar_pose"].to(self.device)
                ref_lidar_pose = data["ref_lidar_pose"].to(self.device)
                pose = torch.bmm(torch.inverse(ref_lidar_pose), cur_lidar_pose)
                # get point cloud
                pcld = data["lidar_pcld"].to(self.device)
                pcld = self.tranform_pcld(pcld, pose)
                # pcld to occ grid
                pcld = ((pcld + self.offset) / self.scale).to(torch.float32).contiguous()
                occ_in = occ_cuda.pcld2occ3d(pcld, self.grid).to(torch.float32) # [B, 45, 700, 700]
                # model predict occ
                occ_out = self.model(occ_in)
                # render and backprop
                origin = torch.zeros_like(pcld)
                origin = self.tranform_pcld(origin, pose)[:, 0, 0:3]
                origin = ((origin + self.offset) / self.scale).to(torch.float32).contiguous()
                pred_dist, gt_dist, grad_sigma = occ_cuda.render(occ_out.to(torch.float32), 
                                                                 origin.to(torch.float32), 
                                                                 pcld.to(torch.float32))
                invalid = torch.isnan(grad_sigma)
                grad_sigma[invalid] = 0.0
                invalid = torch.isnan(pred_dist)
                pred_dist[invalid] = 0.0
                gt_dist[invalid] = 0.0
                invalid = torch.isinf(pred_dist)
                pred_dist[invalid] = 0.0
                gt_dist[invalid] = 0.0

                self.optimizer.zero_grad()
                occ_out.backward(grad_sigma)
                self.optimizer.step()

                error = torch.mean(torch.abs(gt_dist - pred_dist))
                error_list.append(error.item())

                if i%20 == 0:
                    print(f"Epoch {epoch}, sample {i}/{len(self.dataset)}, current error: {sum(error_list)/len(error_list)}", end="\r")
            
            self.scheduler.step()

            # visualize last frame
            size = data["rgb_images"][0][self.viz_cam_idx].size()
            intrinsic = data["intrinsics"][0][self.viz_cam_idx]
            pose = torch.inverse(data["ref_lidar_pose"][0]) @ data["rgb_poses"][0][self.viz_cam_idx]
            avg_error = sum(error_list)/len(error_list)
            if min_error > avg_error:
                min_error = avg_error
                torch.save(self.model.state_dict(), os.path.join(self.outdir, "best.pth"))
            # save visualization
            viz_save_dir = os.path.join(self.outdir, "train_viz", f"viz_epoch_{'%04d' % epoch}_error_{avg_error}")
            os.makedirs(viz_save_dir, exist_ok=True)
            self.occviz.vizDistMap(os.path.join(viz_save_dir, f"in_occ.png"), 
                                    occ_in[0], size[0], size[1], intrinsic, pose, [self.grid[3], self.grid[2], self.grid[1]], self.scale)
            self.occviz.vizDistMap(os.path.join(viz_save_dir, f"out_occ.png"), 
                                    occ_out[0], size[0], size[1], intrinsic, pose, [self.grid[3], self.grid[2], self.grid[1]], self.scale)
            cv2.imwrite(os.path.join(viz_save_dir, f"rgb.png"), data["rgb_images"][0][self.viz_cam_idx].cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = "cuda", help='torch device, e.g. cpu, cuda, cuda:1, ...')
    parser.add_argument('--outdir', type=str, default = "./result", help='output directory')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epoch')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    trainer = Trainer(device, args.outdir, NUSC_ROOT, NUSC_TRAIN_VERSION, lr=args.lr)
    trainer(args.nepoch)