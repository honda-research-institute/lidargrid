import os
import sys
import argparse
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.cpp_extension import load

from torch.utils.tensorboard import SummaryWriter

sys.path.append("../")

from config import *
from dataset import NuScenesPredictionDataset
from model.autoencoder import OccAutoencoder
from model.autoencoder3d import OccPredictor3d
from utils.occvisualizer import OccVisualizer

occ_cuda = load("occ_cuda", sources=[os.path.join(ROOT, "libcpp/occ/occ.cpp"), os.path.join(ROOT, "libcpp/occ/occ.cu")], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])
dvr = load("dvr", sources=[os.path.join(ROOT, "libcpp/dvr/dvr.cpp"), os.path.join(ROOT, "libcpp/dvr/dvr.cu")], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])

class PredictorTrainer:
    def __init__(self, device, outdir, 
                 n_input, n_output, 
                 nusc_root, nusc_version,
                 occ_encoder_pretrain="../pretrain/densify.pth"):
        # set up output
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        tensorboard_dir = os.path.join(self.outdir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=tensorboard_dir, comment="OccGrid Predictor Training")
        # set up data
        self.batch_size = 1
        self.device = device
        self.n_input = n_input
        self.n_output = n_output
        self.dataset = NuScenesPredictionDataset(nusc_root, nusc_version, n_input, n_output, istrain=True)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # configs
        self.grid = np.array([n_input, GRID_H, GRID_L, GRID_W]) # [T, H(Z), L(Y), W(X)]
        self.scale = GRID_SCALE
        self.offset = torch.from_numpy(np.array([self.grid[3], self.grid[2], self.grid[1]]) * self.scale / 2).to(device)
        # model
        self.occ_encoder = OccAutoencoder(channels=self.grid[1]*self.batch_size)
        self.occ_encoder.load_state_dict(torch.load(occ_encoder_pretrain, map_location="cpu"))
        self.occ_encoder = self.occ_encoder.to(device)
        self.occ_encoder.freeze()

        self.predictor = OccPredictor3d(n_input=n_input, n_output=n_output)
        self.predictor = self.predictor.to(device)

        # training settings
        self.lr = 2e-4
        self.lr_decay_step = 1
        self.lr_decay_rate = 0.7
        self.n_epoch = 50
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.lr_decay_step,
                                                         gamma=self.lr_decay_rate)
        
        # visualizer (optional)
        self.occviz = OccVisualizer(occ_cuda)
    
    def __call__(self):
        step = 1
        error_list = []
        for epoch in range(self.n_epoch):
            self.predictor.train()

            for i, dt in enumerate(self.loader):
                input_pcld = dt["input_pcld"].to(self.device)
                input_tindex = dt["input_tindex"].to(self.device)
                output_pcld = dt["output_pcld"].to(self.device)
                output_tindex = dt["output_tindex"].to(self.device)

                # to occ
                input_pcld = ((input_pcld + self.offset) / self.scale).to(torch.float32).contiguous()
                input_occ = dvr.init(input_pcld, input_tindex, self.grid).to(torch.float32)

                B, T, H, W, L = input_occ.size()
                input_occ = input_occ.view((B*T, H, W, L))
                input_occ = self.occ_encoder(input_occ)
                input_occ = input_occ.view((B, T, H, W, L))

                output_pcld = ((output_pcld + self.offset) / self.scale).to(torch.float32).contiguous()
                output_origins = dt["output_origin"].to(self.device)
                output_origins = ((output_origins + self.offset) / self.scale)

                pred_occ = self.predictor(input_occ)

                output_pcldmask = dt["output_pcldmask"].to(self.device).to(torch.float32).contiguous()

                pred_dist, gt_dist, grad_sigma = dvr.render(pred_occ.to(torch.float32).contiguous(), 
                                                            output_origins.to(torch.float32).contiguous(), 
                                                            output_pcld.to(torch.float32).contiguous(), 
                                                            output_tindex.to(torch.float32).contiguous(), 
                                                            output_pcldmask.to(torch.float32).contiguous(), "l1")

                invalid = torch.isnan(grad_sigma)
                grad_sigma[invalid] = 0.0
                invalid = torch.isnan(pred_dist)
                pred_dist[invalid] = 0.0
                gt_dist[invalid] = 0.0
                invalid = torch.isinf(pred_dist)
                pred_dist[invalid] = 0.0
                gt_dist[invalid] = 0.0

                self.optimizer.zero_grad()
                pred_occ.backward(grad_sigma)
                self.optimizer.step()

                error = torch.mean(torch.abs(gt_dist - pred_dist))
                error_list.append(error.item())

                if step % 20 == 0:
                    self.log_writer.add_scalar("Loss/L1 Dist Error", sum(error_list)/len(error_list), step)
                    print(f"Epoch {epoch}, Step {step}/{len(self.dataset)}, l1 dist error: {sum(error_list)/len(error_list)}...", end='\r')
                    error_list = []

                if step % 2000 == 0:
                    self.scheduler.step()
                    torch.save(self.predictor.state_dict(), os.path.join(self.outdir, "best.pth"))

                step += 1

    def visualize(self, occ, data, t=0, cam_id=0, mode="input"):
        # check
        assert t>=0 and t < self.n_input if mode=="input" else self.n_output
        assert cam_id >=0 and cam_id < 6
        assert mode in ["input", "output"]
        # visualize
        img_h, img_w, _ = data[f"{mode}_rgbs"][0][t][cam_id].size()
        intrinsics = data[f"{mode}_intrinsics"][0][t][cam_id]
        rgb_pose = data[f"{mode}_rgb_poses"][0][t][cam_id]
        self.occviz(occ[0][t], img_h, img_w, intrinsics, rgb_pose, 
                    grid=[self.grid[3], self.grid[2], self.grid[1]], scale=self.scale)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = "cuda", help='torch device, e.g. cpu, cuda, cuda:1, ...')
    parser.add_argument('--outdir', type=str, default = "./forecasting_result", help='output directory')
    parser.add_argument('--densify_pretrain', type=str, default = "../pretrain/densify.pth", help='path to pretrained densification network')
    parser.add_argument('--horizon', type=int, default = 2, help='number of input/output frames')
    return parser.parse_args()

if __name__  == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    trainer = PredictorTrainer(
        device=device, 
        outdir=args.outdir, 
        n_input=args.horizon, 
        n_output=args.horizon, 
        nusc_root=NUSC_ROOT, 
        nusc_version=NUSC_TRAIN_VERSION,
        occ_encoder_pretrain=args.densify_pretrain)
    trainer()