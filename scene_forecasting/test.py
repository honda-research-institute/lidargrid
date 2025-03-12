#  Copyright (c) Honda Research Institute USA Inc.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import argparse
from datetime import datetime

import cv2
import tqdm
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.cpp_extension import load

sys.path.append("../")

from config import *
from dataset import NuScenesPredictionDataset
from model.autoencoder import OccAutoencoder
from model.autoencoder3d import OccPredictor3d
from utils.occvisualizer import OccVisualizer
from utils.evaluation import get_rendered_pcds, get_clamped_output, compute_chamfer_distance, compute_ray_errors, compute_chamfer_distance_inner

from matplotlib import cm

occ_cuda = load("occ_cuda", sources=[os.path.join(ROOT, "libcpp/occ/occ.cpp"), os.path.join(ROOT, "libcpp/occ/occ.cu")], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])
dvr = load("dvr", sources=[os.path.join(ROOT, "libcpp/dvr/dvr.cpp"), os.path.join(ROOT, "libcpp/dvr/dvr.cu")], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])

class LogWriter:
    def __init__(self, filename, verbose=True):
        self.file = open(filename, "w")
        self.verbose = verbose

    def write(self, text, end='\n', flush=True):
        if self.verbose:
            print(text, end=end)
        self.file.write(text + end)
        if flush:
            self.file.flush()

class PredictorTest:
    def __init__(self, device, pretrain_model, 
                 n_input, n_output, 
                 nusc_root, nusc_version,
                 occ_encoder_pretrain="./pretrain/densify.pth"):
        # set up output
        self.outdir = os.path.join(os.path.dirname(pretrain_model), "test")
        os.makedirs(self.outdir, exist_ok=True)
        self.logwriter = LogWriter(os.path.join(self.outdir, "log.txt"))
        # set up data
        self.batch_size = 1
        self.device = device
        self.n_input = n_input
        self.n_output = n_output
        self.dataset = NuScenesPredictionDataset(nusc_root, nusc_version, n_input, n_output, istrain=False)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        # configs
        self.grid = np.array([n_input, GRID_H, GRID_L, GRID_W]) # [T, H(Z), L(Y), W(X)]
        self.scale = GRID_SCALE
        self.offset = torch.from_numpy(np.array([self.grid[3], self.grid[2], self.grid[1]]) * self.scale / 2).to(device)
        self.PC_RANGE = np.array([-GRID_W, -GRID_L, -GRID_H, GRID_W, GRID_L, GRID_H]) * GRID_SCALE / 2
        # model
        self.occ_encoder = OccAutoencoder(channels=self.grid[1]*self.batch_size)
        self.occ_encoder.load_state_dict(torch.load(occ_encoder_pretrain, map_location="cpu"))
        self.occ_encoder = self.occ_encoder.to(device)
        self.occ_encoder.freeze()

        self.pretrain_model_path = pretrain_model
        params = torch.load(self.pretrain_model_path, map_location="cpu")
        if 'model_state_dict' in params:
            params = params['model_state_dict']

        self.predictor = OccPredictor3d(n_input=n_input, n_output=n_output)
        self.predictor.load_state_dict(params, strict=False)
        self.predictor = self.predictor.to(device)
        
        # visualizer (optional)
        self.occviz = OccVisualizer(occ_cuda)

        # write log
        self.logwriter.write("="*50)
        self.logwriter.write(f"Start testing at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        self.logwriter.write(f"Dataset Root:  {self.dataset.nusc_root}")
        self.logwriter.write(f"Dataset Version:  {self.dataset.nusc_version}")
        self.logwriter.write(f"number of input:  {n_input}")
        self.logwriter.write(f"number of output:  {n_output}")
        self.logwriter.write(f"Occ Grid:  {self.grid.tolist()}")
        self.logwriter.write(f"Occ Scale:  {self.scale}")
        self.logwriter.write(f"Occ Offset:  {self.offset.cpu().numpy().tolist()}")
        self.logwriter.write(f"PC Range:  {self.PC_RANGE}")
        self.logwriter.write(f"Occ Densify Encoder Pretrain:  {occ_encoder_pretrain}")
        self.logwriter.write(f"Occ Predictor Pretrain:  {self.pretrain_model_path}")
        self.logwriter.write(f"Dataset Class:  {self.dataset.__class__.__name__}")
        self.logwriter.write(f"Occ Predictor Class:  {self.predictor.__class__.__name__}")
        self.logwriter.write("="*50)

    
    def __call__(self, densify=True, visualize=False):
        self.logwriter.write(f"Start testing, using densification: {densify}, visualize: {visualize}...")
        self.predictor.eval()

        metrics = {
            "count": 0.0,
            "chamfer_distance": 0.0,
            "chamfer_distance_inner": 0.0,
            "l1_error": 0.0,
            "absrel_error": 0.0
        }

        for i, dt in tqdm.tqdm(enumerate(self.loader)):
            input_pcld = dt["input_pcld"].to(self.device)
            input_tindex = dt["input_tindex"].to(self.device)
            output_pcld = dt["output_pcld"].to(self.device)
            output_tindex = dt["output_tindex"].to(self.device)

            # to occ
            input_occ_pcld = ((input_pcld + self.offset) / self.scale).to(torch.float32).contiguous()
            input_occ = dvr.init(input_occ_pcld, input_tindex, self.grid).to(torch.float32)
            if densify:
                B, T, H, W, L = input_occ.size()
                input_occ = input_occ.view((B*T, H, W, L))
                input_occ = self.occ_encoder(input_occ)
                input_occ = input_occ.view((B, T, H, W, L))

            output_occ_pcld = ((output_pcld + self.offset) / self.scale).to(torch.float32).contiguous()
            output_origins = dt["output_origin"].to(self.device)
            output_occ_origins = ((output_origins + self.offset) / self.scale)

            pred_occ = self.predictor(input_occ)

            output_pcldmask = dt["output_pcldmask"].to(self.device).to(torch.float32).contiguous()

            pred_dist, gt_dist, grad_sigma = dvr.render(pred_occ.to(torch.float32).contiguous(), 
                                                        output_occ_origins.to(torch.float32).contiguous(), 
                                                        output_occ_pcld.to(torch.float32).contiguous(), 
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

            # evaluate error
            pred_dist *= self.scale
            gt_dist *= self.scale

            pred_pcds = get_rendered_pcds(output_origins[0].cpu().numpy(),
                                          output_pcld[0].cpu().numpy(),
                                          output_tindex[0].cpu().numpy(),
                                          gt_dist[0].cpu().numpy(),
                                          pred_dist[0].cpu().numpy(),
                                          False,
                                          PC_RANGE=self.PC_RANGE)
            
            gt_pcds = get_clamped_output(output_origins[0].cpu().numpy(),
                                         output_pcld[0].cpu().numpy(),
                                         output_tindex[0].cpu().numpy(),
                                         gt_dist[0].cpu().numpy(),
                                         False,
                                         PC_RANGE=self.PC_RANGE)
            
            # calculate evaluation metrics
            count, chamfer_distance, chamfer_distance_inner, l1_errors, absrel_errors = 0.0, 0.0, 0.0, 0.0, 0.0
            for k in range(len(gt_pcds)): # n_output frames
                pred_pcd = pred_pcds[k] # [n_points, 3]
                gt_pcd = gt_pcds[k] # [n_points, 3]
                origin = output_origins[0][k].cpu().numpy()

                # get the metrics
                count += 1
                chamfer_distance += compute_chamfer_distance(pred_pcd, gt_pcd, device)
                chamfer_distance_inner += compute_chamfer_distance_inner(pred_pcd, gt_pcd, device, PC_RANGE=self.PC_RANGE)
                l1_error, absrel_error = compute_ray_errors(pred_pcd, gt_pcd, torch.from_numpy(origin), device, PC_RANGE=self.PC_RANGE)
                l1_errors += l1_error
                absrel_errors += absrel_error
                
            metrics["count"] += count
            metrics["chamfer_distance"] += chamfer_distance
            metrics["chamfer_distance_inner"] += chamfer_distance_inner
            metrics["l1_error"] += l1_errors
            metrics["absrel_error"] += absrel_errors

            self.logwriter.write(f"Sample No. {i}, chamfer distance: {chamfer_distance/count}, chamfer distance inner: {chamfer_distance_inner/count}, l1 error: {l1_errors/count}, absrel_error: {absrel_errors/count}")

            if visualize:
                save_dir = os.path.join(self.outdir, "visualize", f"data_{'%04d' % i}")
                self.visualize(input_occ, dt, save_dir, mode="input")
                self.visualize(pred_occ, dt, save_dir, mode="output")

                self.visualize_pcld(input_pcld, input_tindex, dt, save_dir, mode="input")
                self.visualize_pcld(output_pcld, output_tindex, dt, save_dir, mode="output")

        self.logwriter.write("==========================Final Results============================")
        self.logwriter.write(f'Final Chamfer Distance: {metrics["chamfer_distance"] / metrics["count"]}')
        self.logwriter.write(f'Final Chamfer Distance Inner: {metrics["chamfer_distance_inner"] / metrics["count"]}')
        self.logwriter.write(f'Final L1 Error: {metrics["l1_error"] / metrics["count"]}')
        self.logwriter.write(f'Final AbsRel Error: {metrics["absrel_error"] / metrics["count"]}')
        self.logwriter.write("===================================================================")

    def visualize(self, occ, data, outdir, mode="input"):
        # check
        assert mode in ["input", "output"]
        n_frame = self.n_input if mode=="input" else self.n_output

        for cam_id in range(6):
            save_dir = os.path.join(outdir, mode, f"cam_{'%04d' % cam_id}")
            os.makedirs(save_dir, exist_ok=True)
            for t in range(n_frame):
                img = data[f"{mode}_rgbs"][0][t][cam_id].cpu().numpy()
                img_h, img_w, _ = img.shape
                intrinsics = data[f"{mode}_intrinsics"][0][t][cam_id]
                rgb_pose = data[f"{mode}_rgb_poses"][0][t][cam_id]
                self.occviz.vizDepthColorMap(os.path.join(save_dir, f"depth_{'%04d' % t}.png"), occ[0][t], img_h, img_w, intrinsics, rgb_pose, 
                                        grid=[self.grid[3], self.grid[2], self.grid[1]], scale=self.scale)
                cv2.imwrite(os.path.join(save_dir, f"rgb_{'%04d' % t}.png"), img)

    def visualize_pcld(self, pcld, tindex, data, outdir, mode="input"):
        pcld = pcld[0].cpu().numpy()
        tindex = tindex[0].cpu().numpy()
        n_frame = int(np.max(tindex))+1

        cmap = cm.get_cmap('jet')

        for t in range(n_frame):
            pts_index = np.where((tindex == t))[0]
            for cam_id in range(6):
                _pcld = pcld[pts_index]
                save_dir = os.path.join(outdir, mode, f"cam_{'%04d' % cam_id}")
                os.makedirs(save_dir, exist_ok=True)
                img = data[f"{mode}_rgbs"][0][t][cam_id].cpu().numpy()
                img_h, img_w, _ = img.shape
                intrinsics = data[f"{mode}_intrinsics"][0][t][cam_id]
                rgb_pose = data[f"{mode}_rgb_poses"][0][t][cam_id]
                # pcld2depth
                _pcld = np.concatenate([_pcld, np.expand_dims(np.ones_like(_pcld[:, 0]), axis=1)], axis=1)
                _pcld = (np.linalg.inv(rgb_pose) @ _pcld.T).T
                proj_pcld = (intrinsics[0:3, 0:3] @ _pcld[:, 0:3].T).T
                depth = proj_pcld[:, 2]
                pixels = proj_pcld[:, 0:2] / np.expand_dims(proj_pcld[:, 2], axis=1)
                mask = (pixels[..., 0]<=img_w-1.) & (pixels[..., 0]>=0.) & (pixels[..., 1]<=img_h -1.) & (pixels[..., 1]>=0) & (depth > 0.)
                index = np.where(mask & (depth > 0))[0]
                depth_map = np.zeros((img_h, img_w, 3))
                for idx in index:
                    x, y = int(pixels[idx][0]), int(pixels[idx][1])
                    dp = (depth[idx] - 1) / (25 - 1)
                    color = cmap([dp])[0][:3]
                    color = (color[0] * 255, color[1] * 255, color[2] * 255)
                    cv2.circle(depth_map, (x, y), radius=2, color=color, thickness=2)
                cv2.imwrite(os.path.join(save_dir, f"pcld_{'%04d' % t}.png"), depth_map)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = "cuda", help='torch device, e.g. cpu, cuda, cuda:1, ...')
    parser.add_argument('--forecasting_pretrain', type=str, default = "./forecasting_result/best.pth", help='path to the pretrained forecasting model')
    parser.add_argument('--densify_pretrain', type=str, default = "../pretrain/densify.pth", help='path to pretrained densification network')
    parser.add_argument('--horizon', type=int, default = 2, help='number of input/output frames')
    parser.add_argument('--no_densification', action="store_true", help='use this flag disable densification')
    parser.add_argument('--no_visualize', action="store_true", help='use this flag disable visualization, will be much faster!')
    return parser.parse_args()
            

if __name__  == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    test = PredictorTest(device, args.forecasting_pretrain, args.horizon, args.horizon, NUSC_ROOT, NUSC_TEST_VERSION,
                         occ_encoder_pretrain=args.densify_pretrain)
    test(densify=not args.no_densification, visualize=not args.no_visualize)
