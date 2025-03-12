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

import time

import cv2
import torch
import numpy as np
from matplotlib import cm

class OccVisualizer:
    def __init__(self, occlib):
        self.occlib = occlib
        self.occ = None
        self.pixel = None
        self.rotation = None
        self.translation = None
        self.sizes = None

    def _setData(self, occ, H, W, intrinsic, pose, grid = [700, 700, 45], scale = 0.2):
        # set occupancy grid
        self.occ = occ.to(torch.float32).cuda().contiguous()
        # set pixel map
        h_map = np.array([[i for j in range(W)] for i in range(H)])
        w_map = np.array([[j for j in range(W)] for i in range(H)])
        intrinsic = intrinsic.cpu().numpy()
        fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
        h_map = np.expand_dims((h_map - cy) / fy, axis=-1)
        w_map = np.expand_dims((w_map - cx) / fx, axis=-1)
        pixel = np.concatenate([w_map, h_map, np.ones_like(h_map)], axis=-1, dtype=np.float32)
        self.pixel = torch.from_numpy(pixel).to(torch.float32).cuda().contiguous()
        # set pose / transformation
        self.rotation = pose[0:3, 0:3].to(torch.float32).cuda().contiguous()
        self.translation = pose[0:3, 3].to(torch.float32).cuda().contiguous()
        # set grid and scale
        sizes = np.concatenate([np.array(grid) * scale * -1 / 2, np.array([scale])], axis=0, dtype=np.float32)
        self.sizes = torch.from_numpy(sizes).to(torch.float32).cuda().contiguous()
    
    def vizDistMap(self, save_path, occ, H, W, intrinsic, pose, grid = [700, 700, 45], scale = 0.2):
        self._setData(occ, H, W, intrinsic, pose, grid, scale)
        occ_render = self.occlib.render_demo(self.occ, self.pixel, self.rotation, self.translation, self.sizes)
        occ_dist = occ_render[0].cpu().numpy()
        occ_dist_color = cv2.cvtColor((np.clip(occ_dist*5, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(save_path, occ_dist_color)

    def vizDepthColorMap(self, save_path, occ, H, W, intrinsic, pose, grid = [700, 700, 45], scale = 0.2, depth_range=(1.0, 25.0)):
        cmap = cm.get_cmap('jet')
        self._setData(occ, H, W, intrinsic, pose, grid, scale)
        occ_render = self.occlib.render_demo(self.occ, self.pixel, self.rotation, self.translation, self.sizes)
        occ_depth = occ_render[1].cpu().numpy()
        occ_depth_color = (occ_depth - depth_range[0]) / (depth_range[1] - depth_range[0])
        depth_map = (cmap(occ_depth_color) * 255)[:, :, :3].astype(np.uint8)
        cv2.imwrite(save_path, depth_map)

    def __call__(self, occ, H, W, intrinsic, pose, grid = [700, 700, 45], scale = 0.2,
                 window_name = "OCC_VISUALIZER"):
        ''' Interactive Occupancy Viewer'''

        if self.occ is None: # data not set
            self._setData(occ, H, W, intrinsic, pose, grid, scale)

        cv2.namedWindow(window_name)

        while True:

            start = time.time()

            occ_render = self.occlib.render_demo(self.occ, self.pixel, self.rotation, self.translation, self.sizes)
            occ_dist = occ_render[0].cpu().numpy()
            occ_depth = occ_render[1].cpu().numpy()
            occ_dist_color = cv2.cvtColor((np.clip(occ_dist*5, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            occ_depth_color = cv2.cvtColor((np.clip(occ_depth*5, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            print(f"[Render] Translation: {self.translation.cpu().numpy()}, depth range: ({np.min(occ_depth)}, {np.max(occ_depth)}), time using {time.time() - start} s")

            cv2.imshow(window_name, occ_dist_color)
            key = cv2.waitKey(0)

            if key == ord('q'):
                print('exit.')
                break
            elif key == ord('d'):
                self.translation[0] += 1
            elif key == ord('a'):
                self.translation[0] -= 1
            elif key == ord('w'):
                self.translation[1] += 1
            elif key == ord('s'):
                self.translation[1] -= 1
            elif key == ord('j'):
                self.translation[2] += 1
            elif key == ord('k'):
                self.translation[2] -= 1

        cv2.destroyAllWindows()
