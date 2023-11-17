'''NuScenes data loader for scene forecasting task'''

import torch
import numpy as np

from .nusc import NuScenesDataset

class NuScenesPredictionDataset(NuScenesDataset):
    def __init__(self, nusc_root, nusc_version, n_input, n_output, istrain=True, verbose=True):
        super().__init__(nusc_root, nusc_version, istrain, verbose)
        self.input, self.output = n_input, n_output
        self.setSequence(n_input, n_output)

    def setSequence(self, n_input, n_output):
        self.seqtokens = []
        for scene_token in self.tokendict:
            samples = self.tokendict[scene_token]
            for center_id in range(n_input-1, len(samples)-n_output):
                seq_token_data = {}
                seq_token_data["scene"] = scene_token
                seq_token_data["center"] = samples[center_id]
                seq_token_data["input"] = [samples[center_id-n_input+1+i] for i in range(n_input)]
                seq_token_data["output"] = [samples[center_id+1+i] for i in range(n_output)]
                self.seqtokens.append(seq_token_data)
    
    def markMovingPoints(self, pcld, bboxes):
        mask = np.zeros(len(pcld))
        for bbox in bboxes:
            i = bbox[1] - bbox[0] # [3, ]
            j = bbox[3] - bbox[0] # [3, ]
            k = bbox[4] - bbox[0] # [3, ]
            v = pcld - bbox[0] # [n, 3]
            cur_mask = np.logical_and(np.logical_and(np.logical_and(v @ i > 0, v @ i < i @ i), np.logical_and(v @ j > 0, v @ j < j @ j)), np.logical_and(v @ k > 0, v @ k < k @ k))
            mask = np.logical_or(mask, cur_mask)
        return mask
    
    def generateMaskOccPoints(self, bboxes, scale=0.2):
        occ_points = []
        for bbox in bboxes: # [8, 3]
            dx = bbox[1] - bbox[0] # [3, ]
            dy = bbox[3] - bbox[0] # [3, ]
            dz = bbox[4] - bbox[0] # [3, ]
            xvecs = np.linspace([0, 0, 0], dx, int(np.linalg.norm(dx)/scale) + 1)
            yvecs = np.linspace([0, 0, 0], dy, int(np.linalg.norm(dy)/scale) + 1)
            zvecs = np.linspace([0, 0, 0], dz, int(np.linalg.norm(dz)/scale) + 1)
            points = bbox[0] + np.asarray([[[vx+vy+vz for vz in zvecs] for vy in yvecs] for vx in xvecs]).reshape(-1, 3)
            occ_points.append(points)
        if len(occ_points) > 0:
            return np.concatenate(occ_points, axis=0)
        else:
            return np.array([[0, 0, 0]])
    
    def loadseq(self, tokenlist, ref_pose=None):
        pclds = []
        pcldmasks = []
        occmaskpts = []
        occmask_indices = []
        tindices = []
        rgbs = []
        intrinsics = []
        rgb_poses = []
        lidar_poses = []
        for i, token in enumerate(tokenlist):
            sample = self.nusc.get('sample', token)

            # lidar pcld
            pcld, pose = self.loadLiDARSample(self.nusc, self.nusc_root, sample)
            # NOTE: if no reference pose, then use current lidar pose
            world_pose = pose if ref_pose is None else ref_pose
            pcld, pose = self.transformPCLD(pcld, pose, world_pose)
            tindex = np.full(len(pcld), i, dtype=np.float32)
            pclds.append(pcld)
            tindices.append(tindex)
            lidar_poses.append(pose)

            # 3D BBox
            bbox = self.load3DBBox(self.nusc, sample, select_moving=True)
            if len(bbox) > 0:
                N, P, C = bbox.shape
                bbox = np.concatenate([bbox.reshape(N*P, C), np.ones((N*P, 1))], axis=1)
                bbox = (np.linalg.inv(world_pose) @ bbox.T).T
                bbox = bbox[..., 0:3] / bbox[..., 3].reshape((-1, 1))
                bbox = bbox.reshape(N, P, C)
            
            occ_mask_points = self.generateMaskOccPoints(bbox, scale = 0.15)
            occmaskpts.append(occ_mask_points)
            occmask_indices.append(np.full(len(occ_mask_points), i, dtype=np.float32))

            mask = self.markMovingPoints(pcld, bbox)
            pcldmasks.append(mask)

            # rgb
            if not self.istrain:
                rgb, intrinsic, rgb_pose = self.loadRGBSample(self.nusc, self.nusc_root, sample)
                rgbs.append(rgb)
                intrinsics.append(intrinsic)
                rgb_poses.append(np.linalg.inv(world_pose) @ rgb_pose)

        pclds = np.concatenate(pclds, axis=0)
        pcldmasks = np.concatenate(pcldmasks, axis=0)
        tindices = np.concatenate(tindices, axis=0)

        occmaskpts = np.concatenate(occmaskpts, axis=0)
        occmask_indices = np.concatenate(occmask_indices, axis=0)

        lidar_poses = np.asarray(lidar_poses)
        rgbs = np.asarray(rgbs)
        intrinsics = np.asarray(intrinsics)
        rgb_poses = np.asarray(rgb_poses)

        return pclds, pcldmasks, tindices, occmaskpts, occmask_indices, lidar_poses, rgbs, intrinsics, rgb_poses

    def __len__(self):
        return len(self.seqtokens)
    
    def __getitem__(self, index):
        seqtoken = self.seqtokens[index]
        # center frame
        center_sample = self.nusc.get('sample', seqtoken["center"])
        _, center_pose = self.loadLiDARSample(self.nusc, self.nusc_root, center_sample)
        # input
        input_pcld, input_pcldmask, input_tindex, input_occmaskpts, input_occmask_tindex, input_lidar_poses, input_rgbs, input_intrinsics, input_rgb_poses = self.loadseq(seqtoken["input"], ref_pose=center_pose)
        # output
        output_pcld, output_pcldmask, output_tindex, output_occmaskpts, output_occmask_tindex, output_lidar_poses, output_rgbs, output_intrinsics, output_rgb_poses = self.loadseq(seqtoken["output"], ref_pose=center_pose)

        return {
            "input_pcld": torch.from_numpy(input_pcld),       # [N_in, 3]
            "input_tindex": torch.from_numpy(input_tindex),   # [N_in, ]
            "input_rgbs": torch.from_numpy(input_rgbs),       # [T, 6, H, W, 3]
            "input_intrinsics": torch.from_numpy(input_intrinsics), # [T, 6, 3, 3]
            "input_rgb_poses": torch.from_numpy(input_rgb_poses),   # [T, 6, 4, 4]
            "output_pcld": torch.from_numpy(output_pcld),     # [N_out, 3]
            "output_tindex": torch.from_numpy(output_tindex), # [N_out, ]
            "output_origin": torch.from_numpy(output_lidar_poses[:, 0:3, 3]), # [T, 3]
            "output_rgbs": torch.from_numpy(output_rgbs),       # [T, 6, H, W, 3]
            "output_intrinsics": torch.from_numpy(output_intrinsics), # [T, 6, 3, 3]
            "output_rgb_poses": torch.from_numpy(output_rgb_poses),   # [T, 6, 4, 4]

            # Ground-truth Moving Objects Mask Related
            "input_pcldmask": torch.from_numpy(input_pcldmask), # [N_in,]
            "input_occmaskpts": torch.from_numpy(input_occmaskpts), # [N_in, 3]
            "input_occmask_tindex": torch.from_numpy(input_occmask_tindex), # [N_in,]
            "output_pcldmask": torch.from_numpy(output_pcldmask), # [N_in,]
            "output_occmaskpts": torch.from_numpy(output_occmaskpts), # [N_in, 3]
            "output_occmask_tindex": torch.from_numpy(output_occmask_tindex), # [N_in,]
        }