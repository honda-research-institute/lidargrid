'''Nuscenes data loader for loading a sequence of lidar points. Can be used for moving region detection'''

import torch
import numpy as np

from .nusc import NuScenesDataset

class NuScenesPTSDataset(NuScenesDataset):
    def __init__(self, nusc_root, nusc_version, seq_len=3, has_rgb=True, verbose=True):
        super().__init__(nusc_root, nusc_version, True, verbose)
        self.seq_len = seq_len
        self.has_rgb = has_rgb
        self.datalist = self.to_sequence()

    def to_sequence(self):
        datalist = []
        for scene_token in self.tokendict:
            sample_tokens = self.tokendict[scene_token]
            if len(sample_tokens) >= self.seq_len:
                for i in range(len(sample_tokens)-self.seq_len+1):
                    datalist.append(sample_tokens[i:i+self.seq_len])
        return datalist # [[token_1, token_2, token_3, ...], ...]
    
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
    
    def load_sequence(self, tokens, ref_pose=None):
        pclds, t_indices, lidar_poses, rgbs, rgb_poses, intrinsics = [], [], [], [], [], []
        for i, token in enumerate(tokens):
            sample = self.nusc.get('sample', token)
            # lidar pcld
            pcld, pose = self.loadLiDARSample(self.nusc, self.nusc_root, sample)
            if ref_pose is not None:
                pcld, pose = self.transformPCLD(pcld, pose, ref_pose)
            tindex = np.full(len(pcld), i, dtype=np.float32)
            pclds.append(pcld)
            t_indices.append(tindex)
            lidar_poses.append(pose)
            # rgb
            if self.has_rgb:
                rgb, intrinsic, rgb_pose = self.loadRGBSample(self.nusc, self.nusc_root, sample)
                rgbs.append(rgb)
                intrinsics.append(intrinsic)
                rgb_poses.append(np.linalg.inv(ref_pose) @ rgb_pose)

        # integrate arrays
        return {
            "pclds": np.concatenate(pclds, axis=0),
            "t_indices": np.concatenate(t_indices, axis=0),
            "lidar_poses": np.asarray(lidar_poses),
            "rgbs": np.asarray(rgbs),
            "intrinsics": np.asarray(intrinsics),
            "rgb_poses": np.asarray(rgb_poses)
        }
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        tokens = self.datalist[index]
        # center frame
        center_idx = self.seq_len // 2
        center_sample = self.nusc.get('sample', tokens[center_idx])
        center_pcld, center_pose = self.loadLiDARSample(self.nusc, self.nusc_root, center_sample)
        # center moving points mask
        bbox = self.load3DBBox(self.nusc, center_sample, select_moving=True)
        if len(bbox) > 0:
            N, P, C = bbox.shape
            bbox = np.concatenate([bbox.reshape(N*P, C), np.ones((N*P, 1))], axis=1)
            bbox = (np.linalg.inv(center_pose) @ bbox.T).T
            bbox = bbox[..., 0:3] / bbox[..., 3].reshape((-1, 1))
            bbox = bbox.reshape(N, P, C)
        center_mask = self.markMovingPoints(center_pcld[..., 0:3], bbox)
        # load sequential frames
        sequence_data = self.load_sequence(tokens, ref_pose=center_pose)
        # return
        return {
            "center_pcld": torch.from_numpy(center_pcld[..., 0:3]), # [N, 3]
            "center_mask": torch.from_numpy(center_mask), # [N,]
            "center_idx": center_idx, # int
            "seq_pcld": torch.from_numpy(sequence_data["pclds"]), # [N_all, 3]
            "seq_tindex": torch.from_numpy(sequence_data["t_indices"]), # [N_all, ]
            "seq_lidar_pose": torch.from_numpy(sequence_data["lidar_poses"]), # [seq_len, 4, 4]
            "seq_rgb": torch.from_numpy(sequence_data["rgbs"]), # [seq_len, 6, 900, 1600, 3]
            "seq_intrinsic": torch.from_numpy(sequence_data["intrinsics"]), # [seq_len, 6, 3, 3]
            "seq_rgb_pose": torch.from_numpy(sequence_data["rgb_poses"]) # [seq_len, 6, 4, 4]
        }