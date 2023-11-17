'''Base data loader for NuScenes dataset. Getting rgb, lidar and their corresponding pose'''

import os
import random

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

import torch
from torch.utils.data import Dataset

CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

class NuScenesDataset(Dataset):
    def __init__(self, nusc_root, nusc_version, istrain=True, verbose=True):
        self.nusc = NuScenes(nusc_version, nusc_root, verbose=verbose)
        self.nusc_root = nusc_root
        self.nusc_version = nusc_version
        self.istrain = istrain
        self.verbose = verbose

        self.initialize()
    
    def initialize(self):
        self.n_scenes = len(self.nusc.scene)
        # traverse each scene and each frame to get tokens
        scenes = self.nusc.scene
        self.tokenlist = []
        self.tokendict = {}
        for scene in scenes:
            scene_token = scene["token"]
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_token = first_sample["token"]
            self.tokendict[scene_token] = []
            while sample_token != scene['last_sample_token']:
                sample = self.nusc.get('sample', sample_token)
                self.tokenlist.append((scene_token, sample_token))
                self.tokendict[scene_token].append(sample_token)
                sample_token = sample['next']

    @classmethod
    def loadLiDARSample(self,
                        nusc,
                        nusc_root,
                        sample):
        # get point cloud
        lidar_sample = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_pcld = LidarPointCloud.from_file(os.path.join(nusc_root, lidar_sample["filename"]))
        ego_mask = np.logical_and( # avoid getting near field points (usually on the car)
            np.logical_and(-0.8 <= lidar_pcld.points[0], lidar_pcld.points[0] <= 0.8),
            np.logical_and(-1.5 <= lidar_pcld.points[1], lidar_pcld.points[1] <= 2.5),
        )
        lidar_pcld = np.asarray(lidar_pcld.points[:, np.logical_not(ego_mask)]).T # [n_points, 4]
        # get camera pose
        sd_ep = nusc.get("ego_pose", lidar_sample["ego_pose_token"])
        sd_cs = nusc.get("calibrated_sensor", lidar_sample["calibrated_sensor_token"])
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        global_from_sensor = global_from_ego.dot(ego_from_sensor)
        lidar_pose = np.asarray(global_from_sensor)

        return lidar_pcld, lidar_pose
    
    @classmethod
    def loadRGBSample(self,
                      nusc,
                      nusc_root,
                      sample):
        rgb_img_list = [] # Store rgb images from 6 camera views
        intrinsics_list = [] # Store intrinsics from 6 camera views
        cam_pose_list = [] # Store camera pose from 6 camera views
        for sensor in CAM_LIST:
            # rgb image
            rgb_sample = nusc.get("sample_data", sample["data"][sensor])
            rgb_img = cv2.imread(os.path.join(nusc_root, rgb_sample["filename"]))
            rgb_img_list.append(rgb_img)
            # camera pose
            sd_ep = nusc.get("ego_pose", rgb_sample["ego_pose_token"])
            sd_cs = nusc.get("calibrated_sensor", rgb_sample["calibrated_sensor_token"])
            global_from_ego = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
            )
            ego_from_sensor = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
            )
            cam_pose = global_from_ego.dot(ego_from_sensor) # global from sensor
            cam_pose_list.append(cam_pose)
            # intrinsics
            intrinsics = np.eye(3)
            for dt in nusc.sensor:
                if dt['channel'] == sensor:
                    for calib in nusc.calibrated_sensor:
                        if dt['token'] == calib['sensor_token']:
                            intrinsics = np.array(calib['camera_intrinsic'])
            intrinsics_list.append(intrinsics)
        return np.asarray(rgb_img_list), np.asarray(intrinsics_list), np.asarray(cam_pose_list)
    
    @classmethod
    def transformPCLD(self, pcld, cur_pose, ref_pose):
        pcld[..., 3] = np.ones_like(pcld[..., 3])
        pose = np.linalg.inv(ref_pose) @ cur_pose
        pcld = (pose @ pcld.T).T
        pcld = pcld[..., 0:3] / pcld[..., 3].reshape((-1, 1))
        return pcld, pose

    @classmethod
    def load3DBBox(self, nusc, sample, select_moving=True):
        # all sensors share the same bbox information
        sensor = CAM_LIST[0]
        bboxes = nusc.get_boxes(sample["data"][sensor])
        ret_bboxes = []
        # filtering bboxes
        for bbox in bboxes:
            ann = nusc.get('sample_annotation', bbox.token)
            if select_moving:
                for attr_token in ann["attribute_tokens"]:
                    attr = nusc.get('attribute', attr_token)
                    if attr['name'] in ['vehicle.moving', 'pedestrian.moving', 'cycle.with_rider']:
                        ret_bboxes.append(np.asarray(bbox.corners()).T)
                        break
            else:
                ret_bboxes.append(np.asarray(bbox.corners()).T)
        
        return np.asarray(ret_bboxes)
    
    def getRefLiDARPose(self, scene_token, fixed=True):
        '''Set a frame as reference frame and get its lidar pose. Afterwards, other frames can transform to the coordinate of the reference frame'''
        sample_token_list = self.tokendict[scene_token]
        if fixed: # set the middle frame as reference frame
            sample_token = sample_token_list[len(sample_token_list)//2]
        else: # randomly pick one frame as reference frame
            sample_token = random.sample(sample_token_list, 1)[0]
        sample = self.nusc.get('sample', sample_token)
        _, ref_lidar_pose = self.loadLiDARSample(self.nusc, self.nusc_root, sample)
        return ref_lidar_pose

    def __len__(self):
        return len(self.tokenlist)
    
    def __getitem__(self, index):
        scene_token, sample_token = self.tokenlist[index]
        sample = self.nusc.get('sample', sample_token)
        lidar_pcld, lidar_pose = self.loadLiDARSample(self.nusc, self.nusc_root, sample)
        rgb_images, intrinsics, rgb_poses = self.loadRGBSample(self.nusc, self.nusc_root, sample)
        ref_lidar_pose = self.getRefLiDARPose(scene_token, fixed=not self.istrain)

        return {
            "lidar_pcld": torch.from_numpy(lidar_pcld[:, 0:3].astype(np.float32)),
            "lidar_intensity": torch.from_numpy(lidar_pcld[:, 3].astype(np.float32)),
            "lidar_pose": torch.from_numpy(lidar_pose.astype(np.float32)),
            "ref_lidar_pose": torch.from_numpy(ref_lidar_pose.astype(np.float32)),
            "rgb_images": torch.from_numpy(rgb_images.astype(np.float32)),
            "intrinsics": torch.from_numpy(intrinsics.astype(np.float32)),
            "rgb_poses": torch.from_numpy(rgb_poses.astype(np.float32))
        }