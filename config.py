import os

# paths
ROOT = os.path.dirname(__file__)
NUSC_ROOT = "/ssd1/data/nuscenes/" # path to nuscenes dataset

# dataset settings
NUSC_TRAIN_VERSION = "v1.0-trainval"
NUSC_TEST_VERSION = "v1.0-test"
NUSC_MINI_VERSION = "v1.0-mini"

# grid setting
GRID_H = 45
GRID_L = 700
GRID_W = 700
GRID_SCALE = 0.2
