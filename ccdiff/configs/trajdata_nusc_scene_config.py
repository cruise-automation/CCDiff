#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import math
import numpy as np

from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class NuscTrajdataSceneTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(NuscTrajdataSceneTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        self.trajdata_source_train = ["nusc_trainval-train", "nusc_trainval-train_val"]
        self.trajdata_source_valid = ["nusc_trainval-val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "nusc_trainval" : "/mnt/disks/sdb/nuscenes",
            "nusc_test" : "/mnt/disks/sdb/nuscenes",
            "nusc_mini" : "/mnt/disks/sdb/nuscenes",
        }

        # for debug
        self.trajdata_rebuild_cache = False

        self.rollout.enabled = True
        self.rollout.save_video = True
        self.rollout.every_n_steps = 100000
        self.rollout.warm_start_n_steps = 0

        # training config
        # assuming 1 sec (10 steps) past, 2 sec (20 steps) future
        self.training.batch_size = 4 # 100
        self.training.num_steps = 100000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 10000
        self.save.best_k = 10

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 1 # 2 # 4 # 32
        self.validation.num_data_workers = 6
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 5 # 50

        self.on_ngc = False
        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100

        # vec map params (only used by algorithm which uses vec map)
        self.training_vec_map_params = {
            'S_seg': 15,
            'S_point': 80,
            'map_max_dist': 80,
            'max_heading_error': 0.25*np.pi,
            'ahead_threshold': -40,
            'dist_weight': 1.0,
            'heading_weight': 0.1,
        }


class NuscTrajdataSceneEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(NuscTrajdataSceneEnvConfig, self).__init__()

        self.data_generation_params.trajdata_centric = "scene" # ["agent", "scene"]
        # which types of agents to include from ['unknown', 'vehicle', 'pedestrian', 'bicycle', 'motorcycle']
        self.data_generation_params.trajdata_only_types = ["vehicle"]
        # which types of agents to predict
        self.data_generation_params.trajdata_predict_types = ["vehicle"]
        # list of scene description filters
        self.data_generation_params.trajdata_scene_desc_contains = None
        # whether or not to include the map in the data
        #       TODO: handle mixed map-nomap datasets
        self.data_generation_params.trajdata_incl_map = True
        # For both training and testing:
        # if scene-centric, max distance to scene center to be included for batching
        # if agent-centric, max distance to each agent to be considered neighbors
        self.data_generation_params.trajdata_max_agents_distance = 50 # np.inf # 30
        # standardize position and heading for the predicted agnet
        self.data_generation_params.trajdata_standardize_data = True

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # number of semantic layers that will be used (based on which trajdata dataset is being used)
        self.rasterizer.num_sem_layers = 3 # 7
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([0], [1], [2])
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1.0 / 2.0 # 2 px/m
        # where the agent is on the map, (0.0, 0.0) is the center
        self.rasterizer.ego_center = (-0.5, 0.0)

        # max_agent_num (int, optional): The maximum number of agents to include in a batch for scene-centric batching.
        self.data_generation_params.other_agents_num = 20 # None # 20

        # max_neighbor_num (int, optional): The maximum number of neighbors to include in a batch for agent-centric batching.