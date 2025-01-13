#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from os import device_encoding
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict
from copy import deepcopy

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.utils.batch_utils import batch_utils
from ccdiff.algos.algo_utils import optimize_trajectories
from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
from l5kit.geometry import transform_points
from tbsim.utils.timer import Timers
from tbsim.policies.common import Action, Plan, RolloutAction
from tbsim.policies.common import interpolate_trajectory, enumerate_keys
from tbsim.policies.base import Policy
import matplotlib.pyplot as plt

try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.timer import Timers
from tbsim.utils.agent_centric_transform import get_neighbor_history_relative_states

class GTPolicy(Policy):
    def __init__(self, device):
        super(GTPolicy, self).__init__(device)

    def eval(self):
        pass

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        print(obs["agent_hist"].shape)
        print(obs["neigh_hist"].shape)
        print(obs["history_positions"].shape)
        print(obs["history_yaws"].shape)
        print(obs["history_positions"][:, -1:].shape)
        print(obs["history_yaws"][:, -1:].shape)
        reference_pos = obs["history_positions"][:, -1:]
        reference_yaw = obs["history_yaws"][:, -1:]
        print(reference_pos.shape, obs["target_positions"].shape)
        reference_pos = np.tile(reference_pos, (1, obs["target_positions"].shape[1], 1))
        reference_yaw = np.tile(reference_yaw, (1, obs["target_yaws"].shape[1], 1))

        try: 
            reference_pos[1:] = obs["target_positions"][1:]
            reference_yaw[1:] = obs["target_yaws"][1:]
        except:
            print("only 1 agent at the scene")

        reference_pos[np.where(np.isnan(reference_pos))] = 0
        reference_yaw[np.where(np.isnan(reference_yaw))] = 0
        
        pos = TensorUtils.to_torch(reference_pos, device=self.device)
        yaw = TensorUtils.to_torch(reference_yaw, device=self.device)
        agent_avail = []
        for idx in range(pos.shape[0]): 
            avail_idx = torch.isnan(pos[idx])[0]
            assert torch.logical_xor(torch.isnan(yaw[idx])[0], avail_idx).sum() == 0, "The availabilities are not consistent"
            agent_avail.append(avail_idx)
        
        pos[torch.where(torch.isnan(pos))] = 0
        yaw[torch.where(torch.isnan(yaw))] = 0

        action = Action(
            positions=pos,
            yaws=yaw
        )
        info = dict(
            action_samples=Action(
                positions=pos[:, None, ...],
                yaws=yaw[:, None, ...]
            ).to_dict(),
        )
        return action, info

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        agent_avail = []
        for idx in range(pos.shape[0]): 
            avail_idx = torch.isnan(pos[idx])[0]
            assert torch.logical_xor(torch.isnan(yaw[idx])[0], avail_idx).sum() == 0, "The availabilities are not consistent"
            agent_avail.append(avail_idx)
        
        # assert 1==2
        pos, yaw = interpolate_trajectory(pos, yaw, agent_avail, mode='slow_down')
        pos[torch.where(torch.isnan(pos))] = 0
        yaw[torch.where(torch.isnan(yaw))] = 0
    
        plan = Plan(
            positions=pos,
            yaws=yaw,
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )
        
        return plan, {}


class GTPolicyOpenLoop(Policy):
    def __init__(self, device):
        super(GTPolicyOpenLoop, self).__init__(device)

    def eval(self):
        pass

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # for k in enumerate_keys(obs): 
        #     print(k)
        print(obs["agent_hist"].shape)
        print(obs["neigh_hist"].shape)
        print(obs["history_positions"].shape)
        print(obs["history_yaws"].shape)
        print(obs["history_positions"][:, -1:].shape)
        print(obs["history_yaws"][:, -1:].shape)
        reference_pos = obs["history_positions"][:, -1:]
        reference_yaw = obs["history_yaws"][:, -1:]
        print(reference_pos.shape, obs["target_positions"].shape)

        pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)


        ''' edit traj
        agent_avail = []
        for idx in range(pos.shape[0]): 
            avail_idx = torch.isnan(pos[idx])[0]
            assert torch.logical_xor(torch.isnan(yaw[idx])[0], avail_idx).sum() == 0, "The availabilities are not consistent"
            agent_avail.append(avail_idx)
        
        # assert 1==2
        # pos, yaw = interpolate_trajectory(pos, yaw, agent_avail, mode='slow_down')
        pos[torch.where(torch.isnan(pos))] = 0
        yaw[torch.where(torch.isnan(yaw))] = 0

        # pos += torch.rand_like(pos) * 1.0
        # yaw += torch.randn_like(yaw) * 0.01
        # print(pos.shape, yaw.shape)
        # print([(pos[i].max(), pos[i].min()) for i in range(pos.shape[0])])
        # print([(yaw[i].max(), yaw[i].min()) for i in range(yaw.shape[0])])
        # assert 1==2
        '''
        action = Action(
            positions=pos,
            yaws=yaw
        )
        # info = dict(
        #     action_samples=Action(
        #         positions=pos[:, None, ...],
        #         yaws=yaw[:, None, ...], 
        #     ).to_dict(),
        # )
        return action, {}

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        # agent_avail = []
        # for idx in range(pos.shape[0]): 
        #     avail_idx = torch.isnan(pos[idx])[0]
        #     assert torch.logical_xor(torch.isnan(yaw[idx])[0], avail_idx).sum() == 0, "The availabilities are not consistent"
        #     agent_avail.append(avail_idx)
        
        pos[torch.where(torch.isnan(pos))] = 0
        yaw[torch.where(torch.isnan(yaw))] = 0
        plan = Plan(
            positions=pos,
            yaws=yaw,
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )
        
        return plan, {}


class GTNaNPolicy(Policy):
    '''
    Dummy policy to return GT action from data. If GT is non available fills
    in with nans (instead of 0's as above).
    '''
    def __init__(self, device):
        super(GTNaNPolicy, self).__init__(device)

    def eval(self):
        pass

    def add_policy(self, policy): 
        self.policy = policy

    def _prepare_analysis(self, data_input): 
        data_batch = deepcopy(data_input)
        # print(data_batch["all_other_agents_history_positions"].shape, data_batch["all_other_agents_history_yaws"].shape)
        for k in data_batch.keys(): 
            if isinstance(data_batch[k], np.ndarray):
                # print(k, data_batch[k].shape)
                data_batch[k] = TensorUtils.to_torch(data_batch[k], device=self.device).unsqueeze(0)
        trajectories, _ = get_neighbor_history_relative_states(data_batch)
        return trajectories
    
    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        invalid_mask = ~obs["target_availabilities"]
        gt_pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        # print(invalid_mask.shape, gt_pos.shape)
        gt_pos[invalid_mask, :] = torch.nan
        gt_yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        gt_yaw[invalid_mask, :] = torch.nan
        

        action = Action(
            positions=gt_pos,
            yaws=gt_yaw,
        )
        print(gt_pos.shape, gt_yaw.shape)
        
        return action, {}


