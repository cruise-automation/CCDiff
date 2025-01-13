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

import tbsim.utils.tensor_utils as TensorUtils

from tbsim.policies.common import Action, Plan, RolloutAction
from tbsim.policies.base import Policy

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.timer import Timers
from tbsim.utils.agent_centric_transform import get_neighbor_history_relative_states

class CCDiffHybridPolicyControl(Policy):
    '''
    Dummy policy to return GT action from data. 
    If GT is non available fills in with nans (instead of 0's, so the agents will be removed). 
    Set some of the agents as controllable ones based on the causal discovery results. 
    '''
    def __init__(self, device):
        super(CCDiffHybridPolicyControl, self).__init__(device)
        self.controllable_set = [0, 1]

    def eval(self):
        pass
    
    def set_controllable_set(self, controllable_set): 
        '''
        Set controllable agents index
        '''
        assert isinstance(controllable_set, list)
        self.controllable_set = controllable_set

    def add_policy(self, policy): 
        self.policy = policy

    def _step_controller(self, obs, gt_pos, gt_yaw, invalid_mask, **kwargs): 
        if len(self.controllable_set) > 0: 
            pred = self.policy.get_action(obs, **kwargs)[0].to_dict()
            # print(pred.to_dict())
            print('+++++++++++++++++')
            print(torch.cat([pred['positions'][i][0] for i in range(len(pred['positions']))]))
        else: 
            print('+++++++++++++++++')
            print("skip model generation")

        # TODO: use gt or not
        print("controllable set: ", self.controllable_set)
        if len(self.controllable_set) > 0: 
            gt_pos[self.controllable_set] = pred['positions'][self.controllable_set]
            gt_yaw[self.controllable_set] = pred['yaws'][self.controllable_set]

        return gt_pos, gt_yaw
    
    def _prepare_analysis(self, data_input): 
        data_batch = deepcopy(data_input)
        for k in data_batch.keys(): 
            if isinstance(data_batch[k], np.ndarray):
                data_batch[k] = TensorUtils.to_torch(data_batch[k], device=self.device).unsqueeze(0)
        trajectories, _ = get_neighbor_history_relative_states(data_batch)
        return trajectories
    
    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        invalid_mask = ~obs["target_availabilities"]
        gt_pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
 

        gt_yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)

        edit_pos, edit_yaw = self._step_controller(obs, gt_pos, gt_yaw, invalid_mask, **kwargs)
        edit_pos[invalid_mask] = np.nan
        edit_yaw[invalid_mask] = np.nan
        action = Action(
            positions=edit_pos,
            yaws=edit_yaw,
        )

        return action, {} # {}

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        edit_pos, edit_yaw = self._step_controller(obs, pos, yaw, **kwargs)

        plan = Plan(
            positions=edit_pos,
            yaws=edit_yaw,
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )
        
        return plan, {}

class ReplayPolicy(Policy):
    def __init__(self, action_log, device):
        super(ReplayPolicy, self).__init__(device)
        self.action_log = action_log

    def eval(self):
        pass

    def get_action(self, obs, step_index=None, **kwargs) -> Tuple[Action, Dict]:
        assert step_index is not None
        scene_index = TensorUtils.to_numpy(obs["scene_index"]).astype(np.int64).tolist()
        track_id = TensorUtils.to_numpy(obs["track_id"]).astype(np.int64).tolist()
        pos = []
        yaw = []
        for si, ti in zip(scene_index, track_id):
            scene_log = self.action_log[str(si)]
            if ti == -1:  # ego
                pos.append(scene_log["ego_action"]["positions"][step_index, 0])
                yaw.append(scene_log["ego_action"]["yaws"][step_index, 0])
            else:
                scene_track_id = scene_log["agents_obs"]["track_id"][0]
                agent_ind = np.where(ti == scene_track_id)[0][0]
                pos.append(scene_log["agents_action"]["positions"][step_index, agent_ind])
                yaw.append(scene_log["agents_action"]["yaws"][step_index, agent_ind])

        # stack and create the temporal dimension
        pos = np.stack(pos, axis=0)[:, None, :]
        yaw = np.stack(yaw, axis=0)[:, None, :]

        action = Action(
            positions=pos,
            yaws=yaw
        )
        return action, {}

