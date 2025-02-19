#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from this import d
import torch

import tbsim.utils.l5_utils as l5_utils
import tbsim.utils.trajdata_utils as av_utils
from tbsim import dynamics as dynamics
from tbsim.configs.base import ExperimentConfig


BATCH_TYPE = None


def set_global_batch_type(batch_type):
    global BATCH_TYPE
    assert batch_type in ["trajdata", "l5kit"]
    BATCH_TYPE = batch_type


def batch_utils():
    if BATCH_TYPE == "trajdata":
        return trajdataBatchUtils()
    elif BATCH_TYPE == "l5kit":
        return L5BatchUtils()
    else:
        raise NotImplementedError("Please set BATCH_TYPE in batch_utils.py to {trajdata, l5kit}")


class BatchUtils(object):
    """A base class for processing environment-independent batches"""
    @staticmethod
    def get_last_available_index(avails):
        """
        Args:
            avails (torch.Tensor): target availabilities [B, (A), T]

        Returns:
            last_indices (torch.Tensor): index of the last available frame
        """
        num_frames = avails.shape[-1]
        inds = torch.arange(0, num_frames).to(avails.device)  # [T]
        inds = (avails > 0).float() * inds  # [B, (A), T] arange indices with unavailable indices set to 0
        last_inds = inds.max(dim=-1)[1]  # [B, (A)] calculate the index of the last availale frame
        return last_inds

    @staticmethod
    def get_current_states(batch: dict, dyn_type: dynamics.DynType) -> torch.Tensor:
        """
        Get the dynamic states of the current timestep
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        """
        bs = batch["curr_speed"].shape
        if dyn_type == dynamics.DynType.BICYCLE:
            current_states = torch.zeros(*bs, 6).to(batch["curr_speed"].device)  # [x, y, yaw, vel, dh, veh_len]
            current_states[..., :2] = batch["history_positions"][..., -1, :]
            current_states[..., 2] = batch["history_yaws"][..., -1, 0]
            current_states[..., 3] = batch["curr_speed"].abs()
            current_states[..., [4]] = (batch["history_yaws"][..., 0] - batch["history_yaws"][..., 1]).abs()
            current_states[..., 5] = batch["extent"][..., 0]  # [veh_len]
        else:
            current_states = torch.zeros(*bs, 4).to(batch["curr_speed"].device)  # [x, y, vel, yaw]
            current_states[..., :2] = batch["history_positions"][..., -1, :]
            current_states[..., 2] = batch["curr_speed"]
            current_states[..., 3] = batch["history_yaws"][..., -1, 0]
            # print('speed, pos, yaws: ', batch["curr_speed"].isnan().any(), batch["history_positions"].isnan().any(), 
            #       batch["history_yaws"].isnan().any(), )
        return current_states
    
    @classmethod
    def get_current_states_all_agents(cls, batch: dict, step_time, dyn_type: dynamics.DynType) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def parse_batch(data_batch):
        raise NotImplementedError

    @staticmethod
    def batch_to_raw_all_agents(data_batch, step_time):
        raise NotImplementedError

    @staticmethod
    def batch_to_target_all_agents(data_batch):
        raise NotImplementedError

    @staticmethod
    def get_edges_from_batch(data_batch, ego_predictions=None, all_predictions=None):
        raise NotImplementedError

    @staticmethod
    def generate_edges(raw_type, extents, pos_pred, yaw_pred):
        raise NotImplementedError

    @staticmethod
    def gen_ego_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types):
        raise NotImplementedError

    @staticmethod
    def gen_EC_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types, mask=None):
        raise NotImplementedError

    @staticmethod
    def get_drivable_region_map(rasterized_map):
        raise NotImplementedError

    @staticmethod
    def get_modality_shapes(cfg: ExperimentConfig):
        raise NotImplementedError

class trajdataBatchUtils(BatchUtils):
    """Batch utils for trajdata"""
    @staticmethod
    def parse_batch(data_batch):
        return av_utils.parse_trajdata_batch(data_batch)

    @staticmethod
    def batch_to_raw_all_agents(data_batch, step_time):
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)

        src_pos = torch.cat(
            (
                data_batch["history_positions"].unsqueeze(1),
                data_batch["all_other_agents_history_positions"],
            ),
            dim=1,
        )
        src_yaw = torch.cat(
            (
                data_batch["history_yaws"].unsqueeze(1),
                data_batch["all_other_agents_history_yaws"],
            ),
            dim=1,
        )
        src_mask = torch.cat(
            (
                data_batch["history_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_history_availabilities"],
            ),
            dim=1,
        ).bool()

        extents = torch.cat(
            (
                data_batch["extent"][..., :2].unsqueeze(1),
                data_batch["all_other_agents_extents"][..., :2],
            ),
            dim=1,
        )
        
        curr_speed = torch.cat(
            (
                data_batch["curr_speed"].unsqueeze(1),
                data_batch["all_other_agents_curr_speed"]
            ),
            dim=1,
        )

        return {
            "history_positions": src_pos,
            "history_yaws": src_yaw,
            "curr_speed": curr_speed,
            "raw_types": raw_type,
            "history_availabilities": src_mask,
            "extents": extents,
        }

    @staticmethod
    def batch_to_target_all_agents(data_batch):
        pos = torch.cat(
            (
                data_batch["target_positions"].unsqueeze(1),
                data_batch["all_other_agents_future_positions"],
            ),
            dim=1,
        )
        yaw = torch.cat(
            (
                data_batch["target_yaws"].unsqueeze(1),
                data_batch["all_other_agents_future_yaws"],
            ),
            dim=1,
        )
        avails = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"],
            ),
            dim=1,
        )

        extents = torch.cat(
            (
                data_batch["extent"][..., :2].unsqueeze(1),
                data_batch["all_other_agents_history_extents"][..., -1,:2],
            ),
            dim=1,
        )

        return {
            "target_positions": pos,
            "target_yaws": yaw,
            "target_availabilities": avails,
            "extents": extents
        }
    
    @staticmethod
    def get_current_states_all_agents(batch: dict, step_time, dyn_type: dynamics.DynType) -> torch.Tensor:
        if batch["history_positions"].ndim==3:
            state_all = trajdataBatchUtils.batch_to_raw_all_agents(batch, step_time)
        else:
            state_all = batch
        bs, na = state_all["curr_speed"].shape[:2]
        if dyn_type == dynamics.DynType.BICYCLE:
            current_states = torch.zeros(bs, na, 6).to(state_all["curr_speed"].device)  # [x, y, yaw, vel, dh, veh_len]
            current_states[:, :, :2] = state_all["history_positions"][:, :, -1]
            current_states[:, :, 3] = state_all["curr_speed"].abs()
            current_states[:, :, [4]] = (state_all["history_yaws"][:, :, -1] - state_all["history_yaws"][:, :, 1]).abs()
            current_states[:, :, 5] = state_all["extent"][:, :, -1]  # [veh_len]
        else:
            current_states = torch.zeros(bs, na, 4).to(state_all["curr_speed"].device)  # [x, y, vel, yaw]
            current_states[:, :, :2] = state_all["history_positions"][:, :, -1]
            current_states[:, :, 2] = state_all["curr_speed"]
            current_states[:,:,3:] = state_all["history_yaws"][:,:,-1]
        return current_states

    @staticmethod
    def get_edges_from_batch(data_batch, ego_predictions=None, all_predictions=None):
        raise NotImplementedError

    @staticmethod
    def generate_edges(raw_type, extents, pos_pred, yaw_pred):
        return l5_utils.generate_edges(raw_type, extents, pos_pred, yaw_pred)

    @staticmethod
    def gen_ego_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types):
        return l5_utils.gen_ego_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types)

    @staticmethod
    def gen_EC_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types, mask=None):
        return l5_utils.gen_EC_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types, mask)

    @staticmethod
    def get_drivable_region_map(rasterized_map):
        return av_utils.get_drivable_region_map(rasterized_map)

    @staticmethod
    def get_modality_shapes(cfg: ExperimentConfig):
        return av_utils.get_modality_shapes(cfg)

