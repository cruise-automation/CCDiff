#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import torch
from torch import optim as optim

import tbsim.utils.tensor_utils as TensorUtils
from tbsim import dynamics as dynamics
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
from tbsim.utils.loss_utils import goal_reaching_loss, trajectory_loss, collision_loss

def optimize_trajectories(
        init_u,
        init_x,
        target_trajs,
        target_avails,
        dynamics_model,
        step_time: float,
        data_batch=None,
        goal_loss_weight=1.0,
        traj_loss_weight=0.0,
        coll_loss_weight=0.0,
        num_optim_iterations: int = 50
):
    """An optimization-based trajectory generator"""
    curr_u = init_u.detach().clone()
    curr_u.requires_grad = True
    action_optim = optim.LBFGS(
        [curr_u], max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

    for oidx in range(num_optim_iterations):
        def closure():
            action_optim.zero_grad()

            # get trajectory with current params
            _, pos, yaw = dynamics.forward_dynamics(
                dyn_model=dynamics_model,
                initial_states=init_x,
                actions=curr_u,
                step_time=step_time
            )
            curr_trajs = torch.cat((pos, yaw), dim=-1)
            # compute trajectory optimization losses
            losses = dict()
            losses["goal_loss"] = goal_reaching_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * goal_loss_weight
            losses["traj_loss"] = trajectory_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * traj_loss_weight
            if coll_loss_weight > 0:
                assert data_batch is not None
                coll_edges = batch_utils().get_edges_from_batch(
                    data_batch,
                    ego_predictions=dict(positions=pos, yaws=yaw)
                )
                for c in coll_edges:
                    coll_edges[c] = coll_edges[c][:, :target_trajs.shape[-2]]
                vv_edges = dict(VV=coll_edges["VV"])
                if vv_edges["VV"].shape[0] > 0:
                    losses["coll_loss"] = collision_loss(
                        vv_edges) * coll_loss_weight

            total_loss = torch.hstack(list(losses.values())).sum()

            # backprop
            total_loss.backward()
            return total_loss
        action_optim.step(closure)

    final_raw_trajs, final_pos, final_yaw = dynamics.forward_dynamics(
        dyn_model=dynamics_model,
        initial_states=init_x,
        actions=curr_u,
        step_time=step_time
    )
    final_trajs = torch.cat((final_pos, final_yaw), dim=-1)
    losses = dict()
    losses["goal_loss"] = goal_reaching_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )
    losses["traj_loss"] = trajectory_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )

    return dict(positions=final_pos, yaws=final_yaw), final_raw_trajs, curr_u, losses


def combine_ego_agent_data(batch, ego_keys, agent_keys, mask=None):
    assert len(ego_keys) == len(agent_keys)
    combined_batch = dict()
    for ego_key, agent_key in zip(ego_keys, agent_keys):
        if mask is None:
            size_dim0 = batch[agent_key].shape[0]*batch[agent_key].shape[1]
            combined_batch[ego_key] = torch.cat((batch[ego_key], batch[agent_key].reshape(
                size_dim0, *batch[agent_key].shape[2:])), dim=0)
        else:
            size_dim0 = mask.sum()
            combined_batch[ego_key] = torch.cat((batch[ego_key], batch[agent_key][mask].reshape(
                size_dim0, *batch[agent_key].shape[2:])), dim=0)
    return combined_batch


def yaw_from_pos(pos: torch.Tensor, dt, yaw_correction_speed=0.):
    """
    Compute yaws from position sequences. Optionally suppress yaws computed from low-velocity steps

    Args:
        pos (torch.Tensor): sequence of positions [..., T, 2]
        dt (float): delta timestep to compute speed
        yaw_correction_speed (float): zero out yaw change when the speed is below this threshold (noisy heading)

    Returns:
        accum_yaw (torch.Tensor): sequence of yaws [..., T-1, 1]
    """

    pos_diff = pos[..., 1:, :] - pos[..., :-1, :]
    yaw = torch.atan2(pos_diff[..., 1], pos_diff[..., 0])
    delta_yaw = torch.cat((yaw[..., [0]], yaw[..., 1:] - yaw[..., :-1]), dim=-1)
    speed = torch.norm(pos_diff, dim=-1) / dt
    delta_yaw[speed < yaw_correction_speed] = 0.
    accum_yaw = torch.cumsum(delta_yaw, dim=-1)
    return accum_yaw[..., None]
