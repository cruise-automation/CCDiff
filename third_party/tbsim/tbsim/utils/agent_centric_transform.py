#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt
import networkx as nx
from ccdiff.configs.algo_config import CCDiffConfig

from trajdata.utils.arr_utils import angle_wrap
import tbsim.utils.geometry_utils as GeoUtils
def angle_wrap_torch(radians):
    pi = torch.tensor(np.pi, device=radians.device)
    return (radians + pi) % (2 * pi) - pi

def detect_cliques(matrix): 
    # Function to calculate total weight of a strongly connected component (treated as a clique)
    def component_weight(component, graph):
        return sum(graph[u][v]['weight'] for u in component for v in component if u != v and graph.has_edge(u, v))

    G = nx.Graph()
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] >= 0.5 or matrix[j][i] >= 0.5:  # You can apply a threshold here if needed
                G.add_edge(i, j, weight=max(matrix[i][j], matrix[j][i]))

    # Find all strongly connected components of size >= 3
    all_cliques = [list(component) for component in nx.enumerate_all_cliques(G) if len(component) >= 3]

    # Rank components based on total edge weights
    ranked_components = sorted(all_cliques, key=lambda component: component_weight(component, G), reverse=True)

    # Print the ranked components
    for rank, component in enumerate(ranked_components, 1):
        total_weight = component_weight(component, G)
        print(f"Rank {rank}: Component {component}, Total Weight: {total_weight}")

algo_config = CCDiffConfig()
diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
if 'neighbor_hist' in algo_config.nusc_norm_info:
    neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
if 'neighbor_fut' in algo_config.nusc_norm_info:
    neighbor_fut_norm_info = algo_config.nusc_norm_info['neighbor_fut']


def prepare_scene_agent_hist(pos, yaw, speed, extent, avail, norm_info, scale=True, speed_repr='abs_speed'):
    '''
    Input:
    - pos : (B, M, (Q), T, 2)
    - yaw : (B, M, (Q), T, 1)
    - speed : (B, M, (Q), T)
    - extent: (B, M, (Q), 3)
    - avail: (B, M, (Q), T)
    - norm_info: [2, 5]
    - speed_repr: 'abs_speed' or 'rel_vel'
    Output:
    - hist_in: [B, M, (Q), T, 8] (x,y,cos,sin,v,l,w,avail)
    '''
    M = pos.shape[1]
    T = pos.shape[-2]

    if speed_repr == 'rel_vel_per_step':
        # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
        yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
        # (B, M, M, T, 1) -> (B*M*M*T)
        yaw_self_ = yaw_self.reshape(-1)
        # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2) -> (B*M*M*T, 2)
        pos_self_ = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos).reshape(-1, 2)

        # (B, M1, M2, T, 2) -> (B*M1*M2*T, 2)
        pos_ = pos.view(-1, 2)
        
        # self_from_self_per_timestep
        # (B*M1*M2*T, 3, 3)
        i_from_i_per_time = torch.stack(
            [
                torch.stack([torch.cos(yaw_self_), -torch.sin(yaw_self_), pos_self_[..., 0]], dim=-1),
                torch.stack([torch.sin(yaw_self_), torch.cos(yaw_self_), pos_self_[..., 1]], dim=-1),
                torch.stack([0.0*torch.ones_like(yaw_self_), 0.0*torch.ones_like(yaw_self_), 1.0*torch.ones_like(yaw_self_)], dim=-1)
            ], dim=-2
        )
        i_per_time_from_i = torch.linalg.inv(i_from_i_per_time)
        
        # transform coord
        pos_transformed = torch.einsum("...jk,...k->...j", i_per_time_from_i[..., :-1, :-1], pos_)
        pos_transformed += i_per_time_from_i[..., :-1, -1]
        # (B*M1*M2*T, 2) -> (B, M1, M2, T, 2)
        pos = pos_transformed.view(pos.shape)

        # transform angle
        yaw = angle_wrap_torch(yaw - yaw_self)

        # print('pos.shape', pos.shape)
        # print('yaw.shape', yaw.shape)
        # print(pos[0, 2])
        # print(yaw[0, 2])
        # raise


    # convert yaw to heading vec
    hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # (B, M, (Q), T, 2)
    # only need length, width for pred
    lw = extent[..., :2].unsqueeze(-2).expand(pos.shape) # (B, M, (Q), T, 2)

    # Get time to collision
    if speed_repr in ['rel_vel', 'rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
        d_th = 20
        t_to_col_th = 20

        # print('pos.shape', pos.shape)
        # print('pos[0,0]', pos[0,0])
        # raise

        # estimate relative distance to other agents
        if speed_repr == 'rel_vel_new_new':
            # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
            pos_self = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos)
            pos_diff = torch.abs(pos.detach().clone() - pos_self)
        else:
            pos_diff = pos
        # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
        rel_d = torch.norm(pos_diff, dim=-1).unsqueeze(-1)
        
        # rel_d_lw also consider the lw of both agents (half of the mean of each lw)
        # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
        lw_avg_half = (torch.mean(lw, dim=-1) / 2).unsqueeze(-1)
        # (B, M, (Q), T, 1) -> (B, M, T, 1)
        ego_lw_avg_half = lw_avg_half[...,torch.arange(M), torch.arange(M), :, :]
        # (B, M, (Q), T, 1)
        lw_avg_half_sum = lw_avg_half + ego_lw_avg_half.unsqueeze(2).expand_as(lw_avg_half)
        # (B, M, (Q), T, 1)
        rel_d_lw = rel_d - lw_avg_half_sum
        # normalize rel_d and rel_d_lw
        rel_d = torch.clip(rel_d, min=0, max=d_th)
        rel_d = (d_th - rel_d) / d_th
        rel_d_lw = torch.clip(rel_d_lw, min=0, max=d_th)
        rel_d_lw = (d_th - rel_d_lw) / d_th

        B, M, M, T = speed.shape
        # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
        ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
        # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
        ego_lw = torch.diagonal(lw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand(B, M, M, T, 2).clone()
        ego_vx[torch.isnan(ego_vx)] = 0.0
        
        if speed_repr == 'rel_vel_new_new':
            # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
            yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
            hvec_self = torch.cat([torch.cos(yaw_self), torch.sin(yaw_self)], dim=-1)
            vx = ego_vx * hvec_self[...,0] - speed * hvec[...,0]
            vy = ego_vx * hvec_self[...,1] - speed * hvec[...,1]
        else:
            vx = ego_vx - speed * hvec[...,0]
            vy = 0 - speed * hvec[...,1]

        x_dist = pos_diff[...,0] - (ego_lw[...,0]/2) - (lw[...,0]/2)
        y_dist = pos_diff[...,1] - (ego_lw[...,1]/2) - (lw[...,1]/2)
        x_t_to_col = x_dist / vx
        y_t_to_col = y_dist / vy
        # if collision has not happened and moving in opposite direction, set t_to_col to t_to_col_th
        x_t_to_col[(x_dist>0) & (x_t_to_col<0)] = t_to_col_th
        y_t_to_col[(y_dist>0) & (y_t_to_col<0)] = t_to_col_th
        # if collision already happened, set t_to_col to 0
        x_t_to_col[x_dist<0] = 0
        y_t_to_col[y_dist<0] = 0
        # both directions need to be met for collision to happen
        rel_t_to_col = torch.max(torch.cat([x_t_to_col.unsqueeze(-1), y_t_to_col.unsqueeze(-1)], dim=-1), dim=-1)[0]
        rel_t_to_col = torch.clip(rel_t_to_col, min=0, max=t_to_col_th)
        # normalize rel_t_to_col
        rel_t_to_col = (t_to_col_th - rel_t_to_col.unsqueeze(-1)) / t_to_col_th
        matrix_ttc = rel_t_to_col[0, :, :, :, 0].detach().cpu().numpy()
        matrix_dist = rel_d[0, :, :, :, 0].detach().cpu().numpy()

    # normalize everything
    #  note: don't normalize hvec since already unit vector
    add_coeffs = torch.tensor(norm_info[0]).to(pos.device)
    div_coeffs = torch.tensor(norm_info[1]).to(pos.device)

    if len(pos.shape) == 4:
        add_coeffs_expand = add_coeffs[None, None, None, :]
        div_coeffs_expand = div_coeffs[None, None, None, :]
    else:
        add_coeffs_expand = add_coeffs[None, None, None, None, :]
        div_coeffs_expand = div_coeffs[None, None, None, None, :]

    pos_original = pos.detach().clone()
    if scale:
        pos = (pos + add_coeffs_expand[...,:2]) / div_coeffs_expand[...,:2]
        speed = (speed.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
        lw = (lw + add_coeffs_expand[...,3:]) / div_coeffs_expand[...,3:]
    else:
        speed = speed.unsqueeze(-1)
    
    if speed_repr in ['rel_vel', 'rel_vel_new', 'rel_vel_per_step', 'rel_vel_new_new']:
        speed = speed.squeeze(-1)
        B, M, M, T = speed.shape
        if speed_repr == 'rel_vel_new_new':
            # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M) -> (B, M, 1, 1) -> (B, M, M, T)
            ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1)[...,0].unsqueeze(-1).unsqueeze(-1).expand(B, M, M, T).clone()
            ego_vx[torch.isnan(ego_vx)] = 0.0
            vx = speed * hvec[...,0] - ego_vx
            vy = speed * hvec[...,1]
        else:
            # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
            ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
            ego_vx[torch.isnan(ego_vx)] = 0.0
            vx = speed * hvec[...,0] - ego_vx
            vy = speed * hvec[...,1]
        vvec = torch.cat([vx.unsqueeze(-1), vy.unsqueeze(-1)], dim=-1) # (B, M, M, T, 2)

        # also need to zero out the symmetric entries as we apply anothe matrix transformation
        if speed_repr in ['rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
            # (B, M1, M2, T) -> (B, M2, M1, T)
            avail_perm = avail.permute(0, 2, 1, 3)
            avail = avail * avail_perm

            hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
        elif speed_repr in ['rel_vel_new', 'rel_vel_new_new']:
            hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
        else:
            hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, avail.unsqueeze(-1)], dim=-1)
    elif speed_repr == 'abs_speed':
        # combine to get full input
        hist_in = torch.cat([pos, hvec, speed, lw, avail.unsqueeze(-1)], dim=-1)
    else:
        raise ValueError('Unknown speed representation: {}'.format(speed_repr))
    # zero out values we don't have data for
    hist_in[~avail] = 0.0
    if torch.isnan(hist_in).any():
        hist_in = torch.where(torch.isnan(hist_in), torch.zeros_like(hist_in), hist_in)

        # log nan values
        print('torch.where(torch.isnan(hist_in))', torch.where(torch.isnan(hist_in)))
    
    info = {
        "matrix_ttc": matrix_ttc, 
        "matrix_dist": matrix_dist
    }
    return hist_in, info


def get_neighbor_relative_states(relative_positions, relative_speeds, relative_yaws, data_batch_agent_from_world, data_batch_world_from_agent, data_batch_yaw, data_batch_extent):
    BN, M, _, _ = relative_positions.shape
    B = data_batch_agent_from_world.shape[0]
    N = int(BN // B)

    # [M, M]
    nb_idx = torch.arange(M).unsqueeze(0).repeat(M, 1)

    all_other_agents_relative_positions_list = []
    all_other_agents_relative_yaws_list = []
    all_other_agents_relative_speeds_list = []
    all_other_agents_extent_list = []

    # get relative states
    for k in range(BN):
        i = int(k // N)
        agent_from_world = data_batch_agent_from_world[i]
        world_from_agent = data_batch_world_from_agent[i]

        all_other_agents_relative_positions_list_sub = []
        all_other_agents_relative_yaws_list_sub = []
        all_other_agents_relative_speeds_list_sub = []
        all_other_agents_extent_list_sub = []

        for j in range(M):
            chosen_neigh_inds = nb_idx[j][nb_idx[j]>=0].tolist()

            # (Q. 3. 3)
            center_from_world = agent_from_world[j]
            world_from_neigh = world_from_agent[chosen_neigh_inds]
            center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

            fut_neigh_pos_b_sub = relative_positions[k][chosen_neigh_inds]
            fut_neigh_yaw_b_sub = relative_yaws[k][chosen_neigh_inds]

            all_other_agents_relative_positions_list_sub.append(GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub,center_from_neigh))
            all_other_agents_relative_yaws_list_sub.append(fut_neigh_yaw_b_sub+data_batch_yaw[i][chosen_neigh_inds][:,None,None]-data_batch_yaw[i][j])
            all_other_agents_relative_speeds_list_sub.append(relative_speeds[k][chosen_neigh_inds])
            all_other_agents_extent_list_sub.append(data_batch_extent[i][chosen_neigh_inds])

        all_other_agents_relative_positions_list.append(pad_sequence(all_other_agents_relative_positions_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_relative_yaws_list.append(pad_sequence(all_other_agents_relative_yaws_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_relative_speeds_list.append(pad_sequence(all_other_agents_relative_speeds_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_extent_list.append(pad_sequence(all_other_agents_extent_list_sub, batch_first=True, padding_value=0))

    max_second_dim = max(a.size(1) for a in all_other_agents_relative_positions_list)

    all_other_agents_relative_positions = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_positions_list], dim=0)
    all_other_agents_relative_yaws = angle_wrap(torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_yaws_list], dim=0))
    all_other_agents_relative_speeds = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_speeds_list], dim=0)
    all_other_agents_extents = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_extent_list], dim=0)

    return all_other_agents_relative_positions, all_other_agents_relative_yaws, all_other_agents_relative_speeds, all_other_agents_extents

        
def get_neighbor_history_relative_states(data_batch: dict, include_class_free_cond: bool=False) -> dict:
    '''
    get the neighbor history relative states (only need once per data_batch). We do this because fields like all_other_agents_history_positions in data_batch may not include all the agents controlled and may include agents not controlled.

    - output neighbor_hist: (B, M, M, T_hist, K_vehicle)
    - output neighbor_hist_non_cond: (B, M, M, T_hist, K_vehicle)
    '''
    M = data_batch['history_positions'].shape[1]

    all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents = get_neighbor_relative_states(data_batch['history_positions'], data_batch['history_speeds'], data_batch['history_yaws'], data_batch['agent_from_world'], data_batch['world_from_agent'], data_batch['yaw'], data_batch["extent"])

    # (B, M, T_hist) -> (B, 1, M, T_hist) -> (B, M, M, T_hist)
    all_other_agents_history_availabilities = data_batch["history_availabilities"].unsqueeze(1).repeat(1,M,1,1)

    neighbor_hist = prepare_scene_agent_hist(all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents, all_other_agents_history_availabilities, neighbor_hist_norm_info, scale=True, speed_repr='rel_vel_new_new')

    # # (B, M, M, T_hist, K_vehicle) -> (B, M, M, T_hist, K_neigh)
    # neighbor_hist_feat = self.neighbor_hist_encoder(neighbor_hist)

    # Disable the classifier-free conditions of other agents
    # if include_class_free_cond:
    #     all_other_agents_history_availabilities_non_cond = torch.zeros_like(all_other_agents_history_availabilities, dtype=torch.bool, device=all_other_agents_history_availabilities.device)
    #     neighbor_hist_non_cond = prepare_scene_agent_hist(all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents, all_other_agents_history_availabilities_non_cond, neighbor_hist_norm_info, scale=True, speed_repr='rel_vel_new_new')
    #     # # (B, M, M, T_fut, K_vehicle) -> (B, M, M, T_fut, K_neigh)
    #     neighbor_hist_feat_non_cond = self.neighbor_hist_encoder(neighbor_hist_non_cond)
    # else:
    #     neighbor_hist_feat_non_cond = None
    #     neighbor_hist_non_cond = None
    neighbor_hist_non_cond = None

    return neighbor_hist, neighbor_hist_non_cond
