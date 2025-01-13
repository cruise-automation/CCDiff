#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from collections import OrderedDict
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import tbsim.utils.torch_utils as TorchUtils

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action

from ccdiff.models.diffuser_helpers import EMA
from ccdiff.models.ccdiff import CCDiffuserModel
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask

class CCDiffTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name, do_log=True, guidance_config=None, constraint_config=None):
        """
        The traffic model for the CCDiff
        """
        super(CCDiffTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate

        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None
        
        if algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4 
            action_dim = 2
            output_dim = 2
        else:
            raise
        
        print('registered_name', registered_name)
        diffuser_norm_info = ([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4])
        agent_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_fut_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        if 'nusc' in registered_name:
            diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nusc_norm_info:
                neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nusc_norm_info:
                neighbor_fut_norm_info = algo_config.nusc_norm_info['neighbor_fut']
        else:
            raise


        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        self.cond_drop_graph_p = algo_config.conditioning_drop_graph_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            print(self.cond_drop_graph_p)
            if self.cond_drop_graph_p > 0:
                print('DIFFUSER: Dropping Causal Graph mask conditioning with p = %f during training...' % (self.cond_drop_map_p))
                

        self.nets["policy"] = CCDiffuserModel(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames+1, # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim, 
            action_dim=action_dim,

            output_dim=output_dim,

            n_timesteps=algo_config.n_diffusion_steps,
            
            loss_type=algo_config.loss_type, 
            clip_denoised=algo_config.clip_denoised,

            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight, 
            loss_discount=algo_config.loss_discount, 
            loss_weights=algo_config.loss_weights,
            loss_decay_rates=algo_config.loss_decay_rates,

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            action_loss_only = algo_config.action_loss_only,
            
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,
            neighbor_fut_norm_info=neighbor_fut_norm_info,

            agent_hist_embed_method=algo_config.agent_hist_embed_method,
            neigh_hist_embed_method=algo_config.neigh_hist_embed_method,
            map_embed_method=algo_config.map_embed_method,
            interaction_edge_speed_repr=algo_config.interaction_edge_speed_repr,
            normalize_rel_states=algo_config.normalize_rel_states,
            mask_social_interaction=algo_config.mask_social_interaction,
            mask_edge=algo_config.mask_edge,
            neighbor_inds=algo_config.neighbor_inds,
            edge_attr_separation=algo_config.edge_attr_separation,
            social_attn_radius=algo_config.social_attn_radius,
            use_last_hist_step=algo_config.use_last_hist_step,
            use_noisy_fut_edge=algo_config.use_noisy_fut_edge,
            use_const_speed_edge=algo_config.use_const_speed_edge,
            all_interactive_social=algo_config.all_interactive_social,
            mask_time=algo_config.mask_time,
            layer_num_per_edge_decoder=algo_config.layer_num_per_edge_decoder,
            attn_combination=algo_config.attn_combination,
            single_cond_feat=algo_config.single_cond_feat,

            disable_control_on_stationary=self.disable_control_on_stationary,

            coordinate=self.coordinate,

            # controllable_agent=algo_config.controllable_agent,
        )

        # set up initial guidance and constraints
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}
    
    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        if self.disable_control_on_stationary and global_t == 0:
            self.stationary_mask = get_stationary_mask(obs_dict, self.disable_control_on_stationary, self.moving_speed_th)
            B, M = self.stationary_mask.shape
            # (B, M) -> (B, N, M) -> (B*N, M)
            stationary_mask_expand =  self.stationary_mask.unsqueeze(1).expand(B, num_samp, M).reshape(B*num_samp, M)
        else:
            stationary_mask_expand = None
        cur_policy = self.nets["policy"]

        # this function is only called at validation time, so use ema
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        return cur_policy(obs_dict, plan, num_samp, return_diffusion=True,
                                   return_guidance_losses=True, class_free_guide_w=class_free_guide_w,
                                   apply_guidance=(not guide_as_filter_only),
                                   guide_clean=guide_clean, global_t=global_t, stationary_mask=stationary_mask_expand)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        sample_preds = predictions["positions"]
        B, N, M, T, _ = sample_preds.shape
        # (B, N, M, T, 2) -> (B, M, N, T, 2) -> (B*M, N, T, 2)
        sample_preds = TensorUtils.to_numpy(sample_preds.permute(0, 2, 1, 3, 4).reshape(B*M, N, T, -1))
        # (B, M, T, 2) -> (B*M, T, 2)
        gt = TensorUtils.to_numpy(data_batch["target_positions"].reshape(B*M, T, -1))
        # (B, M, T) -> (B*M, T)
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"].reshape(B*M, T))
        
        # compute ADE & FDE based on trajectory samples
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if self.use_ema and self.cur_train_step % self.ema_update_every == 0:
            self.step_ema(self.cur_train_step)
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
       
        # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(1) # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.use_rasterized_hist:
                # drop layers of map corresponding to histories
                num_sem_layers = batch['maps'].size(1) if batch['maps'] is not None else None
                if self.cond_drop_neighbor_p > 0:
                    # sample different mask so sometimes both get dropped, sometimes only one
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_neighbor_p
                    if num_sem_layers is None:
                        batch["image"][drop_mask] = self.cond_fill_val
                    else:
                        # only fill the layers before semantic map corresponding to trajectories (neighbors and ego)
                        batch["image"][drop_mask, :-num_sem_layers] = self.cond_fill_val
        
        # diffuser only take the data to estimate loss
        losses = self.nets["policy"].compute_losses(batch)

        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        # metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }
    
    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["policy"]
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
        
        losses = TensorUtils.detach(cur_policy.compute_losses(batch))
        
        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict


    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)
        
        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
        )

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}
    
    def get_action(self, obs_dict,
                    num_action_samples=1,
                    class_free_guide_w=0.0, 
                    guide_as_filter_only=False,
                    guide_with_gt=False,
                    guide_clean=False,
                    **kwargs):
        plan = kwargs.get("plan", None)

        cur_policy = self.nets["policy"]

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)

        # already called in policy_composer, but just for good measure...
        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])
        preds = self(obs_dict, plan, num_samp=num_action_samples,
                    class_free_guide_w=class_free_guide_w, guide_as_filter_only=guide_as_filter_only,
                    guide_clean=guide_clean, global_t=kwargs['step_index']) 

        # [B, N, M, T, 2]
        B, N, M, _, _ = preds["positions"].shape

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((M), dtype=torch.long, device=preds["positions"].device)
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_perturbation_guidance.current_guidance is not None:
            # choose sample closest to desired guidance
            guide_losses = preds.pop("guide_losses", None)
            act_idx = choose_action_from_guidance(preds, obs_dict, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, guide_losses)          

        def map_act_idx(x):
            # Assume B == 1 during generation. TBD: need to change this to support general batchsize
            if len(x.shape) == 4:
                # [N, T, M1, M2] -> [M1, N, T, M2]
                x = x.permute(2,0,1,3)
            elif len(x.shape) == 5:
                # [B, N, M, T, 2] -> [N, M, T, 2] -> [M, N, T, 2]
                x = x[0].permute(1,0,2,3)
            elif len(x.shape) == 6: # for "diffusion_steps"
                x = x[0].permute(1,0,2,3,4)
            else:
                raise NotImplementedError
            # [M, N, T, 2] -> [M, T, 2]
            x = x[torch.arange(M), act_idx]
            # [M, T, 2] -> [B, M, T, 2]
            x = x.unsqueeze(0)
            return x
        
        preds_positions = preds["positions"]
        preds_yaws = preds["yaws"]
        preds_trajectories = preds["trajectories"]

        action_preds_positions = map_act_idx(preds_positions)
        action_preds_yaws = map_act_idx(preds_yaws)
        
        attn_weights = map_act_idx(preds["attn_weights"]).squeeze(0)

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N, M)
            
            preds_positions[stationary_mask_expand] = 0
            preds_yaws[stationary_mask_expand] = 0
            preds_trajectories[stationary_mask_expand] = 0

            action_preds_positions[self.stationary_mask] = 0
            action_preds_yaws[self.stationary_mask] = 0

        info = dict(
            action_samples=Action(
                positions=preds_positions, # (B, N, M, T, 2)
                yaws=preds_yaws
            ).to_dict(),
            trajectories=preds_trajectories,
            act_idx=act_idx,
            dyn=self.nets["policy"].dyn,
            attn_weights=attn_weights,
        )
        action = Action(
            positions=action_preds_positions, # (B, M, T, 2)
            yaws=action_preds_yaws
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
        cur_policy.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()


    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_constraints(constraint_config)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_optimization_params(guidance_optimization_params)
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
    
    def set_diffusion_specific_params(self, diffusion_specific_params):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_diffusion_specific_params(diffusion_specific_params)
        cur_policy.set_guidance_dim(self.nets["policy"].current_perturbation_guidance.available_idx)
