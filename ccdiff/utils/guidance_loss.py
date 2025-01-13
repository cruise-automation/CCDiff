#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from tbsim.utils.guidance_loss import *

############## ITERATIVE PERTURBATION ########################
class PerturbationGuidance(object):
    """
    Guide trajectory to satisfy rules by directly perturbing it
    """
    def __init__(self, transform, transform_params, scale_traj=lambda x,y:x, descale_traj=lambda x,y:x, controllable_agent=-1) -> None:
        
        self.transform = transform
        self.transform_params = transform_params
        
        self.scale_traj = scale_traj
        self.descale_traj = descale_traj

        self.current_guidance = None
        self.controllable_agent = controllable_agent # -1
        # print('controllable agents: ', self.controllable_agent)

    def update(self, **kwargs):
        self.current_guidance.update(**kwargs)

    def set_guidance(self, guidance_config_list, example_batch=None):
        self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        self.current_guidance = None

    def perturb_actions_dict(self, actions_dict, data_batch, opt_params, num_samp=1):
        """Given the observation object, add Gaussian noise to positions and yaws

        Args:
            data_batch(Dict[torch.tensor]): observation dict

        Returns:
            data_batch(Dict[torch.tensor]): perturbed observation
        """
        x_initial = torch.cat((actions_dict["target_positions"], actions_dict["target_yaws"]), dim=-1)

        x_guidance, _ = self.perturb(x_initial, data_batch, opt_params, num_samp)
        # print('x_guidance.shape', x_guidance.shape)
        # x_guidance: [B*N, T, 3]
        actions_dict["target_positions"] = x_guidance[..., :2].type(torch.float32)
        actions_dict["target_yaws"] = x_guidance[..., 2:3].type(torch.float32)
        
        return actions_dict

    def perturb(self, x_initial, data_batch, opt_params, num_samp=1, decoder=None, return_grad_of=None):
        '''
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            decoder: decode the perturbed variable to get the trajectory.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial

        # x_guidance may not have gradient enabled when BITS is used
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance_reshaped = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        else:
            x_guidance_reshaped = x_guidance
        # print("X guide: ", x_guidance.shape)
        # guide_dim = min(15, x_guidance.shape[0])
        # guide_dim = x_guidance.shape[-3]
        if self.controllable_agent > 0: 
            # print("controllable_agent: ", self.controllable_agent)
            guide_dim = min(self.controllable_agent, x_guidance.shape[-3]) # for CCDiff    
        else: 
            guide_dim = x_guidance.shape[-3]
        # print("guide dim: ", guide_dim, x_guidance.shape)
        if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
            # opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
            if len(x_guidance.shape) == 3: 
                opt = torch.optim.Adam([x_guidance[:guide_dim]], lr=opt_params['lr'])
            elif len(x_guidance.shape) == 4: 
                opt = torch.optim.Adam([x_guidance[:, :guide_dim]], lr=opt_params['lr'])
        elif opt_params['optimizer'] == 'sgd':
            # opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
            if len(x_guidance.shape) == 3: 
                opt = torch.optim.SGD([x_guidance[:guide_dim]], lr=opt_params['lr'])
            elif len(x_guidance.shape) == 4:
                opt = torch.optim.SGD([x_guidance[:, :guide_dim]], lr=opt_params['lr'])
        else: 
            raise NotImplementedError('Optimizer not implemented')
        # if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
        #     opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
        # elif opt_params['optimizer'] == 'sgd':
        #     opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                # for CVAE, we need to decode the latent
                if decoder is not None:
                    x_guidance_decoded = decoder(x_guidance_reshaped)
                else:
                    x_guidance_decoded = x_guidance_reshaped
                bsize = int(x_guidance_decoded.size(0) / num_samp)

                x_all = self.transform(x_guidance_decoded, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)

                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            opt.step()
            opt.zero_grad()
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta = x_guidance - x_initial
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped
        # print('x_guidance.data - x_initial', x_guidance.data - x_initial)
        # print("per_losses: ", per_losses)
        return x_guidance, per_losses
    
    def perturb_video_diffusion(self, x_initial, data_batch, opt_params, num_samp=1, return_grad_of=None):
        '''
        video_diffusion only
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            return_grad_of: apply the gradient to which variable.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'

        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial

        if len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)

        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                bsize = int(x_guidance.size(0) / num_samp)

                x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
                
                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            x_delta = opt_params['lr'] * return_grad_of.grad
            if x_initial.shape[-1] == 2:
                # only need the grad w.r.t noisy action
                x_delta = x_delta[..., [4,5]]

            x_guidance = x_initial + x_delta
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped

        return x_guidance, per_losses
    
    @torch.no_grad()
    def compute_guidance_loss(self, x_initial, data_batch, num_samp=1):
        '''
        -x_initial: [B*N, T, 2/3]
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        if len(x_initial.shape) == 4:
            BN, M, T, _ = x_initial.shape
            B = int(BN // num_samp)
            x_initial = x_initial.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        bsize = int(x_initial.size(0) / num_samp)
        num_t = x_initial.size(1)

        x_initial_copy = x_initial.clone().detach()
        x_guidance = Variable(x_initial_copy, requires_grad=True)

        x_all = self.transform(x_guidance, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)
        
        x_loss = x_all.reshape((bsize, num_samp, num_t, 6))

        tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)

        return tot_loss, per_losses


# TODO: partially controllable settings, to avoid the NaN issues!
############## ITERATIVE PERTURBATION ########################
class PartialPerturbationGuidance(PerturbationGuidance):
    """
    Guide trajectory to satisfy rules by directly perturbing it
    """
    def __init__(self, transform, transform_params, scale_traj=lambda x,y:x, descale_traj=lambda x,y:x, available_idx=None) -> None:
        super().__init__(transform, transform_params, scale_traj, descale_traj)
        self.available_idx = available_idx
        self.controllable_ts = 52
        print("initialized.!!!!! with controllable ts: ", self.controllable_ts)
    
    def set_controllable_ts(self, ts): 
        self.controllable_ts = ts
    
    def perturb(self, x_initial, data_batch, opt_params, num_samp=1, decoder=None, return_grad_of=None):
        '''
        perturb the gradient and estimate the guidance loss w.r.t. the input trajectory
        Input:
            x_initial: [batch_size*num_samp, (num_agents), time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            opt_params: optimization parameters.
            num_samp: number of samples in x_initial.
            decoder: decode the perturbed variable to get the trajectory.
            return_grad_of: apply the gradient to which variable.
        '''
        # self.available_idx = [0, 6, 7, 3, 14]

        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        # print('PERTURB: ', self.available_idx)
        perturb_th = opt_params['perturb_th']
        
        x_guidance = x_initial
        # x_guidance may not have gradient enabled when BITS is used
        if not x_guidance.requires_grad:
            x_guidance.requires_grad_()
        if len(x_guidance.shape) == 2:
            NM, _ = x_guidance.shape
            N = int(NM // num_samp)
            x_guidance_reshaped = x_guidance.clone() #.reshape(N, num_samp, -1)
            x_guidance = x_guidance.reshape(N, num_samp, -1)
            # print("VAE: ", x_guidance.shape)
        elif len(x_guidance.shape) == 4:
            with torch.enable_grad():
                BN, M, T, _ = x_guidance.shape
                B = int(BN // num_samp)
                x_guidance_reshaped = x_guidance.reshape(B, num_samp, M, T, -1).permute(0, 2, 1, 3, 4).reshape(B*M*num_samp, T, -1)
        else:
            x_guidance_reshaped = x_guidance
        # TODO: perturb on all
        # self.available_idx = None
        if self.available_idx is not None: 
            guide_dim = x_guidance.shape[-3]
            available_idx = self.available_idx[:guide_dim]
            # available_idx = [i for i in range(guide_dim)]
            # print(x_guidance.shape, x_guidance_reshaped.shape, self.available_idx)
            if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
                # opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
                if len(x_guidance.shape) == 3: 
                    opt = torch.optim.Adam([x_guidance[available_idx, :self.controllable_ts]], lr=opt_params['lr'])
                elif len(x_guidance.shape) == 4: 
                    opt = torch.optim.Adam([x_guidance[:, available_idx, :self.controllable_ts]], lr=opt_params['lr'])
            elif opt_params['optimizer'] == 'sgd':
                # opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])
                if len(x_guidance.shape) == 3: 
                    opt = torch.optim.SGD([x_guidance[available_idx, :self.controllable_ts]], lr=opt_params['lr'])
                elif len(x_guidance.shape) == 4:
                    opt = torch.optim.SGD([x_guidance[:, available_idx, :self.controllable_ts]], lr=opt_params['lr'])
        else: 
            if opt_params['optimizer'] == 'adam': # TODO: different for diffusion and CCDiff
                opt = torch.optim.Adam([x_guidance], lr=opt_params['lr'])
            elif opt_params['optimizer'] == 'sgd':
                opt = torch.optim.SGD([x_guidance], lr=opt_params['lr'])        

        # if opt_params['optimizer'] == 'adam':
        #     opt = torch.optim.Adam([x_guidance[available_idx]], lr=opt_params['lr'])
        # elif opt_params['optimizer'] == 'sgd':
        #     opt = torch.optim.SGD([x_guidance[available_idx]], lr=opt_params['lr'])
        # else: 
        #     raise NotImplementedError('Optimizer not implemented')
        
        per_losses = dict()
        for _ in range(opt_params['grad_steps']):
            with torch.enable_grad():
                # for CVAE, we need to decode the latent
                if decoder is not None:
                    x_guidance_decoded = decoder(x_guidance_reshaped)
                else:
                    x_guidance_decoded = x_guidance_reshaped
                bsize = int(x_guidance_decoded.size(0) / num_samp)

                x_all = self.transform(x_guidance_decoded, data_batch, self.transform_params, bsize=bsize, num_samp=num_samp)

                x_loss = x_all.reshape((bsize, num_samp, -1, 6))
                tot_loss, per_losses = self.current_guidance.compute_guidance_loss(x_loss, data_batch)
            
            tot_loss.backward()
            opt.step()
            opt.zero_grad()
            if perturb_th is not None:
                with torch.no_grad():
                    x_delta = x_guidance - x_initial
                    x_delta_clipped = torch.clip(x_delta, -1*perturb_th, perturb_th)
                    x_guidance.data = x_initial + x_delta_clipped
                    if x_delta.norm() > 0: 
                        print('X Delta: ', x_delta.norm(), self.available_idx)

        # print('x_guidance.data - x_initial', x_guidance.data - x_initial)

        return x_guidance, per_losses

    def set_perturbation_idx(self, idx):
        # assert len(idx) <= len(self.available_idx), (idx, self.available_idx)
        self.available_idx = copy.deepcopy(idx)
        if self.available_idx is not None: 
            print("self.available_idx: ", self.available_idx)
            print("set guidance done")