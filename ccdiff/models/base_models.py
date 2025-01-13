#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import numpy as np
import math
import textwrap
from collections import OrderedDict
from typing import Dict, Union, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from tbsim.utils.tensor_utils import reshape_dimensions, flatten
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from ccdiff.utils.guidance_loss import PerturbationGuidance, PartialPerturbationGuidance


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        U-Net forward by concatenating input feature (x1) with mirroring encoding feature maps channel-wise (x2)
        Args:
            x1 (torch.Tensor): [B, C1, H1, W1]
            x2 (torch.Tensor): [B, C2, H2, W2]

        Returns:
            output (torch.Tensor): [B, out_channels, H2, W2]
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out



class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out



class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_dim (int): dimension of outputs
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_shape={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self.output_shape(), self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class SplitMLP(MLP):
    """
    A multi-output MLP network: The model split and reshapes the output layer to the desired output shapes
    """

    def __init__(
            self,
            input_dim: int,
            output_shapes: OrderedDict,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_shapes (dict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """

        assert isinstance(output_shapes, OrderedDict)
        output_dim = 0
        for v in output_shapes.values():
            output_dim += np.prod(v)
        self._output_shapes = output_shapes

        super(SplitMLP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def output_shape(self, input_shape=None):
        return self._output_shapes

    def forward(self, inputs):
        outs = super(SplitMLP, self).forward(inputs)
        out_dict = dict()
        ind = 0
        for k, v in self._output_shapes.items():
            v_dim = int(np.prod(v))
            out_dict[k] = reshape_dimensions(
                outs[:, ind: ind + v_dim], begin_axis=1, end_axis=2, target_dims=v)
            ind += v_dim
        return out_dict



class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
            self,
            input_shape,
            num_kp=None,
            temperature=1.,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list, tuple): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(
            1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat(
                [var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(),
                        feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            feature_dim: int = None,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            output_activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = input_image_shape[0]
        self._feature_dim = feature_dim
        if output_activation is None:
            self._output_activation = nn.Identity()
        else:
            self._output_activation = output_activation()

        # configure conv backbone
        if model_arch == "resnet18":
            self.map_model = resnet18()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (512, out_h, out_w)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (2048, out_h, out_w)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        # configure spatial reduction pooling layer
        if use_spatial_softmax:
            pooling = SpatialSoftmax(
                input_shape=self.conv_out_shape, **spatial_softmax_kwargs)
            self.pool_out_dim = int(
                np.prod(pooling.output_shape(self.conv_out_shape)))
        else:
            pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_out_dim = self.conv_out_shape[0]

        self.map_model.conv1 = nn.Conv2d(
            self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.map_model.avgpool = pooling
        if feature_dim is not None:
            self.map_model.fc = nn.Linear(
                in_features=self.pool_out_dim, out_features=feature_dim)
        else:
            self.map_model.fc = nn.Identity()

    def output_shape(self, input_shape=None):
        if self._feature_dim is not None:
            return [self._feature_dim]
        else:
            return [self.pool_out_dim]

    def feature_channels(self):
        if self.model_arch in ["resnet18", "resnet34"]:
            channels = OrderedDict({
                "layer1": 64,
                "layer2": 128,
                "layer3": 256,
                "layer4": 512,
            })
        else:
            channels = OrderedDict({
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
            })
        return channels

    def feature_scales(self):
        return OrderedDict({
            "layer1": 1/4,
            "layer2": 1/8,
            "layer3": 1/16,
            "layer4": 1/32
        })

    def forward(self, map_inputs) -> torch.Tensor:
        feat = self.map_model(map_inputs)
        feat = self._output_activation(feat)
        return feat

class TrajectoryDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            network_kwargs: dict = None,
            Gaussian_var = False
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
            Gaussian_var (bool): whether output the variance of the predicted trajectory
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self._network_kwargs = network_kwargs
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self.Gaussian_var = Gaussian_var
        self._create_dynamics()
        self._create_networks()

        # TBD: clean this up
        from ccdiff.models.diffuser_helpers import state_grad_general_transform
        # self.current_guidance = None
        self.current_perturbation_guidance = PartialPerturbationGuidance(state_grad_general_transform, {'dt': 0.1}, available_idx=[i for i in range(5)])
        self.guidance_optimization_params = None

    def set_guidance_dim(self, idx): 
        self.current_perturbation_guidance.set_perturbation_idx(idx=idx)

    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        elif self._dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=self._dynamics_kwargs["acce_bound"],
                ddh_bound=self._dynamics_kwargs["ddh_bound"],
                max_hdot=self._dynamics_kwargs["max_yawvel"],
                max_speed=self._dynamics_kwargs["max_speed"]
            )
        else:
            self.dyn = None

    def _create_networks(self):
        raise NotImplementedError

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        raise NotImplementedError

    def _forward_dynamics(self, current_states, actions):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        x, pos, yaw = dynamics.forward_dynamics(
            self.dyn,
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj,x

    def forward(self, inputs, current_states=None, num_steps=None, with_guidance=False, data_batch=None, num_samp=1):
        preds = self._forward_networks(
            inputs, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            if with_guidance:
                def decoder_wrapper(controls):
                    # TBD: hack on compatible with both BITS and BC
                    if len(current_states.shape) == 3:
                        current_states_join = TensorUtils.join_dimensions(current_states, 0, 2)
                    else:
                        current_states_join = current_states
                    trajectories, _ = self._forward_dynamics(current_states=current_states_join, actions=controls)
                    return trajectories
                # TBD: hack on compatible with both BITS and BC
                if len(preds["trajectories"].shape) == 4:
                    preds_trajectories_join = TensorUtils.join_dimensions(preds["trajectories"], 0, 2) # [B*N, T, 3]
                else:
                    preds_trajectories_join = preds["trajectories"]
                controls, _ = self.current_perturbation_guidance.perturb(preds_trajectories_join, data_batch, self.guidance_optimization_params, num_samp=num_samp, decoder=decoder_wrapper)
                # TBD: hack on compatible with both BITS and BC
                if len(preds["trajectories"].shape) == 4:
                    controls = controls[:, None, ...]

                preds["controls"] = controls
            else:
                preds["controls"] = preds["trajectories"]
            preds["trajectories"], x = self._forward_dynamics(
                current_states=current_states,
                actions=preds["controls"]
            )
            preds["terminal_state"] = x[...,-1,:]
            preds["states"] = x
        return preds

    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0:
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_perturbation_guidance.set_guidance(guidance_config_list, example_batch)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.guidance_optimization_params = guidance_optimization_params
    
    def clear_guidance(self):
        self.current_perturbation_guidance.clear_guidance()


class MLPTrajectoryDecoder(TrajectoryDecoder):
    def _create_networks(self):
        net_kwargs = dict() if self._network_kwargs is None else dict(self._network_kwargs)
        if self._network_kwargs is None:
            net_kwargs = dict()
        
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.dyn.udim))
        if self.Gaussian_var:
            pred_shapes["logvar"] = (self.num_steps, self.state_dim)

        state_as_input = net_kwargs.pop("state_as_input")
        if self.dyn is not None:
            assert state_as_input   # TODO: deprecated, set default to True and remove from configs

        if state_as_input and self.dyn is not None:
            feature_dim = self.feature_dim + self.dyn.xdim
        else:
            feature_dim = self.feature_dim

        self.mlp = SplitMLP(
            input_dim=feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        if self._network_kwargs["state_as_input"] and self.dyn is not None:
            inputs = torch.cat((inputs, current_states), dim=-1)

        if inputs.ndim == 2:
            # [B, D]
            preds = self.mlp(inputs)
        elif inputs.ndim == 3:
            # [B, A, D]
            preds = TensorUtils.time_distributed(inputs, self.mlp)
        else:
            raise ValueError(
                "Expecting inputs to have ndim == 2 or 3, got {}".format(inputs.ndim))
        return preds


class MLPECTrajectoryDecoder(TrajectoryDecoder):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            EC_feature_dim = 64,
            network_kwargs: dict = None,
            Gaussian_var = False,
            
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
            Gaussian_var (bool): whether output the variance of the predicted trajectory
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self.EC_feature_dim = EC_feature_dim
        self._network_kwargs = network_kwargs
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self.Gaussian_var = Gaussian_var
        self._create_dynamics()
        self._create_networks()

    def _create_networks(self):
        net_kwargs = dict() if self._network_kwargs is None else dict(self._network_kwargs)
        if self._network_kwargs is None:
            net_kwargs = dict()
        
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.dyn.udim))
        if self.Gaussian_var:
            pred_shapes["logvar"] = (self.num_steps, self.state_dim)

        state_as_input = net_kwargs.pop("state_as_input")
        if self.dyn is not None:
            assert state_as_input   # TODO: deprecated, set default to True and remove from configs

        if state_as_input and self.dyn is not None:
            feature_dim = self.feature_dim + self.dyn.xdim
        else:
            feature_dim = self.feature_dim

        self.mlp = SplitMLP(
            input_dim=feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )
        self.offsetmlp = SplitMLP(
            input_dim=feature_dim+self.EC_feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, EC_feat=None, current_states=None, num_steps=None):
        if self._network_kwargs["state_as_input"] and self.dyn is not None:
            inputs = torch.cat((inputs, current_states), dim=-1)
        if inputs.ndim == 2:
            # [B, D]
            
            preds = self.mlp(inputs)
            if EC_feat is not None:
                bs,M = EC_feat.shape[:2]

                # EC_feat = self.traj_encoder(cond_traj.reshape(-1,T,D)).reshape(bs,M,-1)
                inputs_tile = inputs.unsqueeze(1).tile(1,M,1)
                EC_feat = torch.cat((inputs_tile,EC_feat),dim=-1)
                EC_preds = TensorUtils.time_distributed(EC_feat, self.offsetmlp)
                EC_preds["trajectories"] = EC_preds["trajectories"] + preds["trajectories"].unsqueeze(1)
            else:
                EC_preds = None

        elif inputs.ndim == 3:
            # [B, A, D]
            preds = TensorUtils.time_distributed(inputs, self.mlp)
            if EC_feat is not None:
                assert EC_feat.ndim==4
                bs,A,M= EC_feat.shape[:3]
                # EC_feat = self.traj_encoder(cond_traj.reshape(-1,T,D)).reshape(bs,M,-1)
                inputs_tile = inputs.tile(1,M,1)
                EC_feat = torch.cat((inputs_tile,EC_feat),dim=-1)
                EC_preds = TensorUtils.time_distributed(EC_feat, self.offsetmlp)
                EC_preds = reshape_dimensions(EC_preds,1,2,(A,M))
                EC_preds["trajectories"] = EC_preds["trajectories"] + preds["trajectories"].unsqueeze(2)
            else:
                EC_preds = None
        else:
            raise ValueError(
                "Expecting inputs to have ndim == 2 or 3, got {}".format(inputs.ndim))
        return preds, EC_preds
    
    def _forward_dynamics(self, current_states, actions):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0

        _, pos, yaw = dynamics.forward_dynamics(
            self.dyn,
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj
    def forward(self, inputs, current_states=None, EC_feat=None, num_steps=None):
        preds, EC_preds = self._forward_networks(
            inputs, EC_feat, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            preds["controls"] = preds["trajectories"]
            if EC_preds is None:
                preds["trajectories"] = self._forward_dynamics(
                    current_states=current_states,
                    actions=preds["trajectories"]
                )
            else:
                total_actions = torch.cat((preds["trajectories"].unsqueeze(1),EC_preds["trajectories"]),1)
                bs,A,T,D = total_actions.shape
                current_states_tiled = current_states.unsqueeze(1).repeat(1,total_actions.size(1),1)
                total_trajectories = self._forward_dynamics(
                    current_states=current_states_tiled.reshape(bs*A,-1),
                    actions=total_actions.reshape(bs*A,T,D)
                ).reshape(*total_actions.shape[:-1],-1)
                preds["trajectories"] = total_trajectories[:,0]
                preds["EC_trajectories"] = total_trajectories[:,1:]
        else:
            preds["EC_trajectories"] = EC_preds["trajectories"]
                
        return preds


if __name__ == "__main__":
    model = RasterizedMapEncoder(model_arch="resnet18", input_image_shape=(15, 224, 224), output_channel=4)