#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""A script for evaluating closed-loop simulation"""
from ccdiff.algos.algos import (
    CCDiffTrafficModel,
)
from tbsim.utils.batch_utils import batch_utils
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.policies.hardcoded import (
    ReplayPolicy,
    GTPolicy,
    GTPolicyOpenLoop,
    GTNaNPolicy,
)
from ccdiff.policies.hierarchical import (
    CCDiffHybridPolicyControl
)

from tbsim.configs.base import ExperimentConfig

from tbsim.policies.wrappers import (
    PolicyWrapper,
    RefineWrapper,
    AgentCentricToSceneCentricWrapper,
    SceneCentricToAgentCentricWrapper,
    NaiveAgentCentricToSceneCentricWrapper,
)
from tbsim.configs.config import Dict
from tbsim.utils.experiment_utils import get_checkpoint



class PolicyComposer(object):
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_policy(self):
        raise NotImplementedError


class ReplayAction(PolicyComposer):
    """A policy that replays stored actions."""
    def get_policy(self):
        print("Loading action log from {}".format(self.eval_config.experience_hdf5_path))
        import h5py
        h5 = h5py.File(self.eval_config.experience_hdf5_path, "r")
        if self.eval_config.env == "trajdata":
            # TBD: hack
            if 'nusc' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nusc_bc")
            else: 
                raise
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return ReplayPolicy(h5, self.device), exp_cfg


class GroundTruth(PolicyComposer):
    """A fake policy that replays dataset trajectories."""
    def get_policy(self):
        if self.eval_config.env == "trajdata":
            # TBD: hack
            if 'nusc' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nusc_bc")
            else: 
                raise
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return GTPolicy(device=self.device), exp_cfg

class GroundTruthOpenLoop(PolicyComposer):
    """A fake policy that replays dataset trajectories."""
    def get_policy(self):
        if self.eval_config.env == "trajdata":
            # TBD: hack
            if 'nusc' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nusc_bc")
            else: 
                raise
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return GTPolicyOpenLoop(device=self.device), exp_cfg
    
class GroundTruthNaN(PolicyComposer):
    """A fake policy that replays dataset trajectories."""
    def get_policy(self):
        if self.eval_config.env == "nusc":
            exp_cfg = get_registered_experiment_config("nusc_bc")
        elif self.eval_config.env == "l5kit":
            exp_cfg = get_registered_experiment_config("l5_bc")
        elif self.eval_config.env == "trajdata":
            # TBD: hack
            if 'nusc' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nusc_bc")
            elif 'l5' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_l5_bc")
            elif 'nuplan' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nuplan_bc")
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return GTNaNPolicy(device=self.device), exp_cfg

class CCDiffHierarchicalPolicy(PolicyComposer):
    """A fake policy that replays dataset trajectories."""
    def get_policy(self):
        if self.eval_config.env == "trajdata":
            if 'nusc' in self.eval_config.registered_name:
                exp_cfg = get_registered_experiment_config("trajdata_nusc_ccdiff")
            else: 
                raise
        else:
            raise NotImplementedError("invalid env {}".format(self.eval_config.env))
        return CCDiffHybridPolicyControl(device=self.device), exp_cfg

class CCDiff(PolicyComposer):
    """CCDiff"""
    def get_policy(self, policy=None):
        if policy is not None:
            assert isinstance(policy, CCDiffTrafficModel)
            policy_cfg = None
        else:
            policy_ckpt_path, policy_config_path = get_checkpoint(
                ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
                ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
                ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
                ckpt_root_dir=self.ckpt_root_dir,
            )
            policy_cfg = get_experiment_config_from_file(policy_config_path)

            policy = CCDiffTrafficModel.load_from_checkpoint(
                policy_ckpt_path,
                algo_config=policy_cfg.algo,
                modality_shapes=self.get_modality_shapes(policy_cfg),
                registered_name=policy_cfg["registered_name"],
            ).to(self.device).eval()
            policy_cfg = policy_cfg.clone()
        policy = PolicyWrapper.wrap_controller(
            policy,
            num_action_samples=self.eval_config.policy.num_action_samples,
            class_free_guide_w=self.eval_config.policy.class_free_guide_w,
            guide_as_filter_only=self.eval_config.policy.guide_as_filter_only,
            guide_with_gt=self.eval_config.policy.guide_with_gt,
            guide_clean=self.eval_config.policy.guide_clean,
        )
        if policy_cfg is None or policy_cfg.algo.coordinate == 'agent_centric':
            policy = NaiveAgentCentricToSceneCentricWrapper(policy)
        elif policy_cfg.algo.coordinate == 'scene_centric':
            policy = AgentCentricToSceneCentricWrapper(policy)
        else:
            raise NotImplementedError
        return policy, policy_cfg
