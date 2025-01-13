#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import tbsim.envs.env_metrics as EnvMetrics

from tbsim.utils.batch_utils import batch_utils
from tbsim.configs.base import ExperimentConfig

from tbsim.utils.experiment_utils import get_checkpoint

class MetricsComposer(object):
    """Wrapper for building learned metrics from trained checkpoints."""
    def __init__(self, eval_config, device, ckpt_root_dir="checkpoints/"):
        self.device = device
        self.ckpt_root_dir = ckpt_root_dir
        self.eval_config = eval_config
        self._exp_config = None

    def get_modality_shapes(self, exp_cfg: ExperimentConfig):
        return batch_utils().get_modality_shapes(exp_cfg)

    def get_metrics(self):
        raise NotImplementedError
