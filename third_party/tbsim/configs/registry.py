#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.trajdata_nusc_scene_config import (
    NuscTrajdataSceneTrainConfig,
    NuscTrajdataSceneEnvConfig
)
from ccdiff.configs.algo_config import (
    CCDiffConfig,
)


EXP_CONFIG_REGISTRY = dict()

# --- scene-centric ---
EXP_CONFIG_REGISTRY["trajdata_nusc_ccdiff"] = ExperimentConfig(
    train_config=NuscTrajdataSceneTrainConfig(),
    env_config=NuscTrajdataSceneEnvConfig(),
    algo_config=CCDiffConfig(),
    registered_name="trajdata_nusc_ccdiff"
)


def get_registered_experiment_config(registered_name):
    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()

