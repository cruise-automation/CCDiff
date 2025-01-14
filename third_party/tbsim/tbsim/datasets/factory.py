#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

"""DataModule / Dataset factory"""
from tbsim.utils.config_utils import translate_trajdata_cfg, translate_pass_trajdata_cfg
from tbsim.datasets.trajdata_datamodules import UnifiedDataModule, PassUnifiedDataModule

def datamodule_factory(cls_name: str, config):
    """
    A factory for creating pl.DataModule.
    Args:
        cls_name (str): name of the datamodule class
        config (Config): an Experiment config object
        **kwargs: any other kwargs needed by the datamodule

    Returns:
        A DataModule
    """
    if cls_name.startswith("Unified"):
        trajdata_config = translate_trajdata_cfg(config)
        datamodule = UnifiedDataModule(data_config=trajdata_config, train_config=config.train)
    elif cls_name.startswith("PassUnified"):
        trajdata_config = translate_pass_trajdata_cfg(config)
        datamodule = PassUnifiedDataModule(data_config=trajdata_config, train_config=config.train)
    else:
        raise NotImplementedError("{} is not a supported datamodule type".format(cls_name))
    return datamodule
