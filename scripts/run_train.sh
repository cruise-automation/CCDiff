#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

#!/bin/bash

# debug
python ccdiff/examples/train.py --dataset_path /mnt/sdb/nuScenes \
    --config_name trajdata_nusc_ccdiff  --name train --debug


# train
python ccdiff/examples/train.py --dataset_path /mnt/sdb/nuScenes \
    --config_name trajdata_nusc_ccdiff  --name train --wandb_project ccdiff
