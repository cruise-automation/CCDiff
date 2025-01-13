#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

#!/bin/bash

DATA_PATH="data/gt_results/scene_edit_eval/matrix" 

for file in "$DATA_PATH"/*.tar.gz; do tar -xzf "$file" -C "$DATA_PATH"; done

find "$DATA_PATH" -type f -name "*.npy" -exec mv {} "$DATA_PATH" \;

find "$DATA_PATH" -mindepth 1 -type d -empty -delete
