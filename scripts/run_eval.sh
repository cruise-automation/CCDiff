#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

w_coll=-50.0
w_offroad=1.0

# topK controllable agents
controllable_agent=5
# horizon (number of frames in 10Hz)
step=5

python ccdiff/examples/scene_editor.py   --results_root_dir debug/ --num_scenes_per_batch 1  \
    --dataset_path /mnt/disks/sdb/nuscenes   --env trajdata  \
    --policy_ckpt_dir "ccdiff_trained_models/train/run0/" \
    --policy_ckpt_key iter10_ep0_valLoss16.87.ckpt --eval_class CCDiff  \
    --editing_source 'config' 'heuristic'  --registered_name 'trajdata_nusc_ccdiff' \
    --render --part_control --controllable_agent ${controllable_agent} \
    --w_collision ${w_coll} --w_offroad ${w_offroad} \
    --n_step_action ${step} --save_every_n_frames 5
