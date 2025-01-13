#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import numpy as np
from PIL import Image, ImageDraw

from l5kit.geometry import transform_points
from l5kit.rasterization.render_context import RenderContext
from l5kit.configs.config import load_metadata
from trajdata.maps.raster_map import RasterizedMap

from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.geometry_utils import get_box_world_coords_np
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.trajdata_utils import verify_map


COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
    "ego_contour": "#911A12",
    "ego_fill": "#FE5F55",
}


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def draw_actions(
        state_image,
        trans_mat,
        pred_action=None,
        pred_plan=None,
        pred_plan_info=None,
        ego_action_samples=None,
        plan_samples=None,
        action_marker_size=3,
        plan_marker_size=8,
):
    im = Image.fromarray((state_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    if pred_action is not None:
        raster_traj = agent_to_raster_np(
            pred_action["positions"].reshape(-1, 2), trans_mat)
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#FE5F55", outline="#911A12")
    if ego_action_samples is not None:
        raster_traj = agent_to_raster_np(
            ego_action_samples["positions"].reshape(-1, 2), trans_mat)
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#808080",
                         outline="#911A12")

    if pred_plan is not None:
        pos_raster = agent_to_raster_np(
            pred_plan["positions"][:, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    if plan_samples is not None:
        pos_raster = agent_to_raster_np(
            plan_samples["positions"][0, :, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    im = np.asarray(im)
    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap("jet")
        heatmap = pred_plan_info["location_map"][0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        heatmap = cm(heatmap)

        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(size=(im.shape[1], im.shape[0]))
        heatmap = np.asarray(heatmap)[..., :3]
        padding = np.ones((im.shape[0], 200, 3), dtype=np.uint8) * 255

        composite = heatmap.astype(np.float32) * \
            0.3 + im.astype(np.float32) * 0.7
        composite = composite.astype(np.uint8)
        im = np.concatenate((im, padding, heatmap, padding, composite), axis=1)

    return im


def draw_agent_boxes(image, pos, yaw, extent, raster_from_agent, outline_color, fill_color):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2)).astype(np.int32)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    for b in boxes_raster:
        im_draw.polygon(xy=b.reshape(-1).tolist(),
                        outline=outline_color, fill=fill_color)

    im = np.asarray(im).astype(np.float32) / 255.
    return im


def render_state_trajdata(
        batch: dict,
        batch_idx: int,
        action,
        rgb_idx_groups=None,
) -> np.ndarray:
    if rgb_idx_groups is None:
        # # backwards compat with nusc
        # rgb_idx_groups = [[0, 1, 2], [3, 4], [5, 6]]
        rgb_idx_groups = [[0], [1], [2]]
    pos = batch["history_positions"][batch_idx, -1]
    yaw = batch["history_yaws"][batch_idx, -1]
    extent = batch["extent"][batch_idx, :2]
    
    if batch["maps"] is None:
        # don't have a map, set to white background
        _, h, w = batch["image"][batch_idx].shape
        image = np.ones((h, w, 3))
    else:
        image = RasterizedMap.to_img(
            TensorUtils.to_tensor(verify_map(batch["maps"])[batch_idx]),
            rgb_idx_groups,
        )

    image = draw_agent_boxes(
        image,
        pos=pos[None, :],
        yaw=yaw[None, :],
        extent=extent[None, :],
        raster_from_agent=batch["raster_from_agent"][batch_idx],
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"]
    )

    scene_index = batch["scene_index"][batch_idx]
    agent_scene_index= scene_index == batch["scene_index"]
    agent_scene_index[batch_idx] = 0  # don't plot ego

    neigh_pos = batch["centroid"][agent_scene_index]
    neigh_yaw = batch["yaw"][agent_scene_index]
    neigh_extent = batch["extent"][agent_scene_index, :2]

    if neigh_pos.shape[0] > 0:
        image = draw_agent_boxes(
            image,
            pos=neigh_pos,
            yaw=neigh_yaw[:, None],
            extent=neigh_extent,
            raster_from_agent=batch["raster_from_world"][batch_idx],
            outline_color=COLORS["agent_contour"],
            fill_color=COLORS["agent_fill"]
        )

    plan_info = None
    plan_samples = None
    action_samples = None
    if "plan_info" in action.agents_info:
        plan_info = TensorUtils.map_ndarray(action.agents_info["plan_info"], lambda x: x[[batch_idx]])
    if "plan_samples" in action.agents_info:
        plan_samples = TensorUtils.map_ndarray(action.agents_info["plan_samples"], lambda x: x[[batch_idx]])
    if "action_samples" in action.agents_info:
        action_samples = TensorUtils.map_ndarray(action.agents_info["action_samples"], lambda x: x[[batch_idx]])

    vis_action = TensorUtils.map_ndarray(action.agents.to_dict(), lambda x: x[batch_idx])
    image = draw_actions(
        image,
        trans_mat=batch["raster_from_agent"][batch_idx],
        pred_action=vis_action,
        pred_plan_info=plan_info,
        ego_action_samples=action_samples,
        plan_samples=plan_samples,
        action_marker_size=2,
        plan_marker_size=3
    )
    return image
