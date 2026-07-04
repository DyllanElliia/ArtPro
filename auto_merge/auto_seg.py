import os
import copy
import numpy as np
import open3d as o3d
import torch
import json

from .cal_trans_utils import (
    estimate_constrained_motion_torch,
    best_to_o3d_T,
    residual_nn_torch,
    export_vectors_to_ply,
)
from .part_seg_utils import adaptive_part_segmentation_torch
from .part_seg_utils import (
    visualize_parts,
    visualize_cd_parts,
    visualize_cluster_seeds,
    visualize_dists,
    visualize_pcd_parts,
)

# from .merge_utils import neighbor_motion_merge, filter_static_parts, filter_small_parts

from .tiny_main_utils import (
    estimate_principal_directions,
    get_bounding_box,
    ensure_normals,
)

from tqdm.auto import tqdm


def debug_visual(Pi, labels, parts, info, debug_name, obj_name, Pi_name):
  print(f"save debug visualizations to {debug_name}...")

  visualize_parts(f"{debug_name}/{obj_name}_{Pi_name}_parts.ply", Pi, labels)
  visualize_cd_parts(f"{debug_name}/{obj_name}_{Pi_name}_changed.ply", Pi,
                     info["seed_indices"])
  visualize_cluster_seeds(f"{debug_name}/{obj_name}_{Pi_name}_seeds.ply", Pi,
                          info["cluster_seeds"])
  visualize_dists(f"{debug_name}/{obj_name}_{Pi_name}_dists.png", info["dists"],
                  info["curr_tau"])
