#
# Created by lxl.
#

import os
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
import json

os.environ['USE_KEOPS'] = '1'
from geomloss import SamplesLoss

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import BWScenes
from utils.general_utils import quat_mult, mat2quat, mat2quat_batch
from utils.loss_utils import eval_img_loss, eval_knn_opacities_collision_loss, eval_depth_loss

from scene.multipart_models_base import MPArtModelBasic, COLORS


class MPArtModelJoint(MPArtModelBasic):

  def setup_args_extra(self):
    self.opt.densify_grad_threshold = 0.0002
    self.opt.min_opacity = 0.005

    self.opt.iterations = 6_000
    # self.opt.iterations = 9
    self.opt.densification_interval = 50
    self.opt.opacity_reset_interval = 1500
    self.opt.densify_from_iter = 50
    self.opt.densify_until_iter = 5_000

    self.opt.position_lr = 0.00016

    self.opt.collision_knn = 32
    self.opt.collision_weight = 0.02
    self.opt.collision_from_iter = 1
    self.opt.collision_until_iter = self.opt.densify_until_iter
    self.opt.collision_after_reset_iter = 500

    self.opt.depth_weight = 2.0

    # self.opt.column_lr = 0.00001
    # self.opt.t_lr = 0.00001
    self.opt.column_lr = 0.0
    self.opt.t_lr = 0.0

  def __init__(self, gaussians: GaussianModel, num_movable: int):
    self.canonical_gaussians = copy.deepcopy(gaussians)
    super().__init__(gaussians, num_movable)
    self.dataset_st = GroupParams()
    self.dataset_ed = GroupParams()
    self.mask = None
    self.part_indices = None
    self.setup_args_extra()

  @override
  def set_dataset(self, source_path: str, model_path: str, evaluate=True):
    self.dataset_st.sh_degree = 0
    self.dataset_ed.sh_degree = 0
    self.dataset_st.images = "images"
    self.dataset_ed.images = "images"
    self.dataset_st.resolution = -1
    self.dataset_ed.resolution = -1
    self.dataset_st.white_background = False
    self.dataset_ed.white_background = False
    self.dataset_st.data_device = "cuda"
    self.dataset_ed.data_device = "cuda"
    self.dataset_st.eval = False
    self.dataset_ed.eval = False

    self.dataset_st.source_path = os.path.join(os.path.realpath(source_path),
                                               'start')
    self.dataset_ed.source_path = os.path.join(os.path.realpath(source_path),
                                               'end')
    self.dataset_st.model_path = model_path
    self.dataset_ed.model_path = model_path

    mask_pre = np.load(os.path.join(model_path, 'mask_pre.npy'))
    part_indices_pre = np.load(os.path.join(model_path, 'part_indices_pre.npy'))
    r_pre = np.load(os.path.join(model_path, 'r_pre.npy'))
    t_pre = np.load(os.path.join(model_path, 't_pre.npy'))
    self.mask = torch.tensor(mask_pre, device='cuda', dtype=torch.bool)
    self.part_indices = torch.tensor(part_indices_pre,
                                     device='cuda',
                                     dtype=torch.long)
    self.set_init_params(t_pre, r_pre)

  @override
  def deform(self, iteration):
    r, t = self.get_r_t()  # r: (K, 3, 3), t: (K, 3)
    canonical_xyz = self.canonical_gaussians.get_xyz
    canonical_rotation = self.canonical_gaussians.get_rotation

    # 直接 clone，避免 zeros_like + 赋值
    self.gaussians.get_xyz = canonical_xyz.clone()
    self.gaussians.get_rotation_raw = canonical_rotation.clone()

    # 批量计算所有 part 的逆旋转四元数: (K, 3, 3) -> (K, 4)
    r_inv = r.transpose(-1, -2)  # (K, 3, 3)
    r_inv_quats = mat2quat_batch(r_inv)  # (K, 4)

    # 向量化处理所有 movable parts
    # part_indices 对应每个点属于哪个 part，mask 标记哪些点需要变换
    movable_mask = self.mask  # (N,)
    part_idx = self.part_indices[movable_mask]  # (M,) 其中 M 是 movable 点数

    # 获取 movable 点的坐标和旋转
    xyz_movable = canonical_xyz[movable_mask]  # (M, 3)
    rot_movable = canonical_rotation[movable_mask]  # (M, 4)

    # 根据 part_idx 索引对应的 r 和 t: (M, 3, 3), (M, 3)
    r_per_point = r[part_idx]  # (M, 3, 3)
    t_per_point = t[part_idx]  # (M, 3)
    r_inv_quat_per_point = r_inv_quats[part_idx]  # (M, 4)

    # 批量矩阵乘法: (M, 1, 3) @ (M, 3, 3) -> (M, 1, 3) -> (M, 3)
    xyz_transformed = torch.bmm(xyz_movable.unsqueeze(1),
                                r_per_point).squeeze(1) + t_per_point

    # 批量四元数乘法
    rot_transformed = quat_mult(r_inv_quat_per_point, rot_movable)

    # 写回结果
    self.gaussians.get_xyz[movable_mask] = xyz_transformed
    self.gaussians.get_rotation_raw[movable_mask] = rot_transformed

    self.gaussians.get_scaling_raw = self.canonical_gaussians.get_scaling_raw
    self.gaussians.get_features_dc = self.canonical_gaussians.get_features_dc
    self.gaussians.get_features_rest = self.canonical_gaussians.get_features_rest
    self.gaussians.get_opacity_raw = self.canonical_gaussians.get_opacity_raw
    return self.gaussians

  def _show_losses(self, iteration: int, losses: dict):
    if iteration in [2, 1000, 3000, 5000, 7000, 9000]:
      self.canonical_gaussians.save_ply(os.path.join(
          self.dataset_ed.model_path,
          f'point_cloud/iteration_{iteration - 1}/point_cloud.ply'),
                                        prune=False)
      self.gaussians.save_ply(os.path.join(
          self.dataset_ed.model_path,
          f'point_cloud/iteration_{iteration - 2}/point_cloud.ply'),
                              prune=False)
      self.canonical_gaussians[self.mask].save_ply(
          os.path.join(self.dataset_ed.model_path,
                       f'point_cloud/iteration_{-iteration}/point_cloud.ply'),)

  @override
  def train(self, gt_gaussians=None, bws_st=None, bws_ed=None):
    iterations = self.opt.iterations
    if bws_st is None:
      bws_st = BWScenes(self.dataset_st, self.gaussians, is_new_gaussians=False)
    if bws_ed is None:
      bws_ed = BWScenes(self.dataset_ed, self.gaussians, is_new_gaussians=False)
    self.training_setup(self.opt)

    for k in range(self.num_movable):
      # freeze all motion
      self._column_vec1[k].requires_grad_(False)
      self._column_vec2[k].requires_grad_(False)
      self._t[k].requires_grad_(False)
      self._c[k].requires_grad_(False)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(iterations), desc="Training progress")
    prev_opacity_reset_iter = -114514
    for i in range(1, iterations + 1):
      # Pick a random Camera from st and ed respectively
      viewpoint_cam_st, background_st = bws_st.pop_black() if (
          i % 2 == 0) else bws_st.pop_white()
      viewpoint_cam_ed, background_ed = bws_ed.pop_black() if (
          i % 2 == 0) else bws_ed.pop_white()

      self.deform(i)

      losses = {
          'app_st': None,
          'app_ed': None,
          'depth_st': None,
          'depth_ed': None,
          'collision': None
      }
      requires_collision = (i - prev_opacity_reset_iter
                            >= self.opt.collision_after_reset_iter)
      requires_collision &= (self.opt.collision_from_iter <= i <=
                             self.opt.collision_until_iter)

      gt_image = viewpoint_cam_st.original_image.cuda().float()
      render_pkg = render(viewpoint_cam_st, self.canonical_gaussians, self.pipe,
                          background_st)
      image, viewspace_point_tensor, visibility_filter, radii \
          = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
      losses['app_st'] = eval_img_loss(image, gt_image, self.opt)

      if (self.opt.depth_weight is not None) and (viewpoint_cam_st.image_depth
                                                  is not None):
        gt_depth = viewpoint_cam_st.image_depth.cuda().float()
        depth = render_pkg['depth']
        losses['depth_st'] = eval_depth_loss(depth, gt_depth)

      gt_image = viewpoint_cam_ed.original_image.cuda().float()
      render_pkg = render(viewpoint_cam_ed, self.gaussians, self.pipe,
                          background_ed)
      image, viewspace_point_tensor, visibility_filter, radii \
          = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
      losses['app_ed'] = eval_img_loss(image, gt_image, self.opt)

      if (self.opt.depth_weight is not None) and (viewpoint_cam_ed.image_depth
                                                  is not None):
        gt_depth = viewpoint_cam_ed.image_depth.cuda().float()
        depth = render_pkg['depth']
        losses['depth_ed'] = eval_depth_loss(depth, gt_depth)

      weight_st = losses['app_st'].detach() / (losses['app_st'].detach() +
                                               losses['app_ed'].detach())
      loss = weight_st * losses['app_st'] + (1 - weight_st) * losses['app_ed']

      if (self.opt.collision_weight is not None) and requires_collision:
        losses['collision'] = eval_knn_opacities_collision_loss(
            self.gaussians, self.mask, k=self.opt.collision_knn)
        loss += self.opt.collision_weight * losses['collision'] / 1

      if (losses['depth_st'] is not None) and (losses['depth_ed'] is not None):
        weight_st = losses['depth_st'].detach() / (losses['depth_st'].detach() +
                                                   losses['depth_ed'].detach())
        loss += self.opt.depth_weight * (weight_st * losses['depth_st'] +
                                         (1 - weight_st) * losses['depth_ed'])

      loss.backward()
      with torch.no_grad():
        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if i % 10 == 0:
          progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
          progress_bar.update(10)

        # Densification
        if i < self.opt.densify_until_iter:
          # Keep track of max radii in image-space for pruning
          self.canonical_gaussians.max_radii2D[visibility_filter] = torch.max(
              radii[visibility_filter],
              self.canonical_gaussians.max_radii2D[visibility_filter])
          self.canonical_gaussians.add_densification_stats(
              viewspace_point_tensor, visibility_filter)
          # copy or split
          if i > self.opt.densify_from_iter and i % self.opt.densification_interval == 0:
            size_threshold = 20 if i > self.opt.opacity_reset_interval else None
            self.mask, self.part_indices = self.canonical_gaussians.densify_and_prune(
                self.opt.densify_grad_threshold,
                self.opt.min_opacity,
                bws_st.get_cameras_extent(),
                size_threshold,
                auxiliary_attr=(self.mask, self.part_indices))
          # opacity reset
          if i % self.opt.opacity_reset_interval == 0 or (
              self.dataset_st.white_background and
              i == self.opt.densify_from_iter):
            self.canonical_gaussians.reset_opacity()
            prev_opacity_reset_iter = i

        if i < iterations:
          self.optimizer.step()
          self.canonical_gaussians.optimizer.step()
          self.optimizer.zero_grad(set_to_none=True)
          self.canonical_gaussians.optimizer.zero_grad(set_to_none=False)
      self._show_losses(i, losses)
    progress_bar.close()
    return self.get_t, self.get_r
