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
from scene.dataset_readers import fetchPly, storePly
from utils.general_utils import quat_mult, mat2quat, mat2quat_batch, inverse_sigmoid, inverse_softmax, \
    strip_symmetric, build_scaling_rotation, eval_quad, decompose_covariance_matrix, build_rotation, \
    find_close, find_files_with_suffix, kl_divergence_gaussian, value_to_rgb, shift_aabb_from_collision, \
    get_extended_aabb, get_bb_collision_axis, get_bb_collision_axis_torch,get_expon_lr_func, \
    rotation_axis_from_matrix, rotation_axis_from_matrix_batch, axis_angle_to_matrix_torch, \
    approximate_obb_intersection_volume
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses, eval_cd_loss_sd, \
    eval_knn_opacities_collision_loss, eval_opacity_bce_loss, eval_depth_loss, sample_pts
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH

from utils.dual_quaternion import quaternion_mul, matrix_to_quaternion, dual_quaternion_apply

from main_utils import prepare_output_and_logger

from plyfile import PlyData, PlyElement

from scene.multipart_models_base import MPArtModelBasic, COLORS


class GMMArtModel(MPArtModelBasic):

  def setup_args_extra(self):

    self.opt.iterations = 6_000
    self.opt.warmup_until_iter = 500

    self.opt.cd_from_iter = 50
    self.opt.cd_until_iter = 3000
    self.opt.cd_from_weight = 0
    self.opt.cd_until_weight = 0.5

    self.opt.prior_until_iter = 2000

    self.opt.sgd_interval = 1

    self.opt.column_lr = 0.0024
    self.opt.column_lr_init = self.opt.column_lr
    self.opt.column_lr_final = 0.00001

    self.opt.position_lr_delay_mult = 0.01
    self.opt.position_lr_max_steps = 30_000
    self.opt.deform_lr_max_steps = 40_000

    self.opt.c_lr = 0.000075
    self.opt.c_lr_init = 0.000075
    self.opt.c_lr_final = 0.00002
    self.opt.t_lr_init = 0.00005
    self.opt.t_lr_final = 0.00003

    self.opt.prob_lr = 0.05
    self.opt.prob_lr_init = 0.05
    self.opt.prob_lr_final = 0.0001

    self.opt.position_lr = 0.000016
    self.opt.scaling_lr = 0.005
    self.opt.opacity_lr = 0.05

    self.opt.depth_scaling = 1

    self.opt.depth_weight = None
    self.opt.cd_weight = None
    self.opt.scaling_weight = None
    self.opt.center_weight = None
    self.opt.ppp_weight = None
    self.opt.cr_weight = None
    self.opt.ppp_weight_ed = None
    self.opt.kl_weight = None
    self.opt.sd_weight = None
    self.opt.mpp_weight = None
    self.opt.motion_limit_weight = 1.0
    self.opt.motion_limit_until_iter = 500

    self.opt.depth_weight = 2.0
    self.opt.cd_weight = 1.0
    self.opt.cd_weight_const = 3.0
    self.opt.center_weight = 0.2
    self.opt.scaling_weight = 0.1
    self.opt.ppp_weight = 0.1
    self.opt.spar_weight = 0.02

    self.opt.ppp_pretrain_iters = 500
    self.opt.merge_iter = 1000

    self.opt.rotation_to_translation_interval = 100
    self.opt.rotation_to_translation_max_angle_deg = 8.0
    self.opt.rotation_to_translation_min_shift = 5e-3
    self.opt.rotation_to_translation_min_shift_extent_ratio = 0.25
    self.opt.rotation_to_translation_max_residual_ratio = 0.1

    self.opt.mask_thresh = .85

  def __init__(self,
               gaussians: GaussianModel,
               num_movable: int,
               cluster_folder: str = "clustering",
               new_scheme=True):
    super().__init__(gaussians, num_movable)
    self._prob = nn.Parameter(
        torch.zeros(gaussians.size(), dtype=torch.float,
                    device='cuda').requires_grad_(True))
    # GMM parameters below
    self._xyz = nn.Parameter(
        torch.zeros(self.num_movable, 3, dtype=torch.float,
                    device='cuda').requires_grad_(True))
    self._scaling = nn.Parameter(
        torch.zeros(self.num_movable, 3, dtype=torch.float,
                    device='cuda').requires_grad_(True))
    self._rotation_col1 = nn.Parameter(
        torch.tensor([1, 0, 0], dtype=torch.float,
                     device='cuda').repeat(num_movable, 1).requires_grad_(True))
    self._rotation_col2 = nn.Parameter(
        torch.tensor([0, 1, 0], dtype=torch.float,
                     device='cuda').repeat(num_movable, 1).requires_grad_(True))
    self._opacity = nn.Parameter(
        torch.zeros(num_movable, dtype=torch.float,
                    device='cuda').requires_grad_(True))
    self._p = 1.0
    self.ppp = None
    self.center_ref = None
    self.scaling_ref = None
    self.motion_axis_ref = None
    self.motion_type_ref = None
    self.motion_axis_mask = None

    self.collisions = set()

    self.opt.obb_collision_delta = 0.1
    self.opt.obb_volume_resolution = 24
    self.opt.collision_freeze_rotation = False

    self.bb_axes = None
    self.bb_centers = None
    self.bb_extents = None
    self.bb_axes_deformed = None
    self.bb_centers_deformed = None
    self.neighbors_mat = None

    self.static_parts = set()
    self.mean_mpp_max = [0.0 for _ in range(num_movable)]
    self.part_num_pts = [[] for _ in range(num_movable)]
    self.merged_parts = []

    def build_inverse_covariance_from_scaling_rotation(scaling, rot_col1,
                                                       rot_col2):
      ss = torch.diag_embed(1 / (scaling + 1e-8))
      rr = torch.ones(self.num_movable,
                      3,
                      3,
                      dtype=rot_col1.dtype,
                      device=rot_col1.device)
      for i in range(self.num_movable):
        rr[i] = self.gram_schmidt(rot_col1[i], rot_col2[i])
      return rr @ ss @ ss @ rr.transpose(1, 2)

    self.prob_activation = torch.sigmoid
    self.scaling_activation = torch.exp
    self.scaling_inverse_activation = torch.log
    self.inverse_covariance_activation = build_inverse_covariance_from_scaling_rotation
    self.opacity_activation = torch.sigmoid

    self.original_xyz = self.gaussians.get_xyz.clone().detach()
    self.original_rotation = self.gaussians.get_rotation.clone().detach()
    self.original_opacity = self.gaussians.get_opacity.clone().detach()
    self.original_gaussians = copy.deepcopy(self.gaussians)
    self.is_revolute = np.array([True for _ in range(self.num_movable)])

    self.pcd_gt = None  # end state pcd from depth map
    self.pcds = []  # start state clustered pcds
    self.pcds_sample = []
    self.pcds_deformed = []  # legacy, kept for compatibility
    self.pcd_knn_indices = []

    # Vectorized pcd storage
    self.pcds_all = None  # (N_total, 3) all pcds concatenated
    self.pcds_deformed_all = None  # (N_total, 3) all deformed pcds
    self.pcd_part_labels = None  # (N_total,) part index for each point
    self.pcd_part_ranges = []  # [(start, end), ...] index ranges for each part

    self.ed_knn_indices = []

    self._pcd_pair_min_dist0 = None
    self._obb_pair_volume0 = None

    self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    print(f"use {'new' if new_scheme else 'old'} deform scheme")

    self.new_scheme = new_scheme
    self.gaussians.duplicate(2 if new_scheme else self.num_movable + 1)

    # Precomputed mask cache for vectorized deform (updated every N iterations)
    self._mask_cache_iter = -1  # last iteration when mask was computed
    self._mask_cache_interval = 10  # compute mask every N iterations
    self._cached_counts = None  # (K,) active count per part
    self._cached_max_active = 0  # max active count across all parts
    self._cached_active_part_indices = None  # (M,) part id per active pair
    self._cached_active_src_indices = None  # (M,) source point index per pair
    self._cached_active_dst_indices = None  # (M,) duplicated slot index per pair

    self.cluster_folder = cluster_folder
    print("Using cluster folder:", cluster_folder)
    self.setup_args_extra()

  @property
  def get_prob(self):
    return self.prob_activation(self._prob)

  @property
  def get_mu(self):
    return self._xyz

  @property
  def get_rotation(self):
    return self.gram_schmidt_batch(self._rotation_col1, self._rotation_col2)

  @property
  def get_scaling(self):
    return self.scaling_activation(self._scaling)

  @property
  def get_opacity(self):
    activated = self.opacity_activation(self._opacity)
    if self.merged_parts:
      mask = torch.ones_like(activated)
      mask[self.merged_parts] = 0.0
      return activated * mask
    return activated

  @property
  def get_inverse_covariance(self):
    return self.inverse_covariance_activation(self.get_scaling,
                                              self._rotation_col1,
                                              self._rotation_col2)

  def cosine_anneal(self,
                    step,
                    final_step=-1,
                    start_step=0,
                    start_value=1.0,
                    final_value=0.1):
    if final_step == -1:
      final_step = self.opt.iterations

    if step < start_step:
      value = start_value
    elif step >= final_step:
      value = final_value
    else:
      a = 0.5 * (start_value - final_value)
      b = 0.5 * (start_value + final_value)
      progress = (step - start_step) / (final_step - start_step)
      value = a * math.cos(math.pi * progress) + b
    return value

  def _cal_relative_pos(self, x, mu=None, rot=None, scale=None):
    mu = self.get_mu if mu is None else mu  # x [N, 3], mu [K, 3]
    rot = self.get_rotation if rot is None else rot  # rot [K, 3, 3]
    scale = self.get_scaling if scale is None else scale  # scale [K, 3]
    # [N, K, 3]
    return torch.einsum('kji,nkj->nki', rot, x.unsqueeze(1) - mu) / scale

  def get_ppp(self, pts=None, deformed=False, tau=1.0, eps=1e-8):
    if (self.ppp is not None) and (pts is None):
      return self.ppp
    save = (pts is None)
    if deformed:
      assert pts is not None
      r, t = self.get_r_t()
      r = r.to(dtype=self.get_mu.dtype)
      t = t.to(dtype=self.get_mu.dtype)
      mu = torch.einsum('kji,kj->ki', r, self.get_mu) + t
      rot = r.transpose(1, 2) @ self.get_rotation
      rel_pos = self._cal_relative_pos(pts, mu, rot, self.get_scaling)
    else:
      if pts is None:
        pts = self.original_xyz
      rel_pos = self._cal_relative_pos(pts)
    quad = torch.sum(rel_pos**2, dim=-1)**self._p  # [N, K]
    ppp = self.get_opacity * torch.exp(-quad / tau)
    ppp = ppp.clamp(eps, 1 - eps)
    ppp /= ppp.sum(dim=1, keepdim=True)

    self.ppp = ppp if save else self.ppp
    return ppp

  def pred_mp(self):
    return torch.argmax(self.get_ppp(), dim=1)

  @override
  def set_dataset(self,
                  source_path: str,
                  model_path: str,
                  evaluate=True,
                  thr=-5):
    super().set_dataset(source_path, model_path, evaluate)
    try:
      xyz_ed = np.asarray(fetchPly(
          os.path.join(source_path, 'points3d-100k.ply')).points,
                          dtype=np.float32)
    except:
      print(
          "[Warning] points3d-100k.ply not found, using points3d.ply instead.")
      pcd_ed = fetchPly(os.path.join(source_path, 'points3d.ply'))
      xyz_ed = np.asarray(pcd_ed.points, dtype=np.float32)
      rgb_ed = pcd_ed.colors
      random_indices = np.random.choice(len(xyz_ed),
                                        size=100_000,
                                        replace=False)
      xyz_ed = xyz_ed[random_indices]
      rgb_ed = rgb_ed[random_indices]
      storePly(
          os.path.join(model_path, self.cluster_folder, f'points3d-100k.ply'),
          xyz_ed, rgb_ed)
    xyz_st = np.asarray(fetchPly(
        os.path.join(source_path, '../start/points3d.ply')).points,
                        dtype=np.float32)
    y = torch.tensor(xyz_st, device='cuda', dtype=torch.float32).unsqueeze(0)
    y = sample_pts(y, 10_100_000)
    x = torch.tensor(xyz_ed, device='cuda', dtype=torch.float32).unsqueeze(0)
    cd = chamfer_distance(x,
                          y,
                          batch_reduction=None,
                          point_reduction=None,
                          single_directional=True)[0][0]
    cd /= torch.max(cd)
    mask = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6)) > thr

    # edm = xyz_ed[mask.detach().cpu().numpy()]
    edm = xyz_ed
    storePly(os.path.join(model_path, self.cluster_folder, f'points3d-edm.ply'),
             edm, np.zeros_like(edm))
    self.pcd_gt = torch.tensor(edm, device='cuda', dtype=torch.float32)

    cluster_dir = os.path.join(model_path, self.cluster_folder, 'clusters')
    print("[init] Loading clustered PCDs from:", cluster_dir)
    for i in np.arange(100):
      ply_file = os.path.join(cluster_dir, f'points3d_{i}.ply')
      print(f'  Loading part {i} from {ply_file}')
      if os.path.exists(ply_file):
        pcd = torch.tensor(np.asarray(fetchPly(ply_file).points),
                           device='cuda',
                           dtype=torch.float)
        pcd = sample_pts(pcd, len(pcd) * 100_000 // len(xyz_st))
        self.pcds.append(pcd)
        self.pcds_deformed.append(pcd)  # legacy
        self.pcds_sample.append(sample_pts(pcd, len(pcd) // 10))
      if len(self.pcds) == self.num_movable:
        break
    assert len(self.pcds) == self.num_movable

    # Build vectorized pcd storage
    self.pcds_all = torch.cat(self.pcds, dim=0)  # (N_total, 3)
    self.pcds_deformed_all = self.pcds_all.clone()  # (N_total, 3)
    start = 0
    labels_list = []
    for k, pcd in enumerate(self.pcds):
      end = start + len(pcd)
      self.pcd_part_ranges.append((start, end))
      labels_list.append(
          torch.full((len(pcd),), k, dtype=torch.long, device='cuda'))
      start = end
    self.pcd_part_labels = torch.cat(labels_list, dim=0)  # (N_total,)
    print(
        f"[init] Vectorized PCDs: {self.pcds_all.shape[0]} total points, {self.num_movable} parts"
    )

    for k in range(self.num_movable):
      p1 = self.pcds_sample[k].unsqueeze(0)
      p2 = self.original_xyz.detach().unsqueeze(0)
      _, indices, _ = knn_points(p1, p2, K=1)
      self.pcd_knn_indices.append(indices.flatten())

  def _set_bbs(self, out_path: str):
    dirs_path = os.path.join(out_path, self.cluster_folder)
    axes = np.load(os.path.join(dirs_path, 'axes.npy'))
    bb_centers = np.load(os.path.join(dirs_path, 'bb_centers.npy'))
    bb_extents = np.load(os.path.join(dirs_path, 'bb_extents.npy'))
    self.neighbors_mat = np.load(os.path.join(dirs_path, 'neighbors.npy'))

    self.bb_axes = []
    self.bb_centers = []
    self.bb_extents = []
    for k in np.arange(self.num_movable):
      self.bb_centers.append(
          torch.tensor(bb_centers[k], dtype=torch.float, device='cuda'))
      self.bb_extents.append(
          torch.tensor(bb_extents[k], dtype=torch.float, device='cuda'))
      axes[k] /= np.linalg.norm(axes[k], axis=1, keepdims=True)
      self.bb_axes.append(
          torch.tensor(axes[k], dtype=torch.float, device='cuda'))

    self.bb_axes_deformed = [t.clone().detach() for t in self.bb_axes]
    self.bb_centers_deformed = [t.clone().detach() for t in self.bb_centers]
    self._initialize_pairwise_obb_volume_baseline()

  def _set_init_probabilities(self,
                              prob=None,
                              mu=None,
                              sigma=None,
                              scaling_modifier=1.0,
                              eps=1e-6):
    if prob is not None:
      prob_raw = inverse_sigmoid(torch.clamp(prob, eps, 1 - eps))
      self._prob = prob_raw.clone().detach().to('cuda').requires_grad_(True)
    if mu is not None:
      self._xyz = mu.clone().detach().to('cuda').requires_grad_(True)
    if sigma is not None:
      scaling, rotation = decompose_covariance_matrix(sigma)
      scaling_raw = self.scaling_inverse_activation(scaling_modifier * scaling)
      scaling_raw = torch.clamp(scaling_raw, -16, 16)
      self._scaling = scaling_raw.clone().detach().to('cuda').requires_grad_(
          True)
      self._rotation_col1 = rotation[:, :, 0].clone().detach().to(
          'cuda').requires_grad_(True)
      self._rotation_col2 = rotation[:, :, 1].clone().detach().to(
          'cuda').requires_grad_(True)

  def set_init_part_params(
      self,
      model_path: str,
      scaling_modifier=1.0,
      use_priors=True,  # for ab, set to False
      use_cues_removed=False):  # cue_removed
    prob = torch.tensor(np.load(os.path.join(model_path, 'mpp_init.npy')),
                        device='cuda',
                        dtype=torch.float32)
    mu = torch.tensor(np.load(os.path.join(model_path, 'mu_init.npy')),
                      device='cuda',
                      dtype=torch.float32)
    sigma = torch.tensor(np.load(os.path.join(model_path, 'sigma_init.npy')),
                         device='cuda',
                         dtype=torch.float32)
    c_init = mu

    c_init_path = os.path.join(model_path, 'c_init.npy')
    if os.path.exists(c_init_path):
      print("Using c_init from file:", c_init_path)
      c_init = torch.tensor(np.load(c_init_path),
                            device='cuda',
                            dtype=torch.float32)
    self._set_init_probabilities(prob, mu, sigma, scaling_modifier)
    self._c = [
        nn.Parameter(
            torch.tensor(c, dtype=torch.float,
                         device='cuda').requires_grad_(True)) for c in c_init
    ]
    R_init_path = os.path.join(model_path, 'R_init.npy')
    t_init_path = os.path.join(model_path, 't_init.npy')
    if os.path.exists(R_init_path) and os.path.exists(t_init_path):
      print("Using R_init and t_init from files:", R_init_path, t_init_path)
      R_init = np.load(R_init_path)
      t_init = np.load(t_init_path)
      self.set_init_params(t=[
          torch.tensor(t, dtype=torch.float, device='cuda').requires_grad_(True)
          for t in t_init
      ],
                           r=[
                               torch.tensor(r, dtype=torch.float,
                                            device='cuda').requires_grad_(True)
                               for r in R_init
                           ])
    self.center_ref = mu.clone().detach()

    if use_priors:
      self._set_bbs(model_path)
    self._capture_motion_reference()
    self._capture_initial_motion_params()

  def _capture_motion_reference(self):
    raw_t = torch.stack(self._t).detach()
    r, _ = self.get_r_t()
    r = r.detach()

    trace = r.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    motion_is_revolute = trace < self.opt.trace_r_thresh

    trans_norm = torch.linalg.norm(raw_t, dim=1)
    trans_axis = F.normalize(raw_t, dim=1, eps=1e-8)
    rot_axis_raw = torch.stack([
        r[:, 2, 1] - r[:, 1, 2],
        r[:, 0, 2] - r[:, 2, 0],
        r[:, 1, 0] - r[:, 0, 1],
    ],
                               dim=-1)
    rot_axis_norm = torch.linalg.norm(rot_axis_raw, dim=1)
    rot_axis = rotation_axis_from_matrix_batch(r)

    self.motion_type_ref = motion_is_revolute.detach()
    self.motion_axis_ref = torch.where(motion_is_revolute.unsqueeze(-1),
                                       rot_axis, trans_axis).detach()
    self.motion_axis_mask = torch.where(motion_is_revolute, rot_axis_norm
                                        > 1e-6, trans_norm > 1e-8).detach()

  def _eval_motion_trend_loss(self):
    if (self.motion_axis_ref is None) or (self.motion_type_ref
                                          is None) or (self.motion_axis_mask
                                                       is None):
      return None

    raw_t = torch.stack(self._t)
    r, _ = self.get_r_t()
    eye = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)

    valid_mask = self.motion_axis_mask
    current_trans_axis = F.normalize(raw_t, dim=1, eps=1e-8)
    current_rot_axis = rotation_axis_from_matrix_batch(r)
    current_axis = torch.where(self.motion_type_ref.unsqueeze(-1),
                               current_rot_axis, current_trans_axis)
    ref_axis = F.normalize(self.motion_axis_ref.detach(), dim=1, eps=1e-8)
    axis_loss = 0.0
    if bool(valid_mask.any()):
      dots = torch.clamp(torch.abs((current_axis * ref_axis).sum(dim=1)),
                         max=1.0)
      axis_loss = (1.0 - dots[valid_mask]).mean()

    class_loss = torch.where(self.motion_type_ref,
                             raw_t.pow(2).sum(dim=1),
                             (r - eye).pow(2).mean(dim=(1, 2))).mean()
    # print(
    #     f"motion trend loss: axis {axis_loss.item():.6f}, class {class_loss.item():.6f}"
    # )
    return 10 * axis_loss + class_loss

  def _get_slot_deform(self):
    r, t = self.get_r_t()  # r: (K, 3, 3), t: (K, 3)
    r_transposed = r.transpose(1, 2)  # (K, 3, 3)
    qrs = matrix_to_quaternion(r_transposed)  # (K, 4)
    t0 = torch.cat([torch.zeros(self.num_movable, 1, device=t.device), t],
                   dim=-1)  # (K, 4)
    qds = 0.5 * quaternion_mul(t0, qrs)  # (K, 4)
    return qrs, qds

  def _dual_quat_deform(self, iteration=-1):
    ppp = self.get_ppp(tau=self.cosine_anneal(iteration))
    qr, qd = self._get_slot_deform()
    qr = torch.einsum('nk,kl->nl', ppp, qr.to(dtype=ppp.dtype))  # [N, 4]
    qd = torch.einsum('nk,kl->nl', ppp, qd.to(dtype=ppp.dtype))  # [N, 4]
    xyz = dual_quaternion_apply((qr, qd), self.original_xyz)
    rot = quaternion_mul(qr, self.original_rotation)
    return xyz, rot

  def joint_pred(self):

    def is_likely_prismatic(diff: np.ndarray, thresh=0.02) -> bool:
      if len(diff) < 10:
        return False
      std = np.linalg.norm(np.std(diff, axis=0, ddof=1))
      mean = np.linalg.norm(diff, axis=1).mean()
      is_prismatic = (std / (mean + 1e-5)) < thresh
      return is_prismatic

    corr_path = os.path.join(self.dataset.source_path,
                             '../correspondence_loftr/no_filter')
    prismatic = []
    for npz_file in find_files_with_suffix(corr_path, '.npz'):
      corr = np.load(os.path.join(corr_path, npz_file),
                     allow_pickle=True)['data'][0]
      xyz_st, xyz_ed = corr['src_world'], corr['tgt_world']
      for k in np.arange(self.num_movable):
        if k in prismatic:
          continue
        indices = find_close(self.pcds[k].detach().cpu().numpy(),
                             xyz_st,
                             threshold=0.016)
        displacement = xyz_ed[indices] - xyz_st[indices]
        if is_likely_prismatic(displacement):
          print('predicted joint', k, 'as prismatic!')
          self._column_vec1[k].requires_grad_(False)
          self._column_vec2[k].requires_grad_(False)
          prismatic.append(k)
    print('done with joint pred.\n')
    exit(0)

  def _deform_pcds_vectorized(self, r: torch.Tensor, t: torch.Tensor):
    """Vectorized deformation of all point clouds without for loops.
    
    Args:
      r: (K, 3, 3) rotation matrices (actually their transposes)
      t: (K, 3) translation vectors
    
    Updates self.pcds_deformed_all in-place and syncs legacy self.pcds_deformed.
    """
    labels = self.pcd_part_labels  # (N_total,)
    r_per_point = r[labels]  # (N_total, 3, 3)
    t_per_point = t[labels]  # (N_total, 3)
    # Use bmm instead of einsum: (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3) -> squeeze
    # baddbmm fuses matmul and add: out = t + pcds @ r
    self.pcds_deformed_all = torch.baddbmm(
        t_per_point.unsqueeze(1),  # (N, 1, 3)
        self.pcds_all.unsqueeze(1),  # (N, 1, 3)
        r_per_point  # (N, 3, 3)
    ).squeeze(1)  # (N, 3)

    self.pcds_deformed = [
        self.pcds_deformed_all[start:end]
        for (start, end) in self.pcd_part_ranges
    ]

  @override
  def deform(self, iteration: int):
    r, t = self.get_r_t()  # r: (K, 3, 3), t: (K, 3)
    prob = self.get_prob.unsqueeze(-1)
    ppp = self.get_ppp().unsqueeze(-1)

    # Fully vectorized deform with precomputed masks (updated every N iterations)
    num = self.gaussians.size() // (self.num_movable + 1)
    K = self.num_movable
    N = num
    ppp_sq = ppp.squeeze(-1)  # (N, K)
    PROB_THRESHOLD = 0.01

    # Update mask cache periodically (every _mask_cache_interval iterations)
    if (self._mask_cache_iter < 0 or
        iteration - self._mask_cache_iter >= self._mask_cache_interval):
      self._update_mask_cache(ppp_sq, PROB_THRESHOLD)
      self._mask_cache_iter = iteration
      self._reset_cached_part_geometry(num)

    # Use JIT-compiled matrix_to_quaternion
    r_inv_quats = matrix_to_quaternion(r.transpose(-1, -2))  # (K, 4)

    # Vectorized transform using cached active pairs
    if self._cached_max_active > 0:
      part_indices = self._cached_active_part_indices
      src_indices = self._cached_active_src_indices
      dest_indices = self._cached_active_dst_indices

      xyz_gathered = self.original_xyz[src_indices]  # (M, 3)
      rot_gathered = self.original_rotation[src_indices]  # (M, 4)

      xyz_transformed = torch.baddbmm(t[part_indices].unsqueeze(1),
                                      xyz_gathered.unsqueeze(1),
                                      r[part_indices]).squeeze(1)  # (M, 3)

      rot_transformed = quaternion_mul(r_inv_quats[part_indices],
                                       rot_gathered)  # (M, 4)

      self.gaussians.get_xyz[dest_indices] = xyz_transformed
      self.gaussians.get_rotation_raw[dest_indices] = rot_transformed

    # Opacity for all points (fully vectorized, fused computation)
    # original_opacity: (N, 1), prob: (N, 1), ppp_sq: (N, K) -> result: (N, K)
    opacity_all = self.original_opacity * prob * ppp_sq  # (N, K)
    opacity_all = inverse_sigmoid(opacity_all)
    # Use contiguous + view for efficient memory layout
    self.gaussians.get_opacity_raw[num:num *
                                   (K + 1)] = opacity_all.T.contiguous().view(
                                       -1, 1)

    # Fully vectorized pcds_deformed computation
    self._deform_pcds_vectorized(r, t)

    # Background opacity: (1 - prob) * original_opacity
    bg_opacity = (1.0 - prob) * self.original_opacity
    self.gaussians.get_opacity_raw[:num] = inverse_sigmoid(bg_opacity)
    return self.gaussians

  def _reset_cached_part_geometry(self, num: int):
    """Restore duplicated part slots before sparse transform scatter."""
    k = self.num_movable
    self.gaussians.get_xyz[num:num *
                           (k + 1)] = self.original_xyz.unsqueeze(0).expand(
                               k, -1, -1).reshape(-1, 3)
    self.gaussians.get_rotation_raw[num:num *
                                    (k + 1)] = self.original_rotation.unsqueeze(
                                        0).expand(k, -1, -1).reshape(-1, 4)

  def _update_mask_cache(self, ppp_sq: torch.Tensor, threshold: float):
    """Precompute a compact cache of active part-point pairs.

    Instead of padding each part to a shared width, cache only valid pairs
    `(part_idx, point_idx)` whose probability is above the threshold together
    with their duplicated destination slots. This keeps the deform path batched
    while preventing invalid padded writes.
    """
    num = ppp_sq.shape[0]

    masks = ppp_sq >= threshold  # (N, K) bool
    counts = masks.sum(dim=0)  # (K,)
    max_active = counts.max().item()

    active_pairs = masks.transpose(0, 1).nonzero(as_tuple=False)  # (M, 2)

    self._cached_counts = counts
    self._cached_max_active = max_active

    if active_pairs.numel() == 0:
      empty = torch.empty(0, dtype=torch.long, device=ppp_sq.device)
      self._cached_active_part_indices = empty
      self._cached_active_src_indices = empty
      self._cached_active_dst_indices = empty
      return

    part_indices = active_pairs[:, 0].to(dtype=torch.long)
    src_indices = active_pairs[:, 1].to(dtype=torch.long)
    dst_indices = (part_indices + 1) * num + src_indices

    self._cached_active_part_indices = part_indices
    self._cached_active_src_indices = src_indices
    self._cached_active_dst_indices = dst_indices

  def save_ppp_vis(self, path: str):
    mkdir_p(os.path.dirname(path))
    ppp = self.get_ppp()
    fused_color = torch.zeros(self.original_gaussians.size(),
                              3,
                              dtype=torch.float32)
    for k in range(self.num_movable):
      c = torch.tensor(COLORS[k % len(COLORS)], dtype=torch.float32)
      fused_color += ppp[:, k].unsqueeze(1).cpu() * c

    self.original_gaussians.save_vis(path, fused_color)

  def save_mpp_vis(self, path: str):
    mkdir_p(os.path.dirname(path))
    mpp = self.get_prob
    fused_color = value_to_rgb(mpp)
    self.original_gaussians.save_vis(path, fused_color)

  def save_pp_vis(self, path: str):
    mkdir_p(os.path.dirname(path))
    ppp = self.get_ppp()
    mpp = self.get_prob
    fused_color = torch.zeros(self.original_gaussians.size(),
                              3,
                              dtype=torch.float32)
    for k in range(self.num_movable):
      c = torch.tensor(COLORS[k % len(COLORS)], dtype=torch.float32)
      fused_color += ppp[:, k].unsqueeze(1).cpu() * c
    fused_color[mpp < self.opt.mask_thresh] = 0
    self.original_gaussians.save_vis(path, fused_color)

  def save_all_vis(self, iteration=-20):
    pcd_dir = self.dataset.model_path
    self.save_mpp_vis(
        os.path.join(pcd_dir,
                     f'point_cloud/iteration_{iteration + 2}/point_cloud.ply'))
    self.save_ppp_vis(
        os.path.join(pcd_dir,
                     f'point_cloud/iteration_{iteration + 1}/point_cloud.ply'))
    self.save_pp_vis(
        os.path.join(pcd_dir,
                     f'point_cloud/iteration_{iteration}/point_cloud.ply'))

  @override
  def training_setup(self, training_args):

    l = [
        {
            'params': self._column_vec1,
            'lr': training_args.column_lr,
            "name": "column_vec1"
        },
        {
            'params': self._column_vec2,
            'lr': training_args.column_lr,
            "name": "column_vec2"
        },
        {
            'params': self._t,
            'lr': training_args.t_lr * self.gaussians.spatial_lr_scale,
            "name": "t"
        },
        {
            'params': self._c,
            'lr': training_args.c_lr * self.gaussians.spatial_lr_scale,
            "name": "c"
        },
        {
            'params': [self._prob],
            'lr': training_args.prob_lr,
            "name": "prob"
        },
        {
            'params': [self._xyz],
            'lr': training_args.position_lr * self.gaussians.spatial_lr_scale,
            "name": "xyz"
        },
        {
            'params': [self._scaling],
            'lr': training_args.scaling_lr,
            "name": "scaling"
        },
        {
            'params': [self._rotation_col1],
            'lr': training_args.column_lr,
            "name": "rotation_col1"
        },
        {
            'params': [self._rotation_col2],
            'lr': training_args.column_lr,
            "name": "rotation_col2"
        },
        {
            'params': [self._opacity],
            'lr': training_args.opacity_lr,
            "name": "opacity"
        },
    ]
    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    self.quat_scheduler = get_expon_lr_func(
        lr_init=training_args.column_lr_init,
        lr_final=training_args.column_lr_final,
        lr_delay_mult=training_args.position_lr_delay_mult,
        max_steps=training_args.deform_lr_max_steps)
    self.center_scheduler = get_expon_lr_func(
        lr_init=training_args.c_lr_init * self.gaussians.spatial_lr_scale,
        lr_final=training_args.c_lr_final * self.gaussians.spatial_lr_scale,
        lr_delay_mult=training_args.position_lr_delay_mult,
        max_steps=training_args.position_lr_max_steps)
    self.trans_scheduler = get_expon_lr_func(
        lr_init=training_args.t_lr_init * self.gaussians.spatial_lr_scale,
        lr_final=training_args.t_lr_final * self.gaussians.spatial_lr_scale,
        lr_delay_mult=training_args.position_lr_delay_mult * 0.1,
        max_steps=training_args.position_lr_max_steps)
    self.prob_scheduler = get_expon_lr_func(
        lr_init=training_args.prob_lr_init,
        lr_final=training_args.prob_lr_final,
        lr_delay_mult=training_args.position_lr_delay_mult,
        max_steps=training_args.position_lr_max_steps)
    # self._prob.requires_grad_(False)
    if self.num_movable - len(self.merged_parts) == 1:
      self._xyz.requires_grad_(False)
      self._scaling.requires_grad_(False)
      self._rotation_col1.requires_grad_(False)
      self._rotation_col2.requires_grad_(False)
      self._opacity.requires_grad_(False)
    for c in self._c:
      c.requires_grad_(False)

    return

  def update_learning_rate(self, iteration):
    for param_group in self.optimizer.param_groups:
      if param_group["name"] == "column_vec1" or param_group[
          "name"] == "column_vec2":
        lr = self.quat_scheduler(iteration)
        param_group['lr'] = lr
      elif param_group["name"] in ["t"]:
        lr = self.trans_scheduler(iteration)
        param_group['lr'] = lr
      elif param_group["name"] == "c":
        lr = self.center_scheduler(iteration)
        param_group['lr'] = lr
      elif param_group["name"] == "prob":
        lr = self.prob_scheduler(iteration)
        param_group['lr'] = lr

  def _show_losses(self, iteration: int, losses: dict):
    if iteration == self.opt.warmup_until_iter:
      self.save_ppp_vis(
          os.path.join(self.dataset.model_path,
                       f'point_cloud/iteration_{iteration-1}/point_cloud.ply'))
    # if iteration in [1000, 5000, 000, 15000, self.opt.iterations]:
    if iteration == self.opt.iterations:
      self.gaussians.save_ply(os.path.join(
          self.dataset.model_path,
          f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                              prune=False)
    return

  def _eval_losses(self,
                   render_pkg,
                   viewpoint_cam,
                   gaussians,
                   gt_gaussians=None,
                   it=None):
    requires_cd = self.opt.cd_from_iter <= it <= self.opt.cd_until_iter
    requires_ppp = self.opt.merge_iter < it
    gt_image = viewpoint_cam.original_image.cuda().float()
    losses = {
        'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),  # use
        'bce': None,
        'd': None,  # use
        'cd': None,  # use
        'center': None,  # use
        'scaling': None,  # use
        'ppp': None,  # use
        'mpp': None,
        'spar': None,  # use
        'ppped': None,
        'kl': None,
        'sd': None,
        # 'cue': None,  # cue_removed
        # 'limit_center': None,  # use
        'limit_motion': None,
    }
    loss = losses['im']

    if it is not None and it < self.opt.motion_limit_until_iter:
      # r = self.get_r
      c = self._c
      loss_center = self.center_ref.detach() - torch.stack(c)
      losses['limit_motion'] = 1e-2 * torch.mean((loss_center**2).sum(dim=1))
      if self.opt.motion_limit_weight is not None and it % 3:
        motion_trend_loss = self._eval_motion_trend_loss()
        if motion_trend_loss is not None:
          losses['limit_motion'] += (self.opt.motion_limit_weight *
                                     motion_trend_loss)
      loss += losses['limit_motion']

    if (self.opt.cd_weight_const
        is not None) and (gt_gaussians
                          is not None) and it % 3 == 0 and requires_cd:

      pcd_deformed = self.pcds_deformed_all
      # x = sample_pts(pcd_deformed, 5000)
      x = sample_pts(pcd_deformed, -1)
      y = sample_pts(self.pcd_gt, -1)
      dist, _ = chamfer_distance(x.unsqueeze(0),
                                 y.unsqueeze(0),
                                 norm=2,
                                 batch_reduction=None,
                                 single_directional=True)

      losses['cd'] = dist[0]
      loss += self.opt.cd_weight_const * losses['cd']

    if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth
                                                is not None):
      gt_depth = viewpoint_cam.image_depth.cuda().float()
      losses['d'] = eval_depth_loss(render_pkg['depth'],
                                    gt_depth,
                                    scaling=self.opt.depth_scaling)
      loss += self.opt.depth_weight * losses['d']

    if self.opt.center_weight is not None and self.num_movable > 1 and it is not None and it < 1000:
      c = self.center_ref

      losses['center'] = nn.functional.mse_loss(self.get_mu, c)
      loss += self.opt.center_weight * losses['center']

    if self.opt.scaling_weight is not None and self.num_movable > 1 and not requires_ppp and it is not None and it < 1000:
      losses['scaling'] = (self._scaling - self.scaling_ref).pow(2).mean()
      loss += self.opt.scaling_weight * losses['scaling']

    if self.opt.ppp_weight is not None and self.num_movable - len(
        self.merged_parts) > 1 and requires_ppp:
      ppp = self.get_ppp()
      ppp_knn = [ppp[idx] for idx in self.pcd_knn_indices]
      knn_counts = torch.tensor([vals.shape[0] for vals in ppp_knn],
                                device=ppp.device,
                                dtype=ppp.dtype)  # (K,)
      # Compute sum of ppp_knn[i][:, k] for each (i, k) pair
      # ppp_knn_sums[i, k] = ppp_knn[i][:, k].sum()
      ppp_knn_sums = torch.stack([vals.sum(dim=0) for vals in ppp_knn
                                 ])  # (K, K)
      # For each k, total_count = sum of knn_counts[i] for i != k
      total_knn_count = knn_counts.sum()
      total_counts = total_knn_count - knn_counts  # (K,)
      # For each k, total_sum = sum of ppp_knn_sums[i, k] for i != k
      total_sums = ppp_knn_sums.sum(dim=0) - ppp_knn_sums.diag()  # (K,)
      # Build mask for valid parts (not in merged_parts and total_count > 0)
      valid_mask = total_counts > 0
      if len(self.merged_parts) > 0:
        merged_indices = torch.tensor(list(self.merged_parts),
                                      device=ppp.device,
                                      dtype=torch.long)
        valid_mask[merged_indices] = False
      # Compute loss only for valid parts
      if valid_mask.any():
        losses['ppp'] = (total_sums[valid_mask] /
                         total_counts[valid_mask]).sum()
      else:
        losses['ppp'] = torch.tensor(0.0,
                                     device=ppp.device,
                                     dtype=torch.float32)
      loss += self.opt.ppp_weight * losses['ppp'] / (self.num_movable -
                                                     len(self.merged_parts))

    if self.opt.spar_weight is not None:
      mpp = self.get_prob
      neigh_indices = self.original_gaussians.neighbor_indices
      is_not_same = torch.abs(mpp[neigh_indices] - mpp[:, None])
      losses['spar'] = is_not_same.float().mean()
      loss += self.opt.spar_weight * losses['spar']

    return loss, losses

  def _set_ed_knn_indices(self, gt_gaussians: GaussianModel):
    p1 = gt_gaussians.get_xyz.detach().unsqueeze(0)
    p2 = self.original_xyz.detach().unsqueeze(0)
    _, indices, _ = knn_points(p1, p2, K=1)
    self.ed_knn_indices = indices.flatten()

  def _initialize_pairwise_obb_volume_baseline(self):
    if (self.bb_centers is None) or (self.bb_axes is None) or (self.bb_extents
                                                               is None):
      self._obb_pair_volume0 = None
      return

    n = self.num_movable
    if n < 2:
      self._obb_pair_volume0 = None
      return

    resolution = getattr(self.opt, 'obb_volume_resolution', 24)
    baseline = torch.zeros((n, n), dtype=torch.float32)
    with torch.no_grad():
      for i in range(n):
        for j in range(i + 1, n):
          vol = approximate_obb_intersection_volume(self.bb_centers[i],
                                                    self.bb_extents[i],
                                                    self.bb_axes[i],
                                                    self.bb_centers[j],
                                                    self.bb_extents[j],
                                                    self.bb_axes[j], resolution)
          baseline[i, j] = baseline[j, i] = vol

    self._obb_pair_volume0 = baseline

  def _project_raw_translation(self,
                               part_idx: int,
                               axes: torch.Tensor,
                               axis_idx: int,
                               raw_t: torch.Tensor | None = None,
                               local_t_scale: float = 0.5) -> torch.Tensor:
    source_t = self._t[part_idx].detach() if raw_t is None else raw_t.detach()
    basis = axes.to(device=source_t.device, dtype=source_t.dtype)
    local_t = (source_t @ basis.T).clone()
    # local_t[axis_idx] = 0
    local_t[axis_idx] = local_t[axis_idx] * local_t_scale
    return local_t @ basis

  def _axis_component_magnitude(self, part_idx: int, axes: torch.Tensor,
                                axis_idx: int) -> float:
    basis = axes.to(device=self._t[part_idx].device,
                    dtype=self._t[part_idx].dtype)
    local_t = self._t[part_idx].detach() @ basis.T
    return float(torch.abs(local_t[axis_idx]).item())

  def _apply_raw_translation_prune(self, part_idx: int, axes: torch.Tensor,
                                   axis_idx: int) -> bool:
    new_t = self._project_raw_translation(part_idx, axes, axis_idx)
    current_t = self._t[part_idx].detach()
    if torch.allclose(new_t, current_t, atol=1e-6, rtol=1e-5):
      return False

    with torch.no_grad():
      self._t[part_idx].copy_(
          new_t.to(device=self._t[part_idx].device,
                   dtype=self._t[part_idx].dtype))
    return True

  def _apply_raw_rotation_halve(self, part_idx: int) -> bool:
    """Halve the rotation angle of the given part."""
    rotation = self.get_r[part_idx].detach()
    eps = 1e-6
    cos_term = torch.clamp(((torch.trace(rotation) - 1.0) * 0.5), -1.0 + eps,
                           1.0 - eps)
    angle = torch.arccos(cos_term)
    if angle < eps:
      return False
    axis = rotation_axis_from_matrix(rotation)
    new_angle = angle * 0.5
    new_rotation = axis_angle_to_matrix_torch(axis, new_angle)
    with torch.no_grad():
      self._column_vec1[part_idx][:] = new_rotation[:, 0]
      self._column_vec2[part_idx][:] = new_rotation[:, 1]
    return True

  def _freeze_rotation_of_part(self,
                               part_idx: int,
                               reason: str | None = None) -> bool:
    target1 = torch.tensor([1., 0., 0.],
                           dtype=self._column_vec1[part_idx].dtype,
                           device=self._column_vec1[part_idx].device)
    target2 = torch.tensor([0., 1., 0.],
                           dtype=self._column_vec2[part_idx].dtype,
                           device=self._column_vec2[part_idx].device)
    already_frozen = (
        not self._column_vec1[part_idx].requires_grad and
        not self._column_vec2[part_idx].requires_grad and
        torch.allclose(self._column_vec1[part_idx].detach(), target1) and
        torch.allclose(self._column_vec2[part_idx].detach(), target2))
    if already_frozen:
      return False

    if reason is not None:
      print(reason)

    self._column_vec1[part_idx].requires_grad_(False)
    self._column_vec2[part_idx].requires_grad_(False)
    with torch.no_grad():
      self._column_vec1[part_idx].copy_(target1)
      self._column_vec2[part_idx].copy_(target2)
    return True

  def _refresh_deformed_bbs(
      self,
      part_indices: list[int] | range | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    r, t = self.get_r_t()
    indices = range(self.num_movable) if part_indices is None else part_indices
    for k in indices:
      self.bb_centers_deformed[k] = self.bb_centers[k] @ r[k] + t[k]
      self.bb_axes_deformed[k] = self.bb_axes[k] @ r[k]
    return r, t

  def _world_translation_from_raw(
      self,
      part_idx: int,
      rotation: torch.Tensor,
      raw_t: torch.Tensor | None = None) -> torch.Tensor:
    center = self._c[part_idx].detach().to(device=rotation.device,
                                           dtype=rotation.dtype)
    source_t = self._t[part_idx].detach() if raw_t is None else raw_t.detach()
    source_t = source_t.to(device=rotation.device, dtype=rotation.dtype)
    rotation = rotation.detach()
    return center - center @ rotation + source_t

  def _score_obb_prune_candidate(self, moving_idx: int, axis_idx: int,
                                 other_idx: int, rotation: torch.Tensor,
                                 resolution: int,
                                 current_volume: float) -> dict:
    axes = self.bb_axes[moving_idx]
    pruned_t = self._project_raw_translation(moving_idx, axes, axis_idx)
    world_t = self._world_translation_from_raw(moving_idx, rotation, pruned_t)
    center = self.bb_centers[moving_idx] @ rotation.detach() + world_t
    directions = self.bb_axes[moving_idx] @ rotation.detach()
    remaining = approximate_obb_intersection_volume(
        center, self.bb_extents[moving_idx], directions,
        self.bb_centers_deformed[other_idx], self.bb_extents[other_idx],
        self.bb_axes_deformed[other_idx], resolution)
    return {
        'part_idx':
            moving_idx,
        'axis_idx':
            axis_idx,
        'axes':
            axes,
        'reduction':
            current_volume - remaining,
        'remaining':
            remaining,
        'axis_magnitude':
            self._axis_component_magnitude(moving_idx, axes, axis_idx),
    }

  def _select_best_collision_prune(self, candidates: list[dict]) -> dict | None:
    if not candidates:
      return None

    best = max(candidates,
               key=lambda candidate:
               (candidate['reduction'], candidate['axis_magnitude']))
    reduction_tol = getattr(self.opt, 'collision_prune_reduction_tol', 1e-6)
    if best['reduction'] <= reduction_tol:
      return None
    return best

  def _should_ignore_similar_motion_collision(
      self, part_idx1: int, axis_idx1: int, part_idx2: int, axis_idx2: int,
      translations: torch.Tensor) -> bool:
    if axis_idx1 < 0 or axis_idx2 < 0:
      return False

    bb_axes_deformed = self.bb_axes_deformed
    if bb_axes_deformed is None:
      return False

    max_rot_deg = getattr(self.opt, 'similar_motion_collision_max_rot_deg', 5.0)
    dir_cos_thresh = getattr(self.opt, 'similar_motion_collision_cos_thresh',
                             0.9)
    normal_rel_ratio = getattr(self.opt,
                               'similar_motion_collision_normal_ratio', 0.25)
    min_motion = getattr(self.opt, 'similar_motion_collision_min_motion', 0.2)

    motions: list[tuple[torch.Tensor, float]] = []
    for part_idx in [part_idx1, part_idx2]:
      motion = translations[part_idx].detach()
      motion_norm = float(torch.linalg.norm(motion).item())
      if motion_norm <= min_motion:
        return False

      rotation = self.get_r[part_idx].detach()
      eps = 1e-6
      cos_term = torch.clamp(((torch.trace(rotation) - 1.0) * 0.5), -1.0 + eps,
                             1.0 - eps)
      angle_deg = float((torch.arccos(cos_term) * (180.0 / math.pi)).item())
      if angle_deg > max_rot_deg:
        return False

      motions.append((motion, motion_norm))

    dir1 = motions[0][0] / motions[0][1]
    dir2 = motions[1][0] / motions[1][1]
    direction_cos = float(torch.dot(dir1, dir2).item())
    if direction_cos < dir_cos_thresh:
      return False

    rel_motion = motions[0][0] - motions[1][0]
    avg_motion = 0.5 * (motions[0][1] + motions[1][1])
    if avg_motion <= min_motion:
      return False

    normal1 = bb_axes_deformed[part_idx1][axis_idx1].detach()
    normal2 = bb_axes_deformed[part_idx2][axis_idx2].detach()
    normal1 = normal1 / (torch.linalg.norm(normal1) + 1e-8)
    normal2 = normal2 / (torch.linalg.norm(normal2) + 1e-8)
    normal_rel = max(float(torch.abs(torch.dot(rel_motion, normal1)).item()),
                     float(torch.abs(torch.dot(rel_motion, normal2)).item()))
    if normal_rel > avg_motion * normal_rel_ratio:
      return False

    print('Skip similar-motion collision between '
          f'part {part_idx1} and part {part_idx2} '
          f'(cos={direction_cos:.3f}, rel_normal={normal_rel:.6f}).')
    return True

  def use_priors(self, iteration: int):
    if iteration > self.opt.prior_until_iter or self.bb_axes is None:
      return

    if self.collisions:
      self.collisions.clear()

    # deform BBs
    r, t = self._refresh_deformed_bbs()

    # detects collision
    delta_thresh = getattr(self.opt, 'obb_collision_delta', 1e-2)
    resolution = getattr(self.opt, 'obb_volume_resolution', 24)
    baseline_volumes = self._obb_pair_volume0
    freeze_rotation = getattr(
        self.opt, 'collision_freeze_rotation',
        getattr(self.opt, 'obb_collision_freeze_rot', False))

    for k1 in range(self.num_movable):
      for k2 in range(k1 + 1, self.num_movable):
        if not self.neighbors_mat[k1, k2]:
          continue

        vol0 = 0.0
        if baseline_volumes is not None:
          vol0 = float(baseline_volumes[k1, k2].item())

        vol1 = approximate_obb_intersection_volume(
            self.bb_centers_deformed[k1], self.bb_extents[k1],
            self.bb_axes_deformed[k1], self.bb_centers_deformed[k2],
            self.bb_extents[k2], self.bb_axes_deformed[k2], resolution)

        if (vol1 - vol0) <= delta_thresh:
          continue

        has_collision, idx1, idx2 = get_bb_collision_axis_torch(
            self.bb_centers_deformed[k1],
            self.bb_extents[k1],
            self.bb_axes_deformed[k1],
            self.bb_centers_deformed[k2],
            self.bb_extents[k2],
            self.bb_axes_deformed[k2],
        )

        if not has_collision:
          continue

        if self._should_ignore_similar_motion_collision(k1, idx1, k2, idx2, t):
          continue

        rotation_adjusted = False
        if getattr(self.opt, 'align_to_principal_axis', True):
          adjusted1 = self._maybe_suppress_revolute_rotation(k1)
          adjusted2 = self._maybe_suppress_revolute_rotation(k2)
          rotation_adjusted = adjusted1 or adjusted2

        if rotation_adjusted:
          r, t = self._refresh_deformed_bbs([k1, k2])
          vol1 = approximate_obb_intersection_volume(
              self.bb_centers_deformed[k1], self.bb_extents[k1],
              self.bb_axes_deformed[k1], self.bb_centers_deformed[k2],
              self.bb_extents[k2], self.bb_axes_deformed[k2], resolution)
          if (vol1 - vol0) <= delta_thresh:
            continue

          has_collision, idx1, idx2 = get_bb_collision_axis_torch(
              self.bb_centers_deformed[k1],
              self.bb_extents[k1],
              self.bb_axes_deformed[k1],
              self.bb_centers_deformed[k2],
              self.bb_extents[k2],
              self.bb_axes_deformed[k2],
          )
          if not has_collision:
            continue

          if self._should_ignore_similar_motion_collision(
              k1, idx1, k2, idx2, t):
            continue

        best = self._select_best_collision_prune([
            self._score_obb_prune_candidate(k1, idx1, k2, r[k1], resolution,
                                            vol1),
            self._score_obb_prune_candidate(k2, idx2, k1, r[k2], resolution,
                                            vol1),
        ])
        if best is None:
          continue

        t_pruned = self._apply_raw_translation_prune(best['part_idx'],
                                                     best['axes'],
                                                     best['axis_idx'])
        r_pruned = self._apply_raw_rotation_halve(best['part_idx'])
        if not t_pruned and not r_pruned:
          continue

        print('Apply transient prior prune to '
              f'part {best["part_idx"]} axis {best["axis_idx"]} '
              f'(ΔV={best["reduction"]:.6f}, V={best["remaining"]:.6f})'
              f'(t_pruned={t_pruned}, r_pruned={r_pruned}).')

        # if freeze_rotation:
        #   self._freeze_rotation_of_part(
        #       best['part_idx'],
        #       f'Freezing rotation of part {best["part_idx"]} due to collision.')

        r, t = self._refresh_deformed_bbs([best['part_idx']])
    return

  def _maybe_suppress_revolute_rotation(self, part_idx: int):
    """Snap tiny revolute proposals to the nearest OBB edge to avoid drift."""
    if part_idx >= len(self.is_revolute) or not self.is_revolute[part_idx]:
      return False

    rotation = self.get_r[part_idx].detach()
    eps = 1e-6
    cos_term = torch.clamp(((torch.trace(rotation) - 1.0) * 0.5), -1.0 + eps,
                           1.0 - eps)
    angle = torch.arccos(cos_term)
    angle_deg = angle * (180.0 / math.pi)
    threshold = getattr(self.opt, 'revolute_identity_thresh_deg', 2.5)
    if angle_deg > threshold:
      return False

    print(f'Aligning revolute axis of part {part_idx} to OBB axis.')
    axis = rotation_axis_from_matrix(rotation)
    snapped_axis = self._snap_axis_to_obb(axis, part_idx)
    angle_scale = 0.5

    new_angle = angle * angle_scale
    new_rotation = axis_angle_to_matrix_torch(snapped_axis, new_angle)

    with torch.no_grad():
      self._column_vec1[part_idx][:] = new_rotation[:, 0]
      self._column_vec2[part_idx][:] = new_rotation[:, 1]
    return True

  def _snap_axis_to_obb(self, axis: torch.Tensor,
                        part_idx: int) -> torch.Tensor:
    bb_axes = self.bb_axes[part_idx].to(device=axis.device, dtype=axis.dtype)
    bb_axes = F.normalize(bb_axes, dim=1)
    dots = torch.abs(torch.mv(bb_axes, axis))
    best_idx = int(torch.argmax(dots).item())
    snapped = bb_axes[best_idx]
    if torch.dot(snapped, axis) < 0:
      snapped = -snapped
    return F.normalize(snapped, dim=0)

  def _refresh_rotation_params(self):
    """Project per-part 6D rotation parameters back to an orthonormal basis."""
    with torch.no_grad():
      for idx in range(self.num_movable):
        basis = self.gram_schmidt(self._column_vec1[idx],
                                  self._column_vec2[idx])
        self._column_vec1[idx].copy_(basis[:, 0])
        self._column_vec2[idx].copy_(basis[:, 1])

  def _rotation_to_translation_candidate(self, part_idx: int):
    if self.bb_centers is None or self.bb_extents is None or self.bb_axes is None:
      return None

    rotation = self.get_r[part_idx].detach()
    world_t = self.get_t[part_idx].detach().to(device=rotation.device,
                                               dtype=rotation.dtype)
    raw_t = self._t[part_idx].detach().to(device=rotation.device,
                                          dtype=rotation.dtype)
    bb_center = self.bb_centers[part_idx].detach().to(device=rotation.device,
                                                      dtype=rotation.dtype)
    bb_extents = self.bb_extents[part_idx].detach().to(device=rotation.device,
                                                       dtype=rotation.dtype)
    bb_axes = self.bb_axes[part_idx].detach().to(device=rotation.device,
                                                 dtype=rotation.dtype)

    equivalent_translation = bb_center @ rotation + world_t - bb_center
    rotation_shift = equivalent_translation - raw_t
    rotation_shift_norm = torch.linalg.norm(rotation_shift)

    part_radius = torch.linalg.norm(bb_extents)
    min_shift = torch.maximum(
        rotation.new_tensor(
            getattr(self.opt, 'rotation_to_translation_min_shift', 5e-3)),
        part_radius * getattr(
            self.opt, 'rotation_to_translation_min_shift_extent_ratio', 0.25))
    if rotation_shift_norm <= min_shift:
      return None

    eps = 1e-6
    cos_term = torch.clamp(((torch.trace(rotation) - 1.0) * 0.5), -1.0 + eps,
                           1.0 - eps)
    angle = torch.arccos(cos_term)
    angle_deg = angle * (180.0 / math.pi)
    if angle_deg > getattr(self.opt, 'rotation_to_translation_max_angle_deg',
                           8.0):
      return None

    signs = rotation.new_tensor([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ])
    corner_offsets = (signs * bb_extents.unsqueeze(0)) @ bb_axes
    corner_residual = corner_offsets @ rotation - corner_offsets
    max_corner_residual = torch.linalg.norm(corner_residual, dim=1).max()
    residual_ratio = max_corner_residual / (rotation_shift_norm + eps)
    if residual_ratio > getattr(self.opt,
                                'rotation_to_translation_max_residual_ratio',
                                0.1):
      return None

    return {
        'equivalent_translation': equivalent_translation,
        'rotation_shift_norm': rotation_shift_norm.item(),
        'max_corner_residual': max_corner_residual.item(),
        'residual_ratio': residual_ratio.item(),
        'angle_deg': angle_deg.item(),
    }

  def _maybe_convert_rotation_to_translation(self, part_idx: int):
    candidate = self._rotation_to_translation_candidate(part_idx)
    if candidate is None:
      return False

    print('Convert far-center rotation of '
          f'part {part_idx} to translation '
          f'(angle={candidate["angle_deg"]:.2f} deg, '
          f'shift={candidate["rotation_shift_norm"]:.4f}, '
          f'residual={candidate["max_corner_residual"]:.4f}, '
          f'ratio={candidate["residual_ratio"]:.4f}).')
    with torch.no_grad():
      self._column_vec1[part_idx][:] = torch.tensor(
          [1.0, 0.0, 0.0],
          dtype=self._column_vec1[part_idx].dtype,
          device=self._column_vec1[part_idx].device)
      self._column_vec2[part_idx][:] = torch.tensor(
          [0.0, 1.0, 0.0],
          dtype=self._column_vec2[part_idx].dtype,
          device=self._column_vec2[part_idx].device)
      self._t[part_idx].copy_(candidate['equivalent_translation'].to(
          device=self._t[part_idx].device, dtype=self._t[part_idx].dtype))
      self._c[part_idx].zero_()
      self._column_vec1[part_idx].grad = None
      self._column_vec2[part_idx].grad = None
      self._t[part_idx].grad = None
      self._c[part_idx].grad = None
    if part_idx < len(self.is_revolute):
      self.is_revolute[part_idx] = False
    return True

  def pretrain(self):
    print(
        f"[PPP Pretraining] {self.opt.ppp_pretrain_iters} iters of {self.num_movable}-part PPP pretraining"
    )
    for _ in enumerate(
        tqdm(range(self.opt.ppp_pretrain_iters), desc="Training progress")):
      loss = 0.0
      ppp = self.get_ppp()
      ppp_knn = [ppp[idx] for idx in self.pcd_knn_indices]
      for k in range(self.num_movable):
        loss += sum(
            ppp_knn[i][:, k].sum() for i in range(self.num_movable) if i != k)

      loss.backward()
      with (torch.no_grad()):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._prob[:] = torch.clamp(self._prob, -16, 16)
        self._opacity[:] = torch.clamp(self._opacity, -16, 16)
        self._scaling[:] = torch.clamp(self._scaling, -16, 16)

        self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
        self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
        self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach(
        )
        self.ppp = None

    with (torch.no_grad()):
      self.center_ref = self._xyz.clone().detach()
      self.scaling_ref = self._scaling.clone().detach()
      self.save_all_vis(1)

  @override
  def train(self,
            gt_gaussians,
            gt_num=None,
            curr_iter=None,
            iteration=None,
            bws=None):
    if gt_num is None:
      gt_num = self.num_movable
    if bws is None:
      bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
    if iteration is None and curr_iter is None:
      _ = prepare_output_and_logger(self.dataset)
      iterations = self.opt.iterations

      self.training_setup(self.opt)
      self._set_ed_knn_indices(gt_gaussians)
      self.original_gaussians.initialize_neighbors(num_knn=20, simple=True)

      if self.num_movable > 1:
        self.pretrain()

      progress_bar = tqdm(range(iterations), desc="Training progress")
      begin_iter = 1
      self.opt.early_stop_thresh_r = 1e-3
      self.opt.early_stop_thresh_t = 4e-4
    else:
      progress_bar = tqdm(range(curr_iter, iteration), desc="Training progress")
      begin_iter = curr_iter + 1
      iterations = iteration
      self.opt.early_stop_iter = 12500
      self.opt.early_stop_thresh_r = 1e-4
      self.opt.early_stop_thresh_t = 1e-4

    ema_loss_for_log = 0.0
    # Early stopping variables
    prev_r = None
    prev_t = None
    ema_r_diff = None  # EMA of rotation difference
    ema_t_diff = None  # EMA of translation difference
    ema_alpha = 0.3  # EMA smoothing factor
    converge_count = 0  # Counter for consecutive convergence checks
    early_stop_iter = getattr(self.opt, 'early_stop_iter', 2500)
    early_stop_thresh_r = getattr(self.opt, 'early_stop_thresh_r', 1e-4)
    early_stop_thresh_t = getattr(self.opt, 'early_stop_thresh_t', 1e-4)

    for i in range(begin_iter, iterations + 1):
      if self.opt.cd_weight is not None:
        self.opt.cd_weight = self.cosine_anneal(
            i,
            final_step=self.opt.cd_until_iter,
            start_step=self.opt.cd_from_iter,
            start_value=self.opt.cd_from_weight,
            final_value=self.opt.cd_until_weight)

      if i < 4000 and i % 500 == 0:
        self._refresh_rotation_params()
      self.deform(i)

      # Pick a random Camera
      viewpoint_cam, background = bws.pop_black() if (
          i % 2 == 0) else bws.pop_white()
      render_pkg = render(viewpoint_cam,
                          self.gaussians,
                          self.pipe,
                          background,
                          opacity_filter=0.01)

      loss, losses = self._eval_losses(render_pkg,
                                       viewpoint_cam,
                                       self.gaussians,
                                       gt_gaussians,
                                       it=i)
      try:
        loss.backward(retain_graph=True)
      except RuntimeError:
        print('iteration:', i)
        loss.backward(retain_graph=True)

      with (torch.no_grad()):
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if i % 10 == 0:
          progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
          progress_bar.update(10)

        if i < iterations:
          if i % self.opt.sgd_interval == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.update_learning_rate(i)
            self._prob[:] = torch.clamp(self._prob, -16, 16)
            self._opacity[:] = torch.clamp(self._opacity, -16, 16)
            self._scaling[:] = torch.clamp(self._scaling, -16, 16)
          self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach(
          )
          self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
          self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach(
          )
          self.ppp = None

        conversion_interval = getattr(self.opt,
                                      'rotation_to_translation_interval', 100)
        if conversion_interval > 0 and i % conversion_interval == 0:
          for k in range(self.num_movable):
            self._maybe_convert_rotation_to_translation(k)

        if i % 500 == 0:
          # check for NaN or Inf in motion parameters; reset if exploded
          self._check_and_reset_motion_params(i)

        if i > 1000 and i % 200 == 0:
          self.use_priors(i)

        if i == self.opt.cd_until_iter:
          for k in range(self.num_movable):
            self.is_revolute[k] = (torch.trace(self.get_r[k])
                                   < self.opt.trace_r_thresh_loose)
            print(f'Detected part{k} is ' +
                  ('REVOLUTE' if self.is_revolute[k] else 'PRISMATIC'))
            if self.is_revolute[k]:
              # Re-parameterize so revolute parts use the solved rotation center only.
              rotation = self.get_r[k].detach()
              translation = self.get_t[k].detach()
              eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
              center_matrix = eye - rotation
              center = torch.matmul(translation,
                                    torch.linalg.pinv(center_matrix))
              if not torch.isfinite(center).all():
                center = torch.zeros_like(center)
              self._c[k].copy_(
                  center.to(dtype=self._c[k].dtype, device=self._c[k].device))
              self._t[k].zero_()
              self._t[k].requires_grad_(False)
              self._t[k].grad = None
              continue

            self._column_vec1[k] = nn.Parameter(
                torch.tensor([1, 0, 0], dtype=torch.float,
                             device='cuda').requires_grad_(False))
            self._column_vec2[k] = nn.Parameter(
                torch.tensor([0, 1, 0], dtype=torch.float,
                             device='cuda').requires_grad_(False))

        if i == self.opt.warmup_until_iter:
          print('')
          for k in np.arange(self.num_movable):
            self.is_revolute[k] = (torch.trace(self.get_r[k])
                                   < self.opt.trace_r_thresh)
            print(f'Detected part{k} is ' +
                  ('REVOLUTE' if self.is_revolute[k] else 'PRISMATIC'))
            if self.is_revolute[k]:
              continue
            # self._column_vec1[k] = nn.Parameter(
            #     torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
            # )
            # self._column_vec2[k] = nn.Parameter(
            #     torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
            # )
          if self.num_movable > 1:
            self._xyz.requires_grad_(True)
            self._scaling.requires_grad_(True)
            self._rotation_col1.requires_grad_(True)
            self._rotation_col2.requires_grad_(True)
            self._opacity.requires_grad_(True)
          self._prob.requires_grad_(True)
        self._show_losses(i, losses)

        torch.cuda.empty_cache()

        # Early stopping check, todo: improve this algorithm
        if False and i > early_stop_iter and i % 100 == 0:
          curr_r = [r.detach().clone() for r in self.get_r]
          curr_t = [t.detach().clone() for t in self.get_t]

          if prev_r is not None and prev_t is not None:
            # Compute instantaneous difference between current and previous r, t
            r_diff = sum(
                torch.norm(curr_r[k] - prev_r[k]).item()
                for k in range(self.num_movable)) / self.num_movable
            t_diff = sum(
                torch.norm(curr_t[k] - prev_t[k]).item()
                for k in range(self.num_movable)) / self.num_movable

            # Update EMA of differences
            if ema_r_diff is None:
              ema_r_diff = r_diff
              ema_t_diff = t_diff
            else:
              ema_r_diff = ema_alpha * r_diff + (1 - ema_alpha) * ema_r_diff
              ema_t_diff = ema_alpha * t_diff + (1 - ema_alpha) * ema_t_diff

            # Check if EMA is below threshold
            # Todo, this alg needs further improvement.
            if ema_r_diff < early_stop_thresh_r and ema_t_diff < early_stop_thresh_t:
              converge_count += 1
              print(
                  f'[Early Stop Check] Iteration {i}: ema_r_diff={ema_r_diff:.2e}, ema_t_diff={ema_t_diff:.2e} (converge_count={converge_count}/100)'
              )
              if converge_count >= 100:
                print(
                    f'\n[Early Stop] Iteration {i}: ema_r_diff={ema_r_diff:.2e}, ema_t_diff={ema_t_diff:.2e}'
                )
                print(
                    f'Parameters converged for 100 consecutive checks (thresholds: r={early_stop_thresh_r}, t={early_stop_thresh_t})'
                )
                break
            else:
              converge_count = 0  # Reset counter if not converged
              print(
                  f'[Early Stop Check] Iteration {i}: ema_r_diff={ema_r_diff:.2e}, ema_t_diff={ema_t_diff:.2e} (continuing)'
              )

          prev_r = curr_r
          prev_t = curr_t

    progress_bar.close()
    return self.get_t, self.get_r
