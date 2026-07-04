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

os.environ['USE_KEOPS'] = '1'

from arguments import GroupParams
from scene.gaussian_model import GaussianModel

COLORS = np.array(plt.get_cmap("tab20").colors)[:, :3]


class MPArtModelBasic:

  def setup_args(self):
    self.dataset.sh_degree = 0
    self.dataset.source_path = ""
    self.dataset.model_path = ""
    self.dataset.images = "images"
    self.dataset.resolution = -1
    self.dataset.white_background = False
    self.dataset.data_device = "cuda"
    self.dataset.eval = False

    self.pipe.convert_SHs_python = False
    self.pipe.compute_cov3D_python = False
    self.pipe.debug = False

    self.opt.iterations = 10_000
    self.opt.percent_dense = 0.01
    self.opt.lambda_dssim = 0.2
    self.opt.column_lr = 0.005
    self.opt.t_lr = 0.00005

    self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)
    self.opt.trace_r_thresh_tight = 1 + 2 * math.cos(.1 / 180 * math.pi)
    self.opt.trace_r_thresh_loose = 1 + 2 * math.cos(10. / 180 * math.pi)

  def __init__(self, gaussians: GaussianModel, num_movable: int):
    self.num_movable = num_movable

    self._column_vec1 = [
        nn.Parameter(
            torch.tensor([1, 0, 0], dtype=torch.float,
                         device='cuda').requires_grad_(True))
        for _ in range(self.num_movable)
    ]
    self._column_vec2 = [
        nn.Parameter(
            torch.tensor([0, 1, 0], dtype=torch.float,
                         device='cuda').requires_grad_(True))
        for _ in range(self.num_movable)
    ]
    self._t = [
        nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float,
                         device='cuda').requires_grad_(True))
        for _ in range(self.num_movable)
    ]
    self._c = [
        nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float,
                         device='cuda').requires_grad_(True))
        for _ in range(self.num_movable)
    ]
    # Saved initial motion params for NaN/Inf recovery
    self._init_column_vec1 = None
    self._init_column_vec2 = None
    self._init_t = None
    self._init_c = None
    self.r_activation = None
    self.gaussians = gaussians
    self.optimizer = None
    self.dataset = GroupParams()  # ed
    self.opt = GroupParams()
    self.pipe = GroupParams()
    self.setup_function()

  @staticmethod
  def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
    eps = 1e-11
    norm_a1 = torch.norm(a1)
    b1 = a1 / norm_a1

    b2 = a2 - (b1 @ a2) * b1
    norm_b2 = torch.norm(b2)
    assert norm_b2 > eps
    b2 = b2 / norm_b2

    b3 = torch.linalg.cross(b1, b2)
    return torch.cat([b1.view(3, 1), b2.view(3, 1), b3.view(3, 1)], dim=1)

  @staticmethod
  def gram_schmidt_batch(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """Batched Gram-Schmidt orthogonalization.
    Args:
      a1: (K, 3) tensor
      a2: (K, 3) tensor
    Returns:
      (K, 3, 3) rotation matrices (actually their transposes)
    """
    eps = 1e-11
    norm_a1 = torch.norm(a1, dim=-1, keepdim=True)
    b1 = a1 / norm_a1  # (K, 3)

    proj = (b1 * a2).sum(dim=-1, keepdim=True)  # (K, 1)
    b2 = a2 - proj * b1  # (K, 3)
    norm_b2 = torch.norm(b2, dim=-1, keepdim=True)
    b2 = b2 / (norm_b2 + eps)  # (K, 3)

    b3 = torch.linalg.cross(b1, b2)  # (K, 3)
    return torch.stack([b1, b2, b3], dim=-1)  # (K, 3, 3)

  def _capture_initial_motion_params(self):
    """Snapshot motion params right after init, for NaN/Inf recovery."""
    self._init_column_vec1 = [v.detach().clone() for v in self._column_vec1]
    self._init_column_vec2 = [v.detach().clone() for v in self._column_vec2]
    self._init_t = [v.detach().clone() for v in self._t]
    self._init_c = [v.detach().clone() for v in self._c]

  def _check_and_reset_motion_params(self, iteration: int):
    """Check motion params for NaN/Inf; reset to initial values if found.

    Resets _column_vec1[k], _column_vec2[k], _t[k], _c[k] for any part k
    that has a non-finite value. Also clears gradients and optimizer state.
    """
    fallback_col1 = torch.tensor([1.0, 0.0, 0.0], device='cuda')
    fallback_col2 = torch.tensor([0.0, 1.0, 0.0], device='cuda')
    fallback_t = torch.zeros(3, device='cuda')
    fallback_c = torch.zeros(3, device='cuda')

    has_saved = self._init_column_vec1 is not None
    any_reset = False

    for k in range(self.num_movable):
      params_k = [
          self._column_vec1[k], self._column_vec2[k], self._t[k], self._c[k]
      ]
      if all(torch.isfinite(p).all() for p in params_k):
        continue

      print(f'[iter {iteration}] Part {k}: NaN/Inf detected in motion '
            f'params, resetting to initial values.')
      col1 = self._init_column_vec1[k] if has_saved else fallback_col1
      col2 = self._init_column_vec2[k] if has_saved else fallback_col2
      t_val = self._init_t[k] if has_saved else fallback_t
      c_val = self._init_c[k] if has_saved else fallback_c

      self._column_vec1[k].data.copy_(col1)
      self._column_vec2[k].data.copy_(col2)
      self._t[k].data.copy_(t_val)
      self._c[k].data.copy_(c_val)

      for p in params_k:
        if p.grad is not None:
          p.grad.zero_()
        if self.optimizer is not None and p in self.optimizer.state:
          self.optimizer.state[p] = {}

      any_reset = True

    return any_reset

  def set_dataset(self, source_path: str, model_path: str, evaluate=True):
    self.dataset.eval = evaluate
    self.dataset.source_path = source_path
    self.dataset.model_path = model_path

  def setup_function(self):
    self.r_activation = self.gram_schmidt
    self.gaussians.cancel_grads()
    self.setup_args()

  def get_r_t(self):
    """Vectorized computation of rotation and translation.
    Returns:
      r: (K, 3, 3) tensor of rotation matrices (actually their transposes)
      t: (K, 3) tensor of translations
    """
    col1 = torch.stack(self._column_vec1)  # (K, 3)
    col2 = torch.stack(self._column_vec2)  # (K, 3)
    r = self.gram_schmidt_batch(col1, col2)  # (K, 3, 3)

    c = torch.stack(self._c)  # (K, 3)
    t_raw = torch.stack(self._t)  # (K, 3)
    # t = -c @ r + c + t_raw => t = c - c @ r + t_raw
    t = c - torch.einsum('ki,kij->kj', c, r) + t_raw  # (K, 3)
    return r, t

  @property
  def get_t(self):
    """Returns list of (3,) translation vectors for backward compatibility."""
    _, t = self.get_r_t()
    return [t[k] for k in range(self.num_movable)]

  @property
  def get_r(self):
    """Returns list of (3, 3) rotation matrices for backward compatibility.
    Note: r is actually the transpose of the rotation matrix !!!
    """
    r, _ = self.get_r_t()
    return [r[k] for k in range(self.num_movable)]

  def set_init_params(self, t, r):
    """
        :param t: list of (3, 3) rotation matrices
        :param r: list of (3,) translation vectors
        """
    self._t = [
        nn.Parameter(
            torch.tensor(tt, dtype=torch.float,
                         device='cuda').requires_grad_(True)) for tt in t
    ]

    self._column_vec1 = [
        nn.Parameter(
            torch.tensor(rr[:, 0], dtype=torch.float,
                         device='cuda').requires_grad_(True)) for rr in r
    ]
    self._column_vec2 = [
        nn.Parameter(
            torch.tensor(rr[:, 1], dtype=torch.float,
                         device='cuda').requires_grad_(True)) for rr in r
    ]

  def deform(self, iteration: int):
    pass

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
            'lr': training_args.t_lr * self.gaussians.spatial_lr_scale,
            "name": "c"
        },
    ]
    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    for c in self._c:
      c.requires_grad_(False)

  def train(self, gt_gaussians=None):
    pass
