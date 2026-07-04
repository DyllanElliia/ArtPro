#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
import sys
from datetime import datetime
import numpy as np
import random
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from plyfile import PlyData
import matplotlib.pyplot as plt

from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_ply, load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from sklearn.decomposition import PCA
import open3d as o3d
import pyvista as pv
from sklearn.cluster import KMeans
from tqdm import tqdm
from joblib import Parallel, delayed


class DisjointSet:

  def __init__(self, n: int):
    self.parent = np.arange(n)

  def find(self, x):
    if self.parent[x] != x:
      self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

  def is_connected(self, x, y):
    return self.find(x) == self.find(y)

  def connect(self, x, y):
    """
        :param x: merged
        :param y: retained
        """
    self.parent[self.find(x)] = self.find(y)

  def get_new_indices(self):
    u = np.unique(self.parent)
    return {u[k]: k for k in range(len(u))}, u


def inverse_sigmoid(x):
  return torch.log(x / (1 - x))


def inverse_softmax(x):
  return torch.log(x)


def PILtoTorch(pil_image, resolution):
  resized_image_PIL = pil_image.resize(resolution)
  resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
  if len(resized_image.shape) == 3:
    return resized_image.permute(2, 0, 1)
  else:
    return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init,
                      lr_final,
                      lr_delay_steps=0,
                      lr_delay_mult=1.0,
                      max_steps=1000000):
  """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

  def helper(step):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
      # Disable this parameter
      return 0.0
    if lr_delay_steps > 0:
      # A kind of reverse cosine decay.
      delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
          0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
      delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp

  return helper


def strip_lowerdiag(L):
  uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

  uncertainty[:, 0] = L[:, 0, 0]
  uncertainty[:, 1] = L[:, 0, 1]
  uncertainty[:, 2] = L[:, 0, 2]
  uncertainty[:, 3] = L[:, 1, 1]
  uncertainty[:, 4] = L[:, 1, 2]
  uncertainty[:, 5] = L[:, 2, 2]
  return uncertainty


def strip_symmetric(sym):
  return strip_lowerdiag(sym)


def build_rotation(r):
  norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] +
                    r[:, 3] * r[:, 3])

  q = r / norm[:, None]

  R = torch.zeros((q.size(0), 3, 3), device='cuda')

  r = q[:, 0]
  x = q[:, 1]
  y = q[:, 2]
  z = q[:, 3]

  R[:, 0, 0] = 1 - 2 * (y * y + z * z)
  R[:, 0, 1] = 2 * (x * y - r * z)
  R[:, 0, 2] = 2 * (x * z + r * y)
  R[:, 1, 0] = 2 * (x * y + r * z)
  R[:, 1, 1] = 1 - 2 * (x * x + z * z)
  R[:, 1, 2] = 2 * (y * z - r * x)
  R[:, 2, 0] = 2 * (x * z - r * y)
  R[:, 2, 1] = 2 * (y * z + r * x)
  R[:, 2, 2] = 1 - 2 * (x * x + y * y)
  return R


def build_scaling_rotation(s, r):
  L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
  R = build_rotation(r)

  L[:, 0, 0] = s[:, 0]
  L[:, 1, 1] = s[:, 1]
  L[:, 2, 2] = s[:, 2]

  L = R @ L
  return L


def safe_state(silent):
  old_f = sys.stdout

  class F:

    def __init__(self, silent):
      self.silent = silent

    def write(self, x):
      if not self.silent:
        if x.endswith("\n"):
          old_f.write(
              x.replace(
                  "\n", " [{}]\n".format(
                      str(datetime.now().strftime("%d/%m %H:%M:%S")))))
        else:
          old_f.write(x)

    def flush(self):
      old_f.flush()

  sys.stdout = F(silent)

  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.set_device(torch.device("cuda:0"))


def knn(points: np.ndarray, num_knn: int):
  tree = KDTree(points)
  distances, indices = tree.query(points, k=num_knn + 1)
  distances = distances[:, 1:]
  indices = indices[:, 1:]
  return distances, indices


def quat_mult(q1, q2):
  # w1, x1, y1, z1 = q1.T
  # w2, x2, y2, z2 = q2.T
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  return torch.stack([w, x, y, z]).T


def weighted_l2_loss_v1(x, y, w):
  return torch.sqrt(((x - y)**2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
  return torch.sqrt(((x - y)**2).sum(-1) * w + 1e-20).mean()


def weighted_l1_loss_v2(x, y, w):
  return torch.abs((x - y).sum(-1) * w + 1e-20).mean()


def mat2quat(m):
  t = torch.trace(m)
  if t > 0:
    s = torch.sqrt(1.0 + t) * 2
    w = 0.25 * s
    x = (m[2, 1] - m[1, 2]) / s
    y = (m[0, 2] - m[2, 0]) / s
    z = (m[1, 0] - m[0, 1]) / s
  elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
    s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
    w = (m[2, 1] - m[1, 2]) / s
    x = 0.25 * s
    y = (m[0, 1] + m[1, 0]) / s
    z = (m[0, 2] + m[2, 0]) / s
  elif m[1, 1] > m[2, 2]:
    s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
    w = (m[0, 2] - m[2, 0]) / s
    x = (m[0, 1] + m[1, 0]) / s
    y = 0.25 * s
    z = (m[1, 2] + m[2, 1]) / s
  else:
    s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    w = (m[1, 0] - m[0, 1]) / s
    x = (m[0, 2] + m[2, 0]) / s
    y = (m[1, 2] + m[2, 1]) / s
    z = 0.25 * s
  return torch.tensor([w, x, y, z])


def mat2quat_batch(m):
  t = torch.sum(torch.diagonal(m, dim1=-2, dim2=-1), dim=-1)

  w = torch.zeros(m.shape[0], device=m.device)
  x = torch.zeros(m.shape[0], device=m.device)
  y = torch.zeros(m.shape[0], device=m.device)
  z = torch.zeros(m.shape[0], device=m.device)

  mask = t > 0
  s = torch.sqrt(1.0 + t) * 2
  w[mask] = 0.25 * s[mask]
  x[mask] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
  y[mask] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
  z[mask] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

  mask_1 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
  s_1 = torch.sqrt(1.0 + m[mask_1, 0, 0] - m[mask_1, 1, 1] -
                   m[mask_1, 2, 2]) * 2
  w[mask_1] = (m[mask_1, 2, 1] - m[mask_1, 1, 2]) / s_1
  x[mask_1] = 0.25 * s_1
  y[mask_1] = (m[mask_1, 0, 1] + m[mask_1, 1, 0]) / s_1
  z[mask_1] = (m[mask_1, 0, 2] + m[mask_1, 2, 0]) / s_1

  mask_2 = m[:, 1, 1] > m[:, 2, 2]
  s_2 = torch.sqrt(1.0 + m[mask_2, 1, 1] - m[mask_2, 0, 0] -
                   m[mask_2, 2, 2]) * 2
  w[mask_2] = (m[mask_2, 0, 2] - m[mask_2, 2, 0]) / s_2
  x[mask_2] = (m[mask_2, 0, 1] + m[mask_2, 1, 0]) / s_2
  y[mask_2] = 0.25 * s_2
  z[mask_2] = (m[mask_2, 1, 2] + m[mask_2, 2, 1]) / s_2

  mask_3 = ~mask & ~mask_1 & ~mask_2
  s_3 = torch.sqrt(1.0 + m[mask_3, 2, 2] - m[mask_3, 0, 0] -
                   m[mask_3, 1, 1]) * 2
  w[mask_3] = (m[mask_3, 1, 0] - m[mask_3, 0, 1]) / s_3
  x[mask_3] = (m[mask_3, 0, 2] + m[mask_3, 2, 0]) / s_3
  y[mask_3] = (m[mask_3, 1, 2] + m[mask_3, 2, 1]) / s_3
  z[mask_3] = 0.25 * s_3

  return torch.stack([w, x, y, z], dim=1)


def value_to_rgb(values, cmap_name='viridis'):
  """
    Remap value\in[0,1] to RGB colors.

    Parameters:
    values: Tensor of shape (N,), values in the range [0,1].
    cmap_name: String, optional, name of the Matplotlib colormap.

    Returns:
    Tensor of shape (N, 3), RGB colors in the range [0,1].  
    """
  cmap = plt.get_cmap(cmap_name)
  values = values.detach().cpu().numpy()  # Convert to NumPy for Matplotlib
  colors = cmap(values)[:, :3]  # Get RGB values (ignore alpha channel)
  return torch.tensor(colors, dtype=torch.float32)


def kl_divergence_gaussian(mu0, sigma0, mu1, sigma1):
  """
    Compute KL divergence between two multivariate normal distributions:
    P = N(mu0, sigma0)
    Q = N(mu1, sigma1)
    """
  k = mu0.shape[0]
  sigma1_inv = torch.linalg.inv(sigma1)
  diff = mu1 - mu0

  trace_term = torch.trace(sigma1_inv @ sigma0)
  diff_term = diff @ sigma1_inv @ diff
  log_det_term = torch.logdet(sigma1) - torch.logdet(sigma0)

  kl = 0.5 * (log_det_term - k + trace_term + diff_term)
  return kl


def otsu_with_peak_filtering(data,
                             std_multiplier=3,
                             bins=256,
                             bias_factor=1.25):
  """
    Use the main peak for truncation and then apply Otsu's method to calculate the threshold.

    Parameters:
        data (numpy.ndarray): Input data, 1D array.
        std_multiplier (float): Standard deviation multiplier for the main peak range, default is 3.
        bins (int): Number of bins for Otsu's method, default is 256.

    Returns:
        float: Calculated threshold.
    """
  # Standardize
  min_val = np.min(data)
  max_val = np.max(data)
  data = (data - min_val) / (max_val - min_val)

  # Density estimation
  density = gaussian_kde(data)
  x_vals = np.linspace(min(data), max(data), 1000)
  y_vals = density(x_vals)

  # Find the main peak
  peak_idx = np.argmax(y_vals)
  peak_value = x_vals[peak_idx]

  # Set the truncation range around the main peak
  std_dev = np.std(data)
  lower_bound = peak_value - std_multiplier * std_dev
  upper_bound = peak_value + std_multiplier * std_dev
  filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

  # Otsu method to calculate the threshold
  def otsu_threshold_bias_towards_higher(data, bins=256, bias_factor=1.25):
    """
        Modified Otsu method, biased towards higher thresholds.

        Parameters:
            data (numpy.ndarray): Input data, 1D array.
            bins (int): Number of bins for the histogram, default is 256.
            bias_factor (float): Factor to bias towards higher thresholds (>1 favors higher values).

        Returns:
            float: Calculated threshold.
        """
    # Calculate histogram
    hist, bin_edges = np.histogram(data,
                                   bins=bins,
                                   range=(min(data), max(data)),
                                   density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_weight = np.sum(hist)
    total_mean = np.sum(hist * bin_centers)
    max_between_class_variance = 0
    threshold = 0

    weight_bg = 0
    mean_bg = 0

    for i in range(len(hist)):
      weight_bg += hist[i]
      mean_bg += hist[i] * bin_centers[i]

      weight_fg = total_weight - weight_bg
      if weight_bg <= 0 or weight_fg <= 0:
        continue

      mean_fg = (total_mean - mean_bg) / weight_fg
      # between_class_variance = weight_bg * weight_fg * (mean_bg / weight_bg - mean_fg) ** 2
      # Reduce the difference between the two peak sizes
      between_class_variance = weight_bg**.5 * weight_fg**.5 * (
          mean_bg / weight_bg - mean_fg)**2

      # Increase the factor biased towards higher thresholds
      between_class_variance *= bin_centers[i]**(bias_factor - 1)

      if between_class_variance > max_between_class_variance:
        max_between_class_variance = between_class_variance
        threshold = bin_centers[i]

    return threshold

  threshold = otsu_threshold_bias_towards_higher(filtered_data,
                                                 bins,
                                                 bias_factor=bias_factor)
  # Denormalize
  threshold = min_val + (max_val - min_val) * threshold
  return threshold


def get_per_point_cd(gaussians_x, gaussians_y) -> torch.Tensor:
  x = gaussians_x.get_xyz.unsqueeze(0)
  y = gaussians_y.get_xyz.unsqueeze(0)
  dist, _ = chamfer_distance(x,
                             y,
                             batch_reduction=None,
                             point_reduction=None,
                             single_directional=True)
  return dist[0]


def eval_quad(x: torch.tensor, mat: torch.tensor) -> torch.tensor:
  """
    Calculates x^T mat x
    """
  return torch.einsum('pki,kij,pkj->pk', x, mat, x)


def decompose_covariance_matrix(
    cov: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
  eigenvalues, eigenvectors = torch.linalg.eigh(cov)
  rotation_matrices = eigenvectors
  scaling_values = torch.sqrt(
      torch.relu(eigenvalues)
  )  # Use relu to handle potential numerical issues with negative eigenvalues
  return scaling_values, rotation_matrices


def get_rotation_axis(r: np.ndarray,
                      t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  # r = r.T
  eigenvalues, eigenvectors = np.linalg.eig(r)
  axis = None
  for i in range(3):
    if np.isclose(eigenvalues[i].real, 1.0):
      axis = eigenvectors[:, i].real
      break
  if axis is None:
    return None, None

  axis = axis / np.linalg.norm(axis)

  a = np.eye(3) - r
  # point_on_axis = -np.linalg.pinv(r - np.eye(3)) @ t
  point_on_axis = np.linalg.lstsq(a, t, rcond=None)[0]
  return point_on_axis, axis


def rotation_matrix_from_axis_angle(axis: np.array, angle: float):
  """
    Constructs a rotation matrix from a rotation axis and angle.

    Args:
        axis: Unit vector representing the rotation axis.
        angle: Rotation angle in radians.

    Returns:
        R: 3x3 rotation matrix.
    """
  axis = axis / np.linalg.norm(axis)  #ensure axis is a unit vector.
  K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
  I = np.eye(3)
  R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
  return R


def load_mesh(path):
  if path.endswith('.ply'):
    verts, faces = load_ply(path)
  elif path.endswith('.obj'):
    obj = load_obj(path)
    verts = obj[0]
    faces = obj[1].verts_idx
  return verts, faces


def sample_points_from_ply(file_path: str,
                           n_samples: int = 100_000) -> np.ndarray:
  """
    Reads a .ply file and returns the point cloud as a NumPy array.
    Args:
        file_path: The path to the .ply file.
        n_samples: #pts to sample
    Returns:
        A NumPy array of shape (N, 3) containing the 3D point positions, or None if an error occurs.
    """
  verts, faces = load_mesh(file_path)
  mesh = Meshes(verts=[verts], faces=[faces])
  pts = sample_points_from_meshes(mesh, num_samples=n_samples).squeeze(0)
  return pts.cpu().numpy()


def estimate_principal_directions(normals: np.ndarray,
                                  ort: str = 'qr',
                                  k: int = 3) -> np.ndarray:
  """
        Estimate three orthogonal directions from normals, treating opposite directions as equivalent.

        Args:
            normals: np.ndarray of shape (N, 3), unit normals.
            k: int, number of clusters (default 3).
            ort: str, 'qr' for QR decomposition or 'gs' for Gram-Schmidt.

        Returns:
            directions: np.ndarray of shape (3, 3), where each row is a direction.
        """
  normals = normals[~np.isnan(normals).any(axis=1)]

  # Normalize normals
  norms = np.linalg.norm(normals, axis=1, keepdims=True)
  valid_mask = (norms > 1e-8).flatten()

  if np.sum(valid_mask) == 0:
    return np.eye(3)

  normals = normals[valid_mask]
  norms = norms[valid_mask]

  normals = normals / norms

  # Map to first octant
  normals_abs = np.abs(normals)

  # Cluster normals
  kmeans = KMeans(n_clusters=k, random_state=0).fit(normals_abs)
  labels = kmeans.labels_

  # Sort clusters by size
  unique_labels, counts = np.unique(labels, return_counts=True)
  sort_indices = np.argsort(counts)[::-1]
  top_k_labels = unique_labels[sort_indices][:k]

  # Compute mean directions, ordered by cluster size
  directions = np.zeros((k, 3))
  #
  # for i, label in enumerate(top_k_labels):
  #     cluster_normals = normals[labels == label]
  #     if len(cluster_normals) == 0:
  #         raise ValueError(f"Cluster {label} is empty.")
  #     mean_dir = np.mean(cluster_normals, axis=0)
  #     mean_dir = mean_dir / np.linalg.norm(mean_dir)
  #     directions[i] = mean_dir
  #
  for i, label in enumerate(top_k_labels):
    cluster_normals = normals[labels == label]
    if len(cluster_normals) < 1:  # A cluster could be empty
      continue  # Or handle as an error

    # Correct way: Use PCA (SVD) to find the principal direction
    # The principal direction is the eigenvector of the covariance matrix
    # corresponding to the largest eigenvalue.
    covariance_matrix = np.dot(cluster_normals.T, cluster_normals)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # The principal direction is the eigenvector with the largest eigenvalue.
    # np.linalg.eigh sorts eigenvalues in ascending order, so we take the last eigenvector.
    principal_direction = eigenvectors[:, -1]
    directions[i] = principal_direction / np.linalg.norm(principal_direction)

  # Orthogonalize
  if ort == 'gs':

    def gram_schmidt(vectors):
      u = np.zeros_like(vectors)
      u[0] = vectors[0] / np.linalg.norm(vectors[0])
      u[1] = vectors[1] - np.dot(vectors[1], u[0]) * u[0]
      u[1] = u[1] / np.linalg.norm(u[1])
      u[2] = vectors[2] - np.dot(vectors[2], u[0]) * u[0] - np.dot(
          vectors[2], u[1]) * u[1]
      u[2] = u[2] / np.linalg.norm(u[2])
      return u

    directions = gram_schmidt(directions)
  else:  # Default to QR
    directions, _ = np.linalg.qr(directions)

  return directions


def pca_on_pointcloud(pointcloud):
  """
    Perform PCA on a point cloud to derive the three main directions.
    """
  # Center the point cloud by subtracting the mean
  center = np.mean(pointcloud, axis=0)
  centered_points = pointcloud - center

  # Perform PCA
  pca = PCA(n_components=3)
  pca.fit(centered_points)

  # Extract directions (principal components) and variances
  directions = pca.components_.T  # Transpose to get (3, 3) with each row as a direction
  stds = np.sqrt(pca.explained_variance_)
  return center, directions, stds


def calculate_obb_o3d(
    pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Convert numpy array to open3d PointCloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pointcloud)

  # Compute OBB
  obb = pcd.get_oriented_bounding_box()

  # Extract center
  center = np.array(obb.get_center())

  # Extract rotation matrix (directions) and ensure it's (3, 3)
  R = obb.R  # Rotation matrix from Open3D OBB
  directions = R.T  # Transpose to match convention where rows are directions

  # Extract extents and sort in descending order
  extents = np.array(obb.extent)
  sort_indices = np.argsort(extents)[::-1]  # Sort indices in descending order
  extents = extents[sort_indices]
  directions = directions[sort_indices]
  return center, directions, extents


def calculate_obb_pv(
    pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
    Calculate the Oriented Bounding Box (OBB) of a point cloud using PyVista.
    """
  # Create a PyVista PolyData from your point cloud
  point_cloud = pv.PolyData(pointcloud)

  # Get oriented bounding box with metadata
  obb, corner, axes = point_cloud.oriented_bounding_box(return_meta=True)

  # 'axes' is a 3x3 matrix where each row is a direction vector of an axis
  directions = axes

  # Compute extents by projecting points onto axes
  proj = (pointcloud - corner) @ directions.T
  extents = proj.max(axis=0) - proj.min(axis=0)

  # Sort extents and directions by descending extents
  sort_idx = np.argsort(extents)[::-1]
  extents = extents[sort_idx]
  directions = directions[sort_idx]

  # Compute center as midpoint along each axis
  center_proj = (proj.max(axis=0) + proj.min(axis=0)) / 2
  center = corner + center_proj @ directions
  return center, directions, extents


def get_oriented_aabb(point_cloud, directions):
  """
    Calculates the Axis-Aligned Bounding Box of a point cloud with respect to a new set of axes.

    Args:
        point_cloud (np.ndarray): The (N, 3) point cloud.
        directions (np.ndarray): A (3, 3) array where each row is a direction vector.
                                 These vectors should ideally be orthonormal (unit vectors and mutually perpendicular).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The min corner of the AABB in the new coordinate system.
            - np.ndarray: The max corner of the AABB in the new coordinate system.
    """
  # Ensure directions are unit vectors for accurate projection
  normalized_directions = directions / np.linalg.norm(directions,
                                                      axis=1)[:, np.newaxis]

  # Project the point cloud onto the new axes
  # The result is an (N, 3) array where each column corresponds to projections on one of the direction vectors
  projected_points = point_cloud @ normalized_directions.T

  # Find the minimum and maximum projections along each new axis
  min_projections = np.min(projected_points, axis=0)
  max_projections = np.max(projected_points, axis=0)

  return min_projections, max_projections


def get_bounding_box(point_cloud: np.ndarray,
                     directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
    Calculates the Oriented Bounding Box (OBB) for a point cloud given specific axes.

    Args:
        point_cloud (np.ndarray): The (N, 3) point cloud data.
        directions (np.ndarray): A (3, 3) array where each row is a direction vector for the OBB's axes.
                                 These directions should be orthonormal.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The (3,) center of the OBB in world coordinates.
            - np.ndarray: The (3,) extents (half-lengths) of the OBB along each direction.
    """
  # 1. Project the point cloud onto the given directions
  # The result is an (N, 3) array of the points' coordinates in the new axis system
  projected_points = point_cloud @ directions.T

  # 2. Find the minimum and maximum projections along each new axis
  min_projections = np.min(projected_points, axis=0)
  max_projections = np.max(projected_points, axis=0)

  # 3. Calculate the center and extents in the local coordinate system
  local_center = (min_projections + max_projections) / 2.0
  extents = (max_projections - min_projections) / 2.0

  # 4. Transform the local center back to world coordinates
  # This is a linear combination of the direction vectors
  world_center = local_center @ directions
  return world_center, extents


def estimate_normals_o3d(points, radius_multiplier=3, max_nn=30):
  """
    Estimate normals for a point cloud.
    """
  # Create Open3D point cloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  # Analyze density to find a suitable radius
  # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  # avg_dist_to_3rd_neighbor = 0
  # num_points = len(pcd.points)
  # for i in range(num_points):
  #     [_, _, dists] = pcd_tree.search_knn_vector_3d(pcd.points[i], 4)
  #     if len(dists) > 3:
  #         avg_dist_to_3rd_neighbor += dists[3]
  # avg_dist_to_3rd_neighbor /= num_points
  # radius = avg_dist_to_3rd_neighbor * radius_multiplier

  # Estimate normals
  pcd.estimate_normals(
      # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

  # Extract normals
  normals = np.asarray(pcd.normals)
  return normals


def estimate_normals_ransac_o3d(points,
                                radius=0.01,
                                max_nn=30,
                                distance_threshold=0.001,
                                ransac_n=3,
                                num_iterations=100,
                                min_inliers=10):
  """
    Estimate normals for a point cloud using RANSAC-based local plane fitting.

    Parameters:
    - points: np.ndarray of shape (N, 3) - The input point cloud.
    - radius: float - Radius for neighbor search.
    - max_nn: int - Maximum number of nearest neighbors.
    - distance_threshold: float - Distance threshold for inliers in RANSAC.
    - ransac_n: int - Number of points to sample for plane fitting (default 3 for plane).
    - num_iterations: int - Number of RANSAC iterations per local fit.
    - min_inliers: int - Minimum number of inliers required for a valid fit.

    Returns:
    - normals: np.ndarray of shape (N, 3) - Estimated normals.
    """
  # Create Open3D point cloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  # Build KDTree for efficient neighbor search
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)

  # Initialize normals array
  N = len(points)
  normals = np.zeros((N, 3))

  for i in tqdm(range(N)):
    # Find neighbors
    [_, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[i],
                                                   radius=radius,
                                                   max_nn=max_nn)

    if len(idx) < ransac_n:
      # Not enough points, skip or set to NaN/zero
      continue

    # Create sub-point cloud from neighbors
    sub_pcd = pcd.select_by_index(idx)

    # Fit plane using RANSAC
    plane_model, inliers = sub_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)

    if len(inliers) >= min_inliers:
      # Extract normal from plane model [a, b, c, d] and normalize
      normal = plane_model[:3]
      normal /= np.linalg.norm(normal)
      normals[i] = normal
    # Else, leave as zero or handle differently

  return normals


def estimate_normals_pv(points, feature_angle=30.0, splitting=True):
  """
    Estimate normals using PyVista (VTK's implementation).

    Args:
        points: (N, 3) numpy array.
        feature_angle: Angle to define sharp edges. Normals are not smoothed
                       across edges sharper than this angle.
        splitting: If True, splits vertices at sharp edges to produce crisp
                   normals on each face.
    """
  # Create a PyVista PolyData object
  poly_data = pv.PolyData(points)

  # Compute normals
  poly_data.compute_normals(feature_angle=feature_angle,
                            splitting=splitting,
                            inplace=True)
  return poly_data.point_normals


def get_source_path(cfg_path: str) -> str:
  with open(cfg_path, 'r') as f:
    content = f.read()
  # Split at 'source_path=' and take the second part
  path_part = content.split('source_path=')[1]
  # Split at next comma and take first part, remove quotes and whitespace
  source_path = path_part.split(',')[0].strip("'").strip()
  return source_path


def find_files_with_suffix(directory, suffix):
  if not os.path.exists(directory):
    return []
  matching_files = []
  for filename in os.listdir(directory):
    if filename.endswith(suffix):
      matching_files.append(filename)
  return matching_files


def find_close(x: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
  """
    Finds indices of points in y close to points in x using SciPy's KDTree.
    Args:
        x: A tensor of shape (N, 3) representing the first point cloud.
        y: A tensor of shape (M, 3) representing the second point cloud.
        threshold: The maximum distance for a point in y to be considered "close".
    Returns:
        A tensor containing the indices of points in y that satisfy the
        closeness condition.
    """
  kdtree = KDTree(x)

  # Query the KDTree for each point in y
  # k=1 finds the nearest neighbor
  # distance_upper_bound applies the threshold efficiently
  distances, indices = kdtree.query(y, k=1, distance_upper_bound=threshold)

  # SciPy's query with distance_upper_bound returns inf for distances
  # and N (or len(x)) for indices where no neighbor is found within the threshold.
  # We need to filter these out.
  # Valid indices are those that are NOT equal to N (or x.shape[0])
  valid_mask = indices != x.shape[0]

  # The indices returned by kdtree.query are indices in y_np corresponding
  # to the points that found a neighbor within the threshold in x.
  # We can get these indices directly from the boolean mask's 'where'.
  close_indices = np.where(valid_mask)[0]
  return close_indices


def get_combined_aabb(
    aabb1_min: torch.tensor, aabb1_max: torch.tensor, aabb2_min: torch.tensor,
    aabb2_max: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
  # Find the element-wise minimum of the two min corners
  combined_min = torch.min(aabb1_min, aabb2_min)
  # Find the element-wise maximum of the two max corners
  combined_max = torch.max(aabb1_max, aabb2_max)
  return combined_min, combined_max


def get_extended_aabb(aabb_min_ext: torch.tensor, aabb_max_ext: torch.tensor,
                      aabb_min: torch.tensor, aabb_max: torch.tensor,
                      axes: torch.tensor,
                      t: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
  t_proj = t @ axes.T
  aabb_min_shift, aabb_max_shift = aabb_min + t_proj, aabb_max + t_proj
  # return get_combined_aabb(aabb_min_ext, aabb_max_ext, aabb_min_shift, aabb_max_shift)
  return aabb_min_shift, aabb_max_shift


def shift_aabb_from_collision(
    aabb1_ext: tuple[torch.tensor, torch.tensor],  # (min, max)
    aabb2_ext: tuple[torch.tensor, torch.tensor],  # (min, max)
    axes: torch.tensor,  # (3, 3)
    t1: torch.tensor,
    t2: torch.tensor,
) -> tuple[bool, int]:

  def shift_aabb_from_collision_helper(
      bb1: tuple[torch.tensor, torch.tensor],  # (min, max)
      bb2: tuple[torch.tensor, torch.tensor],  # (min, max)
      bases: torch.tensor,
      t1_: torch.tensor,
      t2_: torch.tensor) -> tuple[bool, int]:
    has_intersection, min_axis_idx = find_minimum_penetration_axis(
        bb1[0], bb1[1], bb2[0], bb2[1])
    if not has_intersection:
      return False, -1
    return True, min_axis_idx

  #end function

  return shift_aabb_from_collision_helper(aabb1_ext, aabb2_ext, axes, t2, t1)


def find_minimum_penetration_axis(aabb1_min: torch.Tensor,
                                  aabb1_max: torch.Tensor,
                                  aabb2_min: torch.Tensor,
                                  aabb2_max: torch.Tensor,
                                  eps: float = 1e-4) -> tuple[bool, int]:
  """
    Judges if two AABBs intersect and returns the axis of minimum penetration.

    Args:
        aabb1_min (torch.Tensor): The min corner (x, y, z) of the first AABB.
        aabb1_max (torch.Tensor): The max corner (x, y, z) of the first AABB.
        aabb2_min (torch.Tensor): The min corner (x, y, z) of the second AABB.
        aabb2_max (torch.Tensor): The max corner (x, y, z) of the second AABB.
        eps (float): collision threshold

    Returns:
        tuple[bool, int]: A tuple containing:
            - bool: True if the AABBs intersect, False otherwise.
            - int: The index of the axis (0, 1, or 2) with the smallest overlap.
                   Returns -1 if there is no intersection.
    """
  # 1. Check for intersection
  # No intersection if there's a gap on any one axis
  if torch.any(aabb1_max < aabb2_min - eps) or torch.any(aabb2_max < aabb1_min +
                                                         eps):
    return False, -1

  # 2. Calculate penetration depths for each axis
  penetrations = torch.min(aabb1_max, aabb2_max) - torch.max(
      aabb1_min, aabb2_min)

  # 3. Find the axis with the minimum penetration and return its index
  min_axis_idx = torch.argmin(penetrations).item()
  return True, min_axis_idx


def check_and_resolve_aabb_collision(
    aabb1_min: torch.Tensor,
    aabb1_max: torch.Tensor,
    aabb2_min: torch.Tensor,
    aabb2_max: torch.Tensor,
    eps: float = 1e-4
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
    Checks for AABB intersection and resolves it by resizing each box along one edge.

    Args:
        aabb1_min/max: The min/max corners of the first AABB.
        aabb2_min/max: The min/max corners of the second AABB.

    Returns:
        A tuple containing:
        - bool: True if an intersection was found and resolved.
        - torch.Tensor: Shift vector for aabb1_min.
        - torch.Tensor: Shift vector for aabb1_max.
        - torch.Tensor: Shift vector for aabb2_min.
        - torch.Tensor: Shift vector for aabb2_max.
    """

  def resolve_aabb_collision_by_resizing(
      bb1_min: torch.Tensor, bb1_max: torch.Tensor, bb2_min: torch.Tensor,
      bb2_max: torch.Tensor
  ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    intersected, min_axis_idx = find_minimum_penetration_axis(bb1_min,
                                                              bb1_max,
                                                              bb2_min,
                                                              bb2_max,
                                                              eps=0)

    device = bb1_min.device
    zero_vec = torch.zeros(3, device=device, dtype=bb1_min.dtype)

    if not intersected:
      return False, zero_vec, zero_vec, zero_vec, zero_vec

    # 1. Calculate penetration depth and the amount each box needs to be resized
    penetration_depth = (torch.min(bb1_max, bb2_max) -
                         torch.max(bb1_min, bb2_min))[min_axis_idx]
    shift_amount = penetration_depth / 2.0

    # 2. Determine which faces are intersecting based on center positions
    center1 = (bb1_min + bb1_max) / 2.0
    center2 = (bb2_min + bb2_max) / 2.0

    # 3. Initialize shift vectors for all four corners
    shift_1_min, shift_1_max = zero_vec.clone(), zero_vec.clone()
    shift_2_min, shift_2_max = zero_vec.clone(), zero_vec.clone()

    if center1[min_axis_idx] < center2[min_axis_idx]:
      # AABB1 is "left" of AABB2. Its right face (max) must move left.
      shift_1_max[min_axis_idx] = -shift_amount
      # AABB2 is "right" of AABB1. Its left face (min) must move right.
      shift_2_min[min_axis_idx] = shift_amount
    else:
      # AABB1 is "right" of AABB2. Its left face (min) must move right.
      shift_1_min[min_axis_idx] = shift_amount
      # AABB2 is "left" of AABB1. Its right face (max) must move left.
      shift_2_max[min_axis_idx] = -shift_amount

    return True, shift_1_min, shift_1_max, shift_2_min, shift_2_max

  aabb1_min_exp, aabb2_min_exp = aabb1_min - eps, aabb2_min - eps
  aabb1_max_exp, aabb2_max_exp = aabb1_max + eps, aabb2_max + eps
  return resolve_aabb_collision_by_resizing(aabb1_min_exp, aabb1_max_exp,
                                            aabb2_min_exp, aabb2_max_exp)


def get_bb_collision_axis(center1: np.ndarray, extents1: np.ndarray,
                          directions1: np.ndarray, center2: np.ndarray,
                          extents2: np.ndarray,
                          directions2: np.ndarray) -> tuple[bool, int, int]:
  """
    Checks if two OBBs intersect and, if so, finds the index of the axis
    from BB1 that provides the minimum separation distance.

    Args:
        center1, extents1, directions1: Properties of the first bounding box (BB1).
        center2, extents2, directions2: Properties of the second bounding box (BB2).

    Returns:
        A tuple containing:
        - bool: True if the boxes intersect, False otherwise.
        - int: The index (0, 1, or 2) of the best separation axis from BB1,
               or -1 if there is no intersection.
    """
  T = center2 - center1

  axes_to_test = [
      directions1[0],
      directions1[1],
      directions1[2],
      directions2[0],
      directions2[1],
      directions2[2],
      np.cross(directions1[0], directions2[0]),
      np.cross(directions1[0], directions2[1]),
      np.cross(directions1[0], directions2[2]),
      np.cross(directions1[1], directions2[0]),
      np.cross(directions1[1], directions2[1]),
      np.cross(directions1[1], directions2[2]),
      np.cross(directions1[2], directions2[0]),
      np.cross(directions1[2], directions2[1]),
      np.cross(directions1[2], directions2[2]),
  ]

  for L in axes_to_test:
    # Check if the axis is a zero vector (more robustly than L==0)
    if np.dot(L, L) < 1e-8:
      continue
    r1 = np.sum(extents1 * np.abs(directions1 @ L))
    r2 = np.sum(extents2 * np.abs(directions2 @ L))
    center_dist_proj = np.abs(T @ L)

    if center_dist_proj > r1 + r2:
      return False, -1, -1  # No intersection

  def _find_min_penetration_axis(axes_to_check: np.ndarray) -> int:
    """Finds the axis with the minimum penetration from a given set."""
    min_pen, best_idx = float('inf'), -1
    for i, ll in enumerate(axes_to_check):
      # Project radii of both boxes onto the axis ll
      _r1 = np.sum(extents1 * np.abs(directions1 @ ll))
      _r2 = np.sum(extents2 * np.abs(directions2 @ ll))
      # Calculate penetration depth
      penetration = (_r1 + _r2) - np.abs(T @ ll)
      if penetration < min_pen:
        min_pen = penetration
        best_idx = i
    return best_idx

  best_idx1 = _find_min_penetration_axis(directions1)
  best_idx2 = _find_min_penetration_axis(directions2)
  return True, best_idx1, best_idx2


def get_bb_collision_axis_torch(
    center1: torch.Tensor,
    extents1: torch.Tensor,
    directions1: torch.Tensor,
    center2: torch.Tensor,
    extents2: torch.Tensor,
    directions2: torch.Tensor,
) -> tuple[bool, int, int]:
  T = center2 - center1

  axes_to_test = [
      directions1[0],
      directions1[1],
      directions1[2],
      directions2[0],
      directions2[1],
      directions2[2],
      torch.cross(directions1[0], directions2[0]),
      torch.cross(directions1[0], directions2[1]),
      torch.cross(directions1[0], directions2[2]),
      torch.cross(directions1[1], directions2[0]),
      torch.cross(directions1[1], directions2[1]),
      torch.cross(directions1[1], directions2[2]),
      torch.cross(directions1[2], directions2[0]),
      torch.cross(directions1[2], directions2[1]),
      torch.cross(directions1[2], directions2[2]),
  ]

  for L in axes_to_test:
    # Check if the axis is a zero vector (more robustly than L==0)
    if L @ L < 1e-8:
      continue
    r1 = torch.sum(extents1 * torch.abs(directions1 @ L))
    r2 = torch.sum(extents2 * torch.abs(directions2 @ L))
    center_dist_proj = torch.abs(T @ L)

    if center_dist_proj > r1 + r2:
      return False, -1, -1  # No intersection

  def _find_min_penetration_axis(axes_to_check: torch.tensor) -> int:
    """Finds the axis with the minimum penetration from a given set."""
    min_pen, best_idx = float('inf'), -1
    for i, ll in enumerate(axes_to_check):
      # Project radii of both boxes onto the axis ll
      _r1 = torch.sum(extents1 * torch.abs(directions1 @ ll))
      _r2 = torch.sum(extents2 * torch.abs(directions2 @ ll))
      # Calculate penetration depth
      penetration = (_r1 + _r2) - torch.abs(T @ ll)
      if penetration < min_pen:
        min_pen = penetration
        best_idx = i
    return best_idx

  best_idx1 = _find_min_penetration_axis(directions1)
  best_idx2 = _find_min_penetration_axis(directions2)
  return True, best_idx1, best_idx2


def find_and_unify_orthogonal(matrices: np.ndarray,
                              threshold: float = 3) -> np.ndarray:
  n_matrices = matrices.shape[0]

  distances = np.zeros((n_matrices, n_matrices))
  for i in np.arange(n_matrices):
    for j in np.arange(i, n_matrices):
      dots = np.clip(matrices[i] @ matrices[j].T, -1.0, 1.0)
      angles_mean = np.mean(np.rad2deg(np.arccos(np.abs(dots).max(axis=1))))
      distances[i, j] = angles_mean
      distances[j, i] = angles_mean
  print((distances < threshold).astype('int'))

  neighbor_counts = np.sum(distances < threshold, axis=1)
  if np.all(neighbor_counts <= 1):  # Handle case with no clusters
    return distances < threshold

  ref_idx = np.argmax(neighbor_counts)
  q_ref = matrices[ref_idx]

  # The cluster includes all matrices close to the reference matrix
  cluster_indices = np.where(distances[ref_idx] < threshold)[0]
  cluster_matrices = matrices[cluster_indices]

  normals = cluster_matrices.reshape(-1, 3)
  unified = estimate_principal_directions(normals, ort='gs')
  matrices[cluster_indices] = unified
  print("unified:", unified)
  print()
  return distances < threshold


def rodrigues_rotation_matrix(u, theta):
  """Rodrigues 旋转矩阵，绕单位向量 u 旋转 theta"""

  def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

  u = normalize(u)
  K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
  I = np.eye(3)
  return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_axis_from_matrix_batch(rotation: torch.Tensor) -> torch.Tensor:
  """
    Extract the rotation axis from a batch of rotation matrices via the
    skew-symmetric part (torch, batched).

    Args:
        rotation (torch.Tensor): (..., 3, 3) rotation matrices.

    Returns:
        torch.Tensor: (..., 3) unit rotation axes. Falls back to [1, 0, 0]
            where the rotation is (near-)identity and the axis is undefined.
    """
  axis = torch.stack([
      rotation[..., 2, 1] - rotation[..., 1, 2],
      rotation[..., 0, 2] - rotation[..., 2, 0],
      rotation[..., 1, 0] - rotation[..., 0, 1],
  ],
                     dim=-1)
  norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
  safe_axis = axis / norm.clamp_min(1e-6)
  default_axis = torch.zeros_like(safe_axis)
  default_axis[..., 0] = 1.0
  return torch.where(norm > 1e-6, safe_axis, default_axis)


def rotation_axis_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
  """
    Extract the rotation axis from a single rotation matrix (torch).

    Args:
        rotation (torch.Tensor): (3, 3) rotation matrix.

    Returns:
        torch.Tensor: (3,) unit rotation axis (see
            rotation_axis_from_matrix_batch for the identity fallback).
    """
  return rotation_axis_from_matrix_batch(rotation.unsqueeze(0))[0]


def axis_angle_to_matrix_torch(axis: torch.Tensor,
                               angle: torch.Tensor) -> torch.Tensor:
  """
    Build a rotation matrix from an axis and angle via Rodrigues' formula
    (torch port of rodrigues_rotation_matrix / rotation_matrix_from_axis_angle).

    Args:
        axis (torch.Tensor): (3,) rotation axis (normalized internally).
        angle (torch.Tensor): scalar rotation angle in radians.

    Returns:
        torch.Tensor: (3, 3) rotation matrix
            R = I + sin(angle) * K + (1 - cos(angle)) * K @ K.
    """
  axis = F.normalize(axis, dim=0)
  sin_a = torch.sin(angle)
  cos_a = torch.cos(angle)
  eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
  skew = torch.zeros((3, 3), device=axis.device, dtype=axis.dtype)
  skew[0, 1] = -axis[2]
  skew[0, 2] = axis[1]
  skew[1, 0] = axis[2]
  skew[1, 2] = -axis[0]
  skew[2, 0] = -axis[1]
  skew[2, 1] = axis[0]
  return eye + sin_a * skew + (1 - cos_a) * (skew @ skew)


def approximate_obb_intersection_volume(center_a: torch.Tensor,
                                        extents_a: torch.Tensor,
                                        axes_a: torch.Tensor,
                                        center_b: torch.Tensor,
                                        extents_b: torch.Tensor,
                                        axes_b: torch.Tensor,
                                        resolution: int) -> float:
  """
    Approximate the intersection volume of two oriented bounding boxes (OBBs)
    by Monte-Carlo-style grid sampling inside box A and testing membership in
    box B.

    Args:
        center_a (torch.Tensor): (3,) center of box A.
        extents_a (torch.Tensor): (3,) half-extents of box A.
        axes_a (torch.Tensor): (3, 3) orthonormal axes of box A (rows).
        center_b (torch.Tensor): (3,) center of box B.
        extents_b (torch.Tensor): (3,) half-extents of box B.
        axes_b (torch.Tensor): (3, 3) orthonormal axes of box B (rows).
        resolution (int): grid samples per axis inside box A (>= 1).

    Returns:
        float: approximate overlap volume in world units.
    """
  resolution = max(int(resolution), 1)
  with torch.no_grad():
    device = center_a.device
    ext_a = extents_a.to(device)
    axes_a_mat = axes_a.to(device)
    axes_b_mat = axes_b.to(device)
    ext_b = extents_b.to(device)

    step = (2.0 * ext_a / resolution).clamp_min(1e-6)
    offsets = [
        torch.linspace(-ext_a[i] + 0.5 * step[i],
                       ext_a[i] - 0.5 * step[i],
                       resolution,
                       device=device) for i in range(3)
    ]
    mesh = torch.meshgrid(offsets[0], offsets[1], offsets[2])
    grid = torch.stack(mesh, dim=-1).reshape(-1, 3)
    points_world = torch.matmul(grid, axes_a_mat) + center_a

    rel_b = torch.matmul(points_world - center_b, axes_b_mat.T)
    inside_b = torch.all(torch.abs(rel_b) <= (ext_b + 1e-6), dim=-1)

    cell_volume = torch.prod(step).item()
    return float(inside_b.sum().item() * cell_volume)
