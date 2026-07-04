import numpy as np
import os

import torch
import open3d as o3d

from sklearn.cluster import KMeans


def estimate_principal_directions(normals: np.ndarray,
                                  ort: str = "qr",
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
  if ort == "gs":

    def gram_schmidt(vectors):
      u = np.zeros_like(vectors)
      u[0] = vectors[0] / np.linalg.norm(vectors[0])
      u[1] = vectors[1] - np.dot(vectors[1], u[0]) * u[0]
      u[1] = u[1] / np.linalg.norm(u[1])
      u[2] = (vectors[2] - np.dot(vectors[2], u[0]) * u[0] -
              np.dot(vectors[2], u[1]) * u[1])
      u[2] = u[2] / np.linalg.norm(u[2])
      return u

    directions = gram_schmidt(directions)
  else:  # Default to QR
    directions, _ = np.linalg.qr(directions)

  return directions


def find_and_unify_orthogonal(matrices: np.ndarray,
                              threshold: float = 3) -> np.ndarray:
  n_matrices = matrices.shape[0]

  distances = np.zeros((n_matrices, n_matrices))
  for i in np.arange(n_matrices):
    for j in np.arange(i, n_matrices):
      dots = np.clip(matrices[i] @ matrices[j].T, -1.0, 1.0)
      angles_mean = np.mean(np.rad2deg(np.acos(np.abs(dots).max(axis=1))))
      distances[i, j] = angles_mean
      distances[j, i] = angles_mean
  print((distances < threshold).astype("int"))

  neighbor_counts = np.sum(distances < threshold, axis=1)
  if np.all(neighbor_counts <= 1):  # Handle case with no clusters
    return

  ref_idx = np.argmax(neighbor_counts)
  q_ref = matrices[ref_idx]

  # The cluster includes all matrices close to the reference matrix
  cluster_indices = np.where(distances[ref_idx] < threshold)[0]
  cluster_matrices = matrices[cluster_indices]

  normals = cluster_matrices.reshape(-1, 3)
  unified = estimate_principal_directions(normals, ort="gs")
  matrices[cluster_indices] = unified
  print(unified)
  print()
  return distances < threshold


def rotation_matrix_from_axis_angle(axis: np.array, angle: float):
  """
    Constructs a rotation matrix from a rotation axis and angle.

    Args:
        axis: Unit vector representing the rotation axis.
        angle: Rotation angle in radians.

    Returns:
        R: 3x3 rotation matrix.
    """
  axis = axis / np.linalg.norm(axis)  # ensure axis is a unit vector.
  K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
  I = np.eye(3)
  R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
  return R


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


import numpy as np
import open3d as o3d
from typing import Optional


def _needs_normals(pcd: o3d.geometry.PointCloud) -> bool:
  if not pcd.has_normals():
    return True
  n = np.asarray(pcd.normals)
  p = np.asarray(pcd.points)
  if n.shape[0] != p.shape[0]:
    return True
  if not np.isfinite(n).all():
    return True
  # If the average norm is too small (close to zero), consider it invalid
  if float(np.linalg.norm(n, axis=1).mean()) < 1e-6:
    return True
  return False


def _auto_radius(pcd: o3d.geometry.PointCloud,
                 k: int = 16,
                 samples: int = 512) -> float:
  """Adaptively estimate the normal radius based on local density: median distance to the k-th nearest neighbor * 2.5."""
  n = len(pcd.points)
  if n == 0:
    return 1e-3
  if n <= k + 1:
    # if the point cloud is too small, use a fraction of its bounding box extent
    extent = np.linalg.norm(
        np.asarray(pcd.get_axis_aligned_bounding_box().get_extent()))
    return max(extent * 0.02, 1e-6)
  idx = np.random.choice(n, size=min(samples, n), replace=False)
  kdt = o3d.geometry.KDTreeFlann(pcd)
  dks = []
  for i in idx:
    # Find the distance to the k-th nearest neighbor (excluding the point itself)
    _, _, d2 = kdt.search_knn_vector_3d(pcd.points[i], k + 1)
    dks.append(np.sqrt(d2[-1]))
  return float(np.median(dks) * 2.5)


def ensure_normals(
    pcd: o3d.geometry.PointCloud,
    radius: Optional[float] = None,
    max_nn: int = 30,
    orient: str = "consistent",  # "consistent" | "camera" | "align"
    camera_location: np.ndarray = np.array([0.0, 0.0, 10.0]),
    k_consistent: int = 30,
) -> o3d.geometry.PointCloud:
  """
    If the point cloud has no valid normals, estimate normals and orient them.
    - orient="consistent": Use consistent tangent plane, suitable for most geometric reconstruction/registration (recommended)
    - orient="camera": Orient towards camera location
    - orient="align": Align with a given direction (Z-axis)
    """
  if not _needs_normals(pcd):
    return pcd

  # Estimate normals
  if radius is None:
    radius = _auto_radius(pcd, k=min(max_nn, 16))
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
      radius=radius, max_nn=max_nn))

  # Orient normals
  try:
    if orient == "consistent":
      # Ensure consistent normal orientation (k typically 20~50)

      pcd.orient_normals_consistent_tangent_plane(k_consistent)
    elif orient == "camera":
      pcd.orient_normals_towards_camera_location(camera_location)
    elif orient == "align":
      pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
  except Exception:
    # In case of small point clouds or degenerate cases, normal orientation may fail. Ignore.
    pass

  return pcd
