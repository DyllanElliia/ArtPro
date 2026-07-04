import os
import torch
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
# from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses, eval_img_loss, eval_opacity_bce_loss, eval_depth_loss
from scene import BWScenes
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms, fetchPly
from utils.general_utils import get_per_point_cd, otsu_with_peak_filtering, inverse_sigmoid, \
    decompose_covariance_matrix, rotation_matrix_from_axis_angle, eval_quad, knn, mat2quat, quat_mult, \
    get_rotation_axis, rodrigues_rotation_matrix
from utils.graphics_utils import getProjectionMatrix, getWorld2View
try:
  from torch.utils.tensorboard import SummaryWriter
  TENSORBOARD_FOUND = True
except ImportError:
  TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
  from argparse import ArgumentParser, Namespace
  from arguments import ModelParams, PipelineParams, OptimizationParams
  if not args.model_path:
    if os.getenv('OAR_JOB_ID'):
      unique_str = os.getenv('OAR_JOB_ID')
    else:
      unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/", unique_str[0:10])

  # Set up output folder
  print("Output folder: {}".format(args.model_path))
  os.makedirs(args.model_path, exist_ok=True)
  with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
    cfg_log_f.write(str(Namespace(**vars(args))))

  # Create Tensorboard writer
  tb_writer = None
  if TENSORBOARD_FOUND:
    tb_writer = SummaryWriter(args.model_path)
  else:
    print("Tensorboard not available: not logging progress")
  return tb_writer


def train_single(dataset,
                 opt,
                 pipe,
                 gaussians: GaussianModel,
                 bce_weight=None,
                 depth_weight=None,
                 skip_train=False):
  _ = prepare_output_and_logger(dataset)
  bws = BWScenes(dataset, gaussians, is_new_gaussians=True)
  gaussians.training_setup(opt)

  ema_loss_for_log = 0.0

  if not skip_train:
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for i in range(1, opt.iterations + 1):
      gaussians.update_learning_rate(i)
      # Every 1000 its we increase the levels of SH up to a maximum degree
      if i % 1000 == 0:
        gaussians.oneupSHdegree()

      # Pick a random Camera
      viewpoint_cam, background = bws.pop_black() if (
          i % 2 == 0) else bws.pop_white()

      # Render
      render_pkg = render(viewpoint_cam, gaussians, pipe, background)
      image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
          render_pkg["visibility_filter"], render_pkg["radii"]
      gt_image = viewpoint_cam.original_image.cuda().float()
      loss = eval_img_loss(image, gt_image, opt)

      if bce_weight is not None:
        loss += bce_weight * eval_opacity_bce_loss(gaussians.get_opacity)
      # if (depth_weight is not None) and (viewpoint_cam.image_depth is not None) and (i > opt.opacity_reset_interval):
      if (depth_weight is not None) and (viewpoint_cam.image_depth is not None):
        depth = render_pkg['depth']
        gt_depth = viewpoint_cam.image_depth.cuda().float()
        loss += depth_weight * eval_depth_loss(depth, gt_depth)

      loss.backward()
      with torch.no_grad():
        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if i % 10 == 0:
          progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
          progress_bar.update(10)

        # Save ply
        if i in [7000, opt.iterations]:
          gaussians.save_ply(os.path.join(
              dataset.model_path, f'point_cloud/iteration_{i}/point_cloud.ply'),
                             prune=True)

        # Densification
        if i < opt.densify_until_iter:
          # Keep track of max radii in image-space for pruning
          gaussians.max_radii2D[visibility_filter] = torch.max(
              gaussians.max_radii2D[visibility_filter],
              radii[visibility_filter])
          gaussians.add_densification_stats(viewspace_point_tensor,
                                            visibility_filter)
          # copy or split
          if i > opt.densify_from_iter and i % opt.densification_interval == 0:
            size_threshold = 20 if i > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                        bws.get_cameras_extent(),
                                        size_threshold)
          # opacity reset
          if i % opt.opacity_reset_interval == 0 or (
              dataset.white_background and i == opt.densify_from_iter):
            gaussians.reset_opacity()
        # Optimizer step
        if i < opt.iterations:
          gaussians.optimizer.step()
          gaussians.optimizer.zero_grad(set_to_none=True)
    progress_bar.close()
  with torch.no_grad():
    prune_mask = (gaussians.get_opacity < 0.005).squeeze()
    gaussians.prune_points(prune_mask)
  torch.save((gaussians.capture(), opt.iterations),
             os.path.join(dataset.model_path, 'chkpnt.pth'))


def get_gaussians(model_path,
                  from_chk=True,
                  iters=None,
                  from_pgsr=False) -> GaussianModel:
  gaussians = GaussianModel(0)
  if from_chk:
    dataset, pipes, opt = get_default_args()
    model_params, _ = torch.load(os.path.join(model_path, 'chkpnt.pth'))
    if from_pgsr:
      gaussians.restore_gpsr(model_params, opt)
    else:
      gaussians.restore(model_params, opt)
  else:
    _, _, opt = get_default_args()
    gaussians.load_ply(
        os.path.join(model_path,
                     f'point_cloud/iteration_{iters}/point_cloud.ply'))
    gaussians.training_setup(opt)
  return gaussians


def print_motion_params(out_path: str):
  t = np.load(os.path.join(out_path, 't_pre.npy'))
  r = np.load(os.path.join(out_path, 'r_pre.npy'))
  print('t:', t)
  print('r:', r)


def plot_hist(x, path: str, bins=100):
  plt.figure()
  try:
    plt.hist(x, bins=bins)
  except:
    plt.hist(x.detach().cpu().numpy(), bins=bins)
  plt.savefig(path)


def mk_output_dir(out_path: str, data_path: str):
  os.makedirs(out_path, exist_ok=True)
  dataset, pipes, opt = get_default_args()
  dataset.eval = True
  dataset.sh_degree = 0
  dataset.source_path = os.path.realpath(data_path)
  dataset.model_path = out_path
  _ = prepare_output_and_logger(dataset)


def init_mpp(gaussians_st: GaussianModel,
             gaussians_ed: GaussianModel,
             thr=None,
             sig_scale=1.0):
  cds_st = get_per_point_cd(gaussians_st, gaussians_ed)
  cds_st_normalized = cds_st / torch.max(cds_st)

  eps = 1e-6
  csn_is = inverse_sigmoid(torch.clamp(cds_st_normalized, eps, 1 - eps))
  if thr is None:
    thr = otsu_with_peak_filtering(csn_is.detach().cpu().numpy(),
                                   bias_factor=1.25)
    print(thr)
  csn_shifted = torch.sigmoid(
      (inverse_sigmoid(cds_st_normalized) - thr) * sig_scale)
  return cds_st_normalized, csn_is, csn_shifted


# def get_cluster_centers(pts: torch.tensor, num: int) -> torch.tensor:
#     kmeans = KMeans(n_clusters=num, random_state=42)
#     kmeans.fit(pts.detach().cpu().numpy())
#     centroids = kmeans.cluster_centers_
#     return torch.tensor(centroids, device=pts.device)
#
# def get_ppp_from_dist(pts: torch.tensor, centers: torch.tensor):
#     dist_to_centers = (pts.unsqueeze(1) - centers).norm(dim=2)
#     prob = dist_to_centers / torch.sum(dist_to_centers, dim=1, keepdim=True)
#     return prob


def get_ppp_from_gmm(train_pts: torch.tensor, test_pts: torch.tensor,
                     num: int) -> torch.tensor:
  gmm = GaussianMixture(n_components=num, random_state=42)
  gmm.fit(train_pts.detach().cpu().numpy())
  prob = gmm.predict_proba(test_pts.detach().cpu().numpy())
  return torch.tensor(prob, device=test_pts.device)


def get_ppp_from_gmm_v2(train_pts: torch.tensor, test_pts: torch.tensor,
                        num: int) -> torch.tensor:
  gmm = GaussianMixture(n_components=num, random_state=42)
  gmm.fit(train_pts.detach().cpu().numpy())
  means = gmm.means_
  covariances = gmm.covariances_

  inv_covariances = np.linalg.inv(covariances)
  diff = test_pts.detach().cpu().numpy()[:, np.newaxis, :] - means
  quad = np.einsum('pki,kij,pkj->pk', diff, inv_covariances, diff)
  # prob = 1 / (np.sqrt(quad) + 1e-6)
  prob = 1 / (quad**2 + 1e-6)
  prob /= np.sum(prob, axis=1, keepdims=True)
  return torch.tensor(prob, device=test_pts.device)


def eval_init_gmm_params(train_pts: torch.tensor,
                         num: int) -> tuple[torch.tensor, torch.tensor]:
  gmm = GaussianMixture(n_components=num, random_state=42)
  gmm.fit(train_pts.detach().cpu().numpy())
  means = torch.tensor(gmm.means_, device=train_pts.device)
  covariances = torch.tensor(gmm.covariances_, device=train_pts.device)
  return means, covariances


def eval_mu_sigma(pts: np.ndarray) -> tuple[np.array, np.array]:
  gmm = GaussianMixture(n_components=1, random_state=42)
  gmm.fit(pts)
  return gmm.means_, gmm.covariances_


def modify_scaling(cov: torch.tensor, scaling_modifier=1.0) -> torch.tensor:
  scaling, rotation = decompose_covariance_matrix(cov)
  scaling = torch.diag_embed(scaling * scaling_modifier)
  return rotation @ scaling @ scaling @ rotation.transpose(1, 2)


def get_depths(
    pts: np.ndarray,  # given pts in world coordinate
    depth_map: np.ndarray,
    R: np.array,
    T: np.array,
    FovY: np.array,
    FovX: np.array,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
  zfar = 100.0
  znear = 0.01
  projection_matrix = getProjectionMatrix(
      znear, zfar, FovX, FovY, blender_convention=True).cpu().numpy()
  w2c_colmap = getWorld2View(R, T)
  c2w_colmap = np.linalg.inv(w2c_colmap)
  c2w_blender = c2w_colmap.copy()
  c2w_blender[:3, 1:3] *= -1
  w2c_blender = np.linalg.inv(c2w_blender)

  # Transform world coordinates to camera space
  pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 4)
  pts_camera = (w2c_blender @ pts_homo.T).T  # (N, 4)
  # Transform camera space to clip space
  pts_clip = (projection_matrix @ pts_camera.T).T  # (N, 4)
  pts_clip /= pts_clip[:, 3, np.newaxis]  # Perspective division
  pts_camera /= pts_camera[:, 3, np.newaxis]  # Perspective division
  # Convert to pixel coordinates
  x_pixel = ((pts_clip[:, 0] + 1) * image_width / 2).astype(int)
  y_pixel = ((1 - pts_clip[:, 1]) * image_height / 2).astype(int)
  # Clamp pixel coordinates to valid range
  x_pixel = np.clip(x_pixel, 0, image_width - 1)
  y_pixel = np.clip(y_pixel, 0, image_height - 1)
  # Get depths from the depth map
  pixel_depths = depth_map[y_pixel, x_pixel]

  pixel_depths[pixel_depths < 0.01] = 114.514
  return -pts_camera[:, 2], pixel_depths


def eval_visibility(pts: np.array,
                    data_path: str,
                    eps: float = 0.01) -> np.array:
  data_path = os.path.realpath(data_path)
  train_cam_infos = readCamerasFromTransforms(data_path,
                                              "transforms_train.json",
                                              white_background=False)
  vis = np.zeros(len(pts), dtype='bool')
  for cam_info in train_cam_infos:
    depth_pts, depth_obj = get_depths(
        pts=pts,
        depth_map=cam_info.image_d,
        R=cam_info.R,
        T=cam_info.T,
        FovX=cam_info.FovX,
        FovY=cam_info.FovY,
        image_width=cam_info.width,
        image_height=cam_info.height,
    )
    vis |= (depth_pts < depth_obj + eps)
  return vis


def get_vis_mask(gaussians: GaussianModel,
                 data_path: str,
                 eps: float = 0.01) -> torch.tensor:
  xyz = gaussians.get_xyz.detach().cpu().numpy()
  return torch.tensor(eval_visibility(xyz, data_path, eps),
                      device=gaussians.get_xyz.device)


def estimate_se3(p: torch.tensor, p_prime: torch.tensor, k_neighbors=21):
  p = p.detach().cpu().numpy()
  p_prime = p_prime.detach().cpu().numpy()
  nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(p)
  _, indices = nbrs.kneighbors(p)  # indices: (n_samples, k_neighbors)

  def estimate_se3_batch(p_subset, p_prime_subset):
    # p_subset 和 p_prime_subset: (n_samples, k_neighbors, 3)
    # 计算质心
    p_mean = np.mean(p_subset, axis=1, keepdims=True)  # (n_samples, 1, 3)
    p_prime_mean = np.mean(p_prime_subset, axis=1, keepdims=True)
    # 去中心化
    p_centered = p_subset - p_mean  # (n_samples, k_neighbors, 3)
    p_prime_centered = p_prime_subset - p_prime_mean
    # 协方差矩阵 H
    H = np.einsum('ijk,ijl->ikl', p_centered,
                  p_prime_centered)  # (n_samples, 3, 3)
    # SVD 分解
    U, _, Vt = np.linalg.svd(H)  # U, Vt: (n_samples, 3, 3)
    R = np.einsum('ijk,ikl->ijl', Vt, U)  # (n_samples, 3, 3)
    # 修正旋转矩阵（确保 det(R) = 1）
    det_R = np.linalg.det(R)
    Vt[:, :, -1] *= np.sign(det_R)[:, None]  # 调整最后一行
    R = np.einsum('ijk,ikl->ijl', Vt, U)
    # 计算平移
    t = p_prime_mean.squeeze(axis=1) - np.einsum('ijk,ik->ij', R,
                                                 p_mean.squeeze(axis=1))
    return R, t

  # 获取邻域点集
  p_neighbors = p[indices]  # (n_samples, k_neighbors, 3)
  p_prime_neighbors = p_prime[indices]
  return estimate_se3_batch(
      p_neighbors, p_prime_neighbors)  # R: (n_samples, 3, 3), t: (n_samples, 3)


def save_axis_mesh(d: np.ndarray,
                   o: np.ndarray,
                   filepath: str,
                   o_ref=None,
                   to_gaussians=False,
                   c=(1, 1, 1),
                   arrow_scale=1.0):
  r_arrow = None
  if o_ref is None:
    o_ref = np.zeros(3)
  if (np.abs(o) < 1e-6).all():  # prismatic
    o = o_ref
  else:
    t = np.dot(o_ref - o, d) / np.dot(d, d)
    o = o + t * d

  axis = o3d.geometry.TriangleMesh.create_arrow(
      cylinder_radius=0.01 * arrow_scale,
      cone_radius=0.02 * arrow_scale,
      cylinder_height=0.7 * arrow_scale,
      cone_height=0.04 * arrow_scale)
  arrow = np.array([0., 0., 1.], dtype=np.float32)
  n = np.cross(arrow, d)
  if np.linalg.norm(n) > 1e-4:
    rad = np.arccos(np.dot(arrow, d))
    r_arrow = rotation_matrix_from_axis_angle(n, rad)
    axis.rotate(r_arrow, center=(0, 0, 0))
  axis.translate(o[:3])
  o3d.io.write_triangle_mesh(filepath, axis)

  if to_gaussians:
    with torch.no_grad():
      ags = get_gaussians('output/zarrow', from_chk=False, iters=30000)
      r_arrow = torch.tensor(r_arrow,
                             device=ags.get_xyz.device,
                             dtype=ags.get_xyz.dtype)
      o = torch.tensor(o, device=ags.get_xyz.device, dtype=ags.get_xyz.dtype)
      if np.linalg.norm(n) > 1e-4:
        ags.get_xyz[:] = torch.einsum('ij,nj->ni', r_arrow, ags.get_xyz[:])
        ags.get_rotation_raw[:] = quat_mult(mat2quat(r_arrow),
                                            ags.get_rotation[:])
      ags.get_xyz[:] = ags.get_xyz[:] + o[:3]
      fused_color = torch.zeros(ags.size(), 3)
      fused_color[:] = torch.tensor(c)
      ags.save_vis(filepath, fused_color)
  # fi


def list_of_clusters(cluster_dir: str, num_movable: int = 0, ret_normal=False):
  pts = []
  nms = []
  for i in np.arange(80):
    ply_file = os.path.join(cluster_dir, f'points3d_{i}.ply')
    if not os.path.exists(ply_file):
      continue
    pts.append(np.asarray(fetchPly(ply_file).points))
    nms.append(np.asarray(fetchPly(ply_file).normals))
    if len(pts) == num_movable:
      break
  assert len(pts) == num_movable
  if ret_normal:
    return pts, nms
  return pts


def put_axes(out_path,
             st_path,
             num_movable: int,
             cluster_path: str = "clustering"):
  print(f"Pass put_axes funtion...")
  return
  gaussians_st = get_gaussians(st_path, from_chk=False, iters=30000)
  for k in range(num_movable):
    axis0 = GaussianModel(0).load_ply(
        os.path.join(out_path, f'{cluster_path}/axes_gaussians/axis{k}_0.ply'))
    axis1 = GaussianModel(0).load_ply(
        os.path.join(out_path, f'{cluster_path}/axes_gaussians/axis{k}_1.ply'))
    axis2 = GaussianModel(0).load_ply(
        os.path.join(out_path, f'{cluster_path}/axes_gaussians/axis{k}_2.ply'))
    obj_with_axes = gaussians_st + axis0 + axis1 + axis2
    obj_with_axes.save_ply(
        os.path.join(out_path,
                     f'point_cloud/iteration_-{k + 100}/point_cloud.ply'))
  return


def get_tr_proximity_matrix(r, t):

  def are_transforms_close(R1,
                           t1,
                           R2,
                           t2,
                           translation_threshold=0.05,
                           rotation_threshold_deg=5):
    """
        判断两个三维刚体变换 (R, t) 是否足够接近。

        接近的定义是平移向量之间的欧几里得距离和旋转矩阵之间的夹角
        分别小于给定的阈值。

        Args:
            R1 (np.ndarray): 形状为 (3, 3) 的旋转矩阵1。
            t1 (np.ndarray): 形状为 (3,) 的平移向量1。
            R2 (np.ndarray): 形状为 (3, 3) 的旋转矩阵2。
            t2 (np.ndarray): 形状为 (3,) 的平移向量2。
            translation_threshold (float): 允许的平移向量之间的最大欧几里得距离。
            rotation_threshold_deg (float): 允许的旋转矩阵之间的最大夹角（单位：度）。

        Returns:
            bool: 如果平移和旋转的差异都在阈值内，则返回 True，否则返回 False。
        """
    # 1. 计算平移向量之间的欧几里得距离
    translation_dist = np.linalg.norm(t1 - t2)

    if translation_dist / (np.linalg.norm(t1) +
                           np.linalg.norm(t2)) > translation_threshold:
      return False

    # 2. 计算旋转矩阵之间的夹角
    # 相对旋转矩阵 R_rel 可以将 R1 的姿态变换到 R2 的姿态
    R_rel = np.dot(R2, R1.T)

    # 旋转矩阵的迹（trace）和旋转角度 theta 的关系是: trace(R) = 1 + 2*cos(theta)
    # 因此, theta = arccos((trace(R) - 1) / 2)
    # 使用 np.clip 防止由于浮点数精度问题导致 trace 的值略微超出 [-1, 3] 的范围
    trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
    angle_rad = np.arccos((trace_val - 1) / 2.0)
    angle_deg = np.rad2deg(angle_rad)

    if angle_deg > rotation_threshold_deg:
      return False
    return True

  K = r.shape[0]
  proximity_matrix = np.zeros((K, K), dtype=bool)
  for i in range(K):
    for j in range(K):
      proximity_matrix[i, j] = are_transforms_close(r[i], t[i], r[j], t[j])
  return proximity_matrix


def get_minimum_angles(axes, r, t):
  num_movable = axes.shape[0]

  angles = np.zeros(num_movable)
  for k in range(num_movable):
    d = t[k] / np.linalg.norm(t[k])
    if np.abs(np.trace(r[k])) < 1 + 2 * np.cos(np.deg2rad(5)):
      d = get_rotation_axis(r[k], t[k])[1]

    dots = np.clip(axes[k] @ d, -1.0, 1.0)
    ang_rad = np.arccos(np.abs(dots).max())
    angles[k] = np.rad2deg(ang_rad)
  return angles


def get_obb_proximity_matrix(axes, centers, extents, num_samples=100000):
  """
    Calculates a matrix of proximity scores between OBBs based on intersection
    volume ratio, approximated using a vectorized Monte Carlo method.

    The score matrix[i, j] is an approximation of:
    intersection_volume(OBB_i, OBB_j) / volume(OBB_i)

    Args:
        axes (np.ndarray): Shape (K, 3, 3). Orientation axes for K OBBs.
        centers (np.ndarray): Shape (K, 3). Center points for K OBBs.
        extents (np.ndarray): Shape (K, 3). Half-lengths (extents) for K OBBs.
        num_samples (int): Number of random points to sample for the approximation.

    Returns:
        np.ndarray: A (K, K) float matrix of proximity scores.
    """

  def are_points_inside_obb(points, obb_center, obb_extents, obb_axes):
    """
        Checks if an array of points are inside an OBB using vectorization.

        Args:
            points (np.ndarray): Shape (N, 3). The N points to check.
            obb_center (np.ndarray): Shape (3,). The center of the OBB.
            obb_extents (np.ndarray): Shape (3,). The half-lengths of the OBB.
            obb_axes (np.ndarray): Shape (3, 3). The orientation axes of the OBB.

        Returns:
            np.ndarray: A boolean array of shape (N,) where True indicates the
                        corresponding point is inside the OBB.
        """
    # 1. Translate points to be relative to the OBB's center.
    # Broadcasting handles this: (N, 3) - (3,) -> (N, 3)
    vecs_to_points = points - obb_center

    # 2. Rotate the points into the OBB's local coordinate system.
    # obb_axes is a rotation matrix R, its transpose R.T is the inverse rotation.
    # We perform (P - C) @ R.T for all points at once.
    # (N, 3) @ (3, 3) -> (N, 3)
    local_points = np.dot(vecs_to_points,
                          obb_axes.T)  # Using dot for clarity, same as @

    # 3. Check if the local coordinates of all points are within the extents.
    # np.abs() is element-wise.
    # The comparison uses broadcasting: (N, 3) <= (3,) -> (N, 3)
    # np.all(..., axis=1) checks along the xyz-axis for each point.
    # The result is a boolean array of shape (N,).
    return np.all(np.abs(local_points) <= obb_extents, axis=1)

  K = centers.shape[0]
  proximity_matrix = np.zeros((K, K), dtype=float)
  merge_likelihood = np.zeros(K)

  for i in range(K):
    # Generate random points in OBB i's local coords and transform to world
    local_samples = np.random.uniform(-1, 1, (num_samples, 3)) * extents[i]
    world_samples = centers[i] + np.dot(local_samples, axes[i])

    is_in_others = np.zeros(num_samples, dtype=bool)
    for j in range(K):
      if i == j:
        proximity_matrix[i, j] = -1
        continue

      # Use the vectorized function to check all points at once
      is_inside = are_points_inside_obb(world_samples, centers[j], extents[j],
                                        axes[j])
      is_in_others |= is_inside

      # np.sum() on a boolean array counts the number of True values
      intersection_count = np.sum(is_inside)

      proximity_matrix[i, j] = intersection_count / num_samples
    merge_likelihood[i] = is_in_others.astype(int).mean()

  return proximity_matrix, merge_likelihood


def fetch_motion_params(out_path: str,
                        factor: float = 0.5) -> tuple[np.ndarray, np.array]:
  with open(os.path.join(out_path, 'motion_init.json'), 'r') as json_file:
    motions = json.load(json_file)

  num_parts = len(motions)
  translations = np.zeros((num_parts, 3))
  rotations = np.asarray([np.eye(3) for _ in range(num_parts)])

  for k, motion in enumerate(motions):
    if motion['mode'] == 'trans':
      translations[k] = factor * motion['t'] * np.array(motion['u'])
    elif motion['mode'] == 'rot':
      u = np.asarray(motion["u"])
      omega = factor * motion["omega"]
      a = np.asarray(motion["a_point"])
      r = rodrigues_rotation_matrix(u, omega)
      t = a - r @ a
      translations[k], rotations[k] = t, r.T
    else:
      raise ValueError(f"Unknown mode: {motion['mode']}")
  return translations, rotations


if __name__ == '__main__':
  arrow_path = 'output/zarrow/mesh.ply'
  save_axis_mesh(np.array([0., 0., 1.]), np.zeros(3), arrow_path)
  pass
