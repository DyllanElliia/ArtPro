import os
import torch

import open3d as o3d

from scene import Scene, GaussianModel
from utils.general_utils import safe_state, estimate_normals_o3d, get_bounding_box, DisjointSet, find_files_with_suffix

try:
  from torch.utils.tensorboard import SummaryWriter
  TENSORBOARD_FOUND = True
except ImportError:
  TENSORBOARD_FOUND = False

import json
import numpy as np
from pytorch3d.loss import chamfer_distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from arguments import get_default_args
from utils.general_utils import eval_quad, inverse_sigmoid, value_to_rgb, estimate_principal_directions, \
    find_and_unify_orthogonal
# from scene.multipart_models_quat import MPArtModelJoint, GMMArtModel, COLORS
from scene.multipart_models_cycle import GMMArtModel, COLORS
from scene.multipart_models_final import MPArtModelJoint
from scene.dataset_readers import fetchPly, storePly

from main_utils import train_single, get_gaussians, plot_hist, \
    mk_output_dir, init_mpp, eval_mu_sigma, save_axis_mesh, list_of_clusters, \
    put_axes, get_obb_proximity_matrix, get_tr_proximity_matrix, get_minimum_angles
from metric_utils import get_gt_motion_params, interpret_transforms, eval_axis_metrics, \
    get_pred_point_cloud, get_gt_point_clouds, eval_geo_metrics, stat_axis_metrics

cd_thr = -5
dbs_eps = 0.008
from_pgsr = False


def train_single_demo(path, data_path, skip_train=False):
  dataset, pipes, opt = get_default_args()
  safe_state(False)
  torch.autograd.set_detect_anomaly(False)

  dataset.eval = True
  dataset.sh_degree = 0
  gaussians = GaussianModel(dataset.sh_degree)
  dataset.source_path = os.path.realpath(data_path)
  dataset.model_path = path
  train_single(dataset,
               opt,
               pipes,
               gaussians,
               depth_weight=4.0,
               bce_weight=0.01,
               skip_train=skip_train)


def get_K(out_path: str, cluster_folder: str = "clustering") -> int:
  folder = 'clusters'
  # folder = 'clusters-unmerged'
  return len(
      find_files_with_suffix(
          os.path.join(out_path, f'{cluster_folder}/{folder}'), '.ply'))


def cluster_demo(out_path: str,
                 data_path: str,
                 num_movable: int,
                 cluster_path: str = "clustering",
                 thr: int = -5,
                 eps: float = 0.008):
  if num_movable == 10:
    thr = -10
    eps = 0.004

  mk_output_dir(out_path, os.path.join(data_path, 'start'))
  ply_path = os.path.join(out_path, cluster_path)
  os.makedirs(ply_path, exist_ok=True)
  os.makedirs(os.path.join(ply_path, 'clusters'), exist_ok=True)

  st_data = os.path.join(os.path.realpath(data_path), 'start')
  ed_data = os.path.join(os.path.realpath(data_path), 'end')
  xyz_st = np.asarray(fetchPly(os.path.join(st_data, 'points3d.ply')).points)
  xyz_ed = np.asarray(fetchPly(os.path.join(ed_data, 'points3d.ply')).points)

  x = torch.tensor(xyz_st, device='cuda').unsqueeze(0)
  y = torch.tensor(xyz_ed, device='cuda').unsqueeze(0)
  cd = chamfer_distance(x,
                        y,
                        batch_reduction=None,
                        point_reduction=None,
                        single_directional=True)[0][0]
  cd /= torch.max(cd)
  cd_is = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6))
  plot_hist(cd_is, os.path.join(ply_path, 'cd_is-1m.png'))

  mask = (cd_is > thr)
  x = x[0][mask].detach().cpu().numpy()

  if True:
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(x)
    distances, _ = neigh.kneighbors(x)
    distances = np.sort(distances[:, -1])
    plot_hist(distances, os.path.join(ply_path, 'dist-1m.png'))

  clustering = DBSCAN(eps=eps, min_samples=num_movable).fit(x)

  labels = clustering.labels_
  pts = sorted([(k, x[labels == k]) for k in np.unique_values(labels)],
               key=lambda item: len(item[1]),
               reverse=True)
  for k, pcd in pts[:num_movable]:
    if k == -1:
      k, pcd = pts[num_movable]
      print('warning: has -1')
    normals = estimate_normals_o3d(pcd)
    storePly(os.path.join(ply_path, f'clusters/points3d_{k}.ply'), pcd,
             np.zeros_like(pcd), normals)
  storePly(os.path.join(ply_path, f'points3d.ply'), x, np.zeros_like(x))
  print(f'Saved clustered point clouds to {ply_path}.')


def _cluster_ab_from_arrays(Pi, Pj, num_movable, thr, eps, min_samples=None):
  assert Pi.ndim == 2 and Pi.shape[1] == 3, "Pi must be of shape (N, 3)"
  assert Pj.ndim == 2 and Pj.shape[1] == 3, "Pj must be of shape (M, 3)"

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  x = torch.as_tensor(Pi, dtype=torch.float32, device=device).unsqueeze(0)
  y = torch.as_tensor(Pj, dtype=torch.float32, device=device).unsqueeze(0)
  with torch.no_grad():
    cd = chamfer_distance(x,
                          y,
                          batch_reduction=None,
                          point_reduction=None,
                          single_directional=True)[0][0]
    cd = cd / (torch.max(cd) + 1e-12)
    cd = torch.clamp(cd, 1e-6, 1 - 1e-6)
    cd_is = inverse_sigmoid(cd)

  dists_np = cd_is.detach().cpu().numpy()
  seed_mask = (cd_is > thr)
  seed_indices = torch.nonzero(
      seed_mask, as_tuple=False).squeeze(1).detach().cpu().numpy()

  info = {
      "dists": dists_np,
      "seed_indices": seed_indices,
      "curr_tau": float(thr),
      "cluster_seeds": [],
  }

  if seed_indices.size == 0:
    labels = np.zeros(Pi.shape[0], dtype=np.int32)
    empty_arr = np.empty(0, dtype=np.int32)
    info["parts_idx"] = []
    info["cluster_labels"] = empty_arr.copy()
    info["noise_indices"] = empty_arr.copy()
    info["cluster_seeds"] = [empty_arr]
    return labels, [], info

  points_seed = Pi[seed_indices]
  if min_samples is None:
    min_samples = max(3, min(num_movable, seed_indices.size))
  min_samples = max(1, min(min_samples, seed_indices.size))

  clustering = DBSCAN(eps=float(eps), min_samples=min_samples)
  clustering.fit(points_seed)
  labels_sel = clustering.labels_

  clusters = []
  for lab in np.unique(labels_sel):
    if lab < 0:
      continue
    cluster_idx = seed_indices[labels_sel == lab]
    clusters.append(np.asarray(cluster_idx, dtype=np.int32))
  clusters.sort(key=lambda arr: arr.size, reverse=True)

  noise_indices = seed_indices[labels_sel == -1]
  info["noise_indices"] = noise_indices.astype(np.int32, copy=False)
  if clusters:
    fill_needed = max(0, num_movable - len(clusters))
    if fill_needed > 0 and noise_indices.size:
      splits = np.array_split(noise_indices, fill_needed)
      for split in splits:
        if split.size:
          clusters.append(np.asarray(split, dtype=np.int32))
  else:
    splits = np.array_split(seed_indices,
                            min(num_movable, max(1, seed_indices.size)))
    clusters = [
        np.asarray(split, dtype=np.int32) for split in splits if split.size
    ]

  info["cluster_seeds"] = clusters
  selected_parts = clusters[:num_movable]

  labels = np.zeros(Pi.shape[0], dtype=np.int32)
  parts = []
  for pid, idx in enumerate(selected_parts, start=1):
    if idx.size == 0:
      continue
    labels[idx] = pid
    parts.append(idx)

  info["parts_idx"] = parts
  info["cluster_labels"] = labels_sel.astype(np.int32, copy=False)
  return labels, parts, info


def cluster_ab_demo(Pi_or_out_path,
                    Pj_or_data_path,
                    num_movable,
                    cluster_path="clustering",
                    thr=-5,
                    eps=0.008,
                    min_samples=None):
  """Cluster points based on AB distance and return adaptive-style outputs."""
  if num_movable == 10:
    thr = -10
    eps = 0.008

  if isinstance(Pi_or_out_path, (str, bytes, os.PathLike)):
    out_path = Pi_or_out_path
    data_path = Pj_or_data_path
    st_data = os.path.join(os.path.realpath(data_path), 'start')
    ed_data = os.path.join(os.path.realpath(data_path), 'end')
    Pi = np.asarray(fetchPly(os.path.join(st_data, 'points3d.ply')).points,
                    dtype=np.float32)
    Pj = np.asarray(fetchPly(os.path.join(ed_data, 'points3d.ply')).points,
                    dtype=np.float32)

    labels, parts, info = _cluster_ab_from_arrays(Pi, Pj, num_movable, thr, eps,
                                                  min_samples)

    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    ply_path = os.path.join(out_path, cluster_path)
    os.makedirs(ply_path, exist_ok=True)
    os.makedirs(os.path.join(ply_path, 'clusters'), exist_ok=True)

    plot_hist(info["dists"], os.path.join(ply_path, 'cd_is-1m.png'))

    if info["seed_indices"].size:
      changed = Pi[info["seed_indices"]]
      storePly(os.path.join(ply_path, 'points3d.ply'), changed,
               np.zeros_like(changed))

      if changed.shape[0] >= 2:
        k = min(3, changed.shape[0])
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(changed)
        distances, _ = neigh.kneighbors(changed)
        distances = np.sort(distances[:, -1])
        plot_hist(distances, os.path.join(ply_path, 'dist-1m.png'))

    for pid, idx in enumerate(parts):
      pts = Pi[idx]
      if pts.shape[0] == 0:
        continue
      normals = estimate_normals_o3d(pts)
      storePly(os.path.join(ply_path, f'clusters/points3d_{pid}.ply'), pts,
               np.zeros_like(pts), normals)

    print(f'Saved clustered point clouds to {ply_path}.')
    return labels, parts, info

  Pi = np.asarray(Pi_or_out_path, dtype=np.float32)
  Pj = np.asarray(Pj_or_data_path, dtype=np.float32)
  return _cluster_ab_from_arrays(Pi, Pj, num_movable, thr, eps, min_samples)


def part_init_demo(out_path,
                   st_path,
                   ed_path,
                   cluster_path: str = "clustering",
                   num_movable: int = 0):
  if num_movable == 0:
    num_movable = get_K(out_path, cluster_path)

  gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr)
  gaussians_ed = get_gaussians(ed_path, from_chk=True, from_pgsr=from_pgsr)
  cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed, thr=-4.5)
  mask_m = (mpp > .5)
  gaussians_st[mask_m].save_ply(
      os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
  plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
  np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())

  pts = list_of_clusters(os.path.join(out_path, f'{cluster_path}/clusters'),
                         num_movable)
  mu = np.zeros((num_movable, 3))
  sigma = np.zeros((num_movable, 3, 3))
  for i in np.arange(num_movable):
    if pts[i].shape[0] < 2:
      if pts[i].shape[0] == 1:
        mu[i] = pts[i][0]
      else:
        mu[i] = np.zeros(3)
      sigma[i] = np.eye(3) * 0.001
    else:
      mu[i], sigma[i] = eval_mu_sigma(pts[i])
  np.save(os.path.join(out_path, 'mu_init.npy'), mu)
  np.save(os.path.join(out_path, 'sigma_init.npy'), sigma)
  print(f"Saved mu and sigma for {num_movable} parts to {out_path}.")


def joint_init_demo(out_path: str,
                    st_path: str,
                    cluster_path: str = "clustering",
                    num_movable: int = 0):
  if num_movable == 0:
    num_movable = get_K(out_path)

  def rotate_axes(axes_to_rotate: np.ndarray, theta_deg: float) -> np.ndarray:
    random_axis = np.random.rand(3)
    while np.linalg.norm(random_axis) < 1e-4:
      random_axis = np.random.rand(3)
    random_axis /= np.linalg.norm(random_axis)

    theta_rad = np.deg2rad(theta_deg)
    x, y, z = random_axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    I = np.eye(3)
    R = I + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * np.dot(K, K)
    return axes_to_rotate @ R.T

  nrm_dir = os.path.join(out_path, f'{cluster_path}/nrm_axes')
  gaussians_dir = os.path.join(out_path, f'{cluster_path}/axes_gaussians')
  os.makedirs(nrm_dir, exist_ok=True)
  print(nrm_dir)
  os.makedirs(gaussians_dir, exist_ok=True)

  parts, nms = list_of_clusters(os.path.join(out_path,
                                             f'{cluster_path}/clusters'),
                                num_movable,
                                ret_normal=True)
  mu = np.load(os.path.join(out_path, 'mu_init.npy'))

  axes = np.zeros((num_movable, 3, 3))
  P_axiss = []
  for k, nm in enumerate(nms):
    try:
      axes[k] = estimate_principal_directions(nm, ort='gs')
    except Exception as e:
      print(f'Warning: part {k} principal direction estimation failed: {e}')
      axes[k] = np.eye(3)
    # check nan
    if np.any(np.isnan(axes[k])):
      print(f'Warning: part {k} has nan axes, use default axes.')
      axes[k] = np.eye(3)
    P_axiss.append(axes[k])

  # for k in [0, 1]:
  #     axes[k] = rotate_axes(axes[k], 5)

  neighbors = find_and_unify_orthogonal(axes)

  bb_centers, bb_extents = [], []
  for k, pcd in enumerate(parts):
    o = np.zeros(3)
    # dirs = dirs_p if (types[k] == 'p') else estimate_principal_directions(nms[k], ort='gs')
    dirs = axes[k]
    # for i in np.arange(3):
    #   d = -dirs[i] if (dirs[i] @ mu[k] < 0) else dirs[i]
    #   save_axis_mesh(d, o, os.path.join(nrm_dir, f'axis{k}_{i}.ply'), mu[k])
    #   save_axis_mesh(d,
    #                  o,
    #                  os.path.join(gaussians_dir, f'axis{k}_{i}.ply'),
    #                  mu[k],
    #                  to_gaussians=True,
    #                  c=COLORS[i])
    # axes.append(dirs)
    print(dirs)
    centers, extents = get_bounding_box(pcd, dirs)
    bb_centers.append(centers)
    bb_extents.append(extents)

    print('done with axis', k)
  np.save(os.path.join(out_path, f'{cluster_path}/axes.npy'), axes)
  np.save(os.path.join(out_path, f'{cluster_path}/bb_centers.npy'), bb_centers)
  np.save(os.path.join(out_path, f'{cluster_path}/bb_extents.npy'), bb_extents)
  np.save(os.path.join(out_path, f'{cluster_path}/neighbors.npy'), neighbors)

  obb_vectors, obb_origins = [], []
  for center, axes, extent in zip(bb_centers, P_axiss, bb_extents):
    for axis_vec, axis_extent in zip(axes, extent):
      vec = axis_vec * axis_extent  # (3,3)
      obb_vectors.extend([vec, -vec])
      obb_origins.extend([center, center])
  from utils.mesh_vis_utils import export_vectors_to_ply
  export_vectors_to_ply(
      np.asarray(obb_vectors),
      os.path.join(nrm_dir, f'axiss_all.ply'),
      mode="lines",
      origin=np.asarray(obb_origins),
  )

  put_axes(out_path, st_path, num_movable, cluster_path)
  return bb_centers, P_axiss, bb_extents


def get_two_states_gaussians(st_path: str, ed_path: str):
  torch.autograd.set_detect_anomaly(False)
  gaussians_st = get_gaussians(st_path, from_chk=True,
                               from_pgsr=from_pgsr).cancel_grads()
  gaussians_ed = get_gaussians(ed_path, from_chk=True,
                               from_pgsr=from_pgsr).cancel_grads()
  return gaussians_st, gaussians_ed


def art_optim_demo(out_path: str,
                   st_path: str,
                   ed_path: str,
                   data_path: str,
                   cluster_folder: str = "clustering",
                   num_movable: int = 0,
                   thr=0.85,
                   gt_num=None,
                   depth_weight: float = None,
                   obb_collision_delta: float = None,
                   obb_collision_freeze_rot: bool = None,
                   continue_training: bool = False,
                   curr_iter: int = None,
                   iteration: int = None,
                   am=None,
                   bws=None):
  torch.autograd.set_detect_anomaly(False)
  print("[run] Load gaussians from:", st_path, ed_path)
  gaussians_st = get_gaussians(st_path, from_chk=True,
                               from_pgsr=from_pgsr).cancel_grads()
  gaussians_ed = get_gaussians(ed_path, from_chk=True,
                               from_pgsr=from_pgsr).cancel_grads()
  if continue_training is False:
    if num_movable == 0:
      num_movable = get_K(out_path)
    if gt_num is None:
      gt_num = num_movable

    am = GMMArtModel(gaussians_st,
                     num_movable,
                     cluster_folder=cluster_folder,
                     new_scheme=False)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'),
                   model_path=out_path,
                   thr=cd_thr)
    am.set_init_part_params(out_path, scaling_modifier=1)
    if obb_collision_delta is not None:
      print("[run] Set obb_collision_delta to:", obb_collision_delta)
      am.opt.obb_collision_delta = obb_collision_delta
    if obb_collision_freeze_rot is not None:
      print("[run] Set obb_collision_freeze_rot to:", obb_collision_freeze_rot)
      am.opt.obb_collision_freeze_rot = obb_collision_freeze_rot
    # am.set_init_part_params(out_path, scaling_modifier=1, use_priors=True)

    # am.set_init_params(*fetch_motion_params(out_path, factor=0.5))
    am.save_all_vis(-10)
    curr_iter, iteration = None, None
  else:
    print(
        f"[run] continue training from iteration {curr_iter} to {iteration}...")
    iteration += curr_iter
  if depth_weight is not None:
    print("[run] Set depth weight to:", depth_weight)
    am.opt.depth_weight = depth_weight
  t, r = am.train(gt_gaussians=gaussians_ed,
                  gt_num=gt_num,
                  curr_iter=curr_iter,
                  iteration=iteration,
                  bws=bws)

  ppp = am.get_ppp().detach().cpu().numpy()
  part_indices = np.argmax(ppp, axis=1)
  mpp = am.get_prob.detach().cpu().numpy()
  mask = (mpp > thr)

  gaussians_st = get_gaussians(st_path, from_chk=True,
                               from_pgsr=from_pgsr).cancel_grads()
  am.save_all_vis(-20)
  for i in range(num_movable):
    gaussians_st[mask & (part_indices == i)].save_ply(
        os.path.join(out_path,
                     f'point_cloud/iteration_{21 + i}/point_cloud.ply'))
  gaussians_st[~mask].save_ply(
      os.path.join(out_path, 'point_cloud/iteration_20/point_cloud.ply'))
  rotations_np = [rr.detach().cpu().numpy() for rr in r]
  translations_np = [tt.detach().cpu().numpy() for tt in t]
  np.save(os.path.join(out_path, 'r_pre.npy'), rotations_np)
  np.save(os.path.join(out_path, 't_pre.npy'), translations_np)
  np.save(os.path.join(out_path, 'mask_pre.npy'), mask)
  np.save(os.path.join(out_path, 'part_indices_pre'), part_indices)

  # Get
  xyz_start = gaussians_st.get_xyz.detach().cpu().numpy()
  xyz_end = np.asarray(
      o3d.io.read_point_cloud(
          os.path.join(os.path.realpath(data_path), 'end',
                       'points3d.ply')).points)

  Pcd_parts = []
  parts_idx_list = []
  motions_list = []
  part_ids = []

  pure_translate = [t.detach().cpu().numpy() for t in am._t]
  pure_rotate_c = [c.detach().cpu().numpy() for c in am._c]
  global_indices = np.arange(part_indices.shape[0])
  part_mask = np.zeros(part_indices.shape[0], dtype=bool)

  for part_id in range(num_movable):
    in_part = (part_indices == part_id) & mask
    idx = global_indices[in_part]
    if idx.size == 0:
      continue
    part_mask[idx] = True
    Pcd_parts.append(xyz_start[idx].astype(np.float32, copy=False))
    parts_idx_list.append(idx.astype(np.int32, copy=False))
    motions_list.append({
        "R": rotations_np[part_id],
        # "t": translations_np[part_id]
        "t": pure_translate[part_id],
        "c": pure_rotate_c[part_id],
        "R_old": rotations_np[part_id],
        "t_old": translations_np[part_id],
    })
    part_ids.append(part_id)

  static_idx = np.nonzero(~part_mask)[0]
  Ps = xyz_start[static_idx] if static_idx.size else np.empty(
      (0, xyz_start.shape[1]), dtype=xyz_start.dtype)
  parts_results = {
      "Pm": Pcd_parts,
      "Pi": xyz_start.astype(np.float32, copy=False),
      "parts_idx": parts_idx_list,
      "motions": motions_list,
      "Pj": xyz_end.astype(np.float32, copy=False),
      "Ps": Ps.astype(np.float32, copy=False),
      "part_ids": part_ids,
      "ArtModel": am,
  }

  return parts_results


def merge_subparts(out_path: str, num_movable: int = 0) -> int:
  if num_movable == 0:
    num_movable = get_K(out_path)

  part_indices = np.load(os.path.join(out_path, 'part_indices_pre.npy'))
  r = np.load(os.path.join(out_path, 'r_pre.npy'))
  t = np.load(os.path.join(out_path, 't_pre.npy'))
  axes = np.load(os.path.join(out_path, 'clustering/axes.npy'))
  bb_centers = np.load(os.path.join(out_path, 'clustering/bb_centers.npy'))
  bb_extents = np.load(os.path.join(out_path, 'clustering/bb_extents.npy'))

  volume_ratios, _ = get_obb_proximity_matrix(axes, bb_centers,
                                              bb_extents * 1.05)
  tr_close = get_tr_proximity_matrix(r, t)
  min_angles = get_minimum_angles(axes, r, t)
  print(volume_ratios)
  print(tr_close)
  print(min_angles)

  ds = DisjointSet(num_movable)
  for k in range(num_movable):
    if min_angles[k] > 2:
      j = np.argmax(volume_ratios[k])
      if min_angles[j] < 2:
        ds.connect(k, j)
        print('merge:', k, j)
      continue

    j, obb_max = -1, 0.04
    for i in range(num_movable):
      if i == k or not tr_close[k][i] or ds.is_connected(k, i):
        continue
      if volume_ratios[k][i] > obb_max:
        j, obb_max = i, volume_ratios[k][i]

    if j != -1:
      if min_angles[j] < min_angles[k]:
        ds.connect(k, j)
      else:
        ds.connect(j, k)
      print('merge:', k, j)

  merge_indices, uniques = ds.get_new_indices()
  new_part_indices = np.zeros_like(part_indices, dtype=int)
  for k in range(num_movable):
    new_part_indices[part_indices == k] = merge_indices[ds.parent[k]]
  new_r, new_t = [], []
  for idx in uniques:
    new_r.append(r[idx])
    new_t.append(t[idx])

  np.save(os.path.join(out_path, '_r_pre.npy'), new_r)
  np.save(os.path.join(out_path, '_t_pre.npy'), new_t)
  np.save(os.path.join(out_path, '_part_indices_pre'), new_part_indices)
  return len(uniques)


def vis_axes_pp_demo(out_path: str,
                     cluster_folder: str,
                     st_path: str,
                     ply_folder: str = 'ply',
                     mu=None,
                     arrow_scale=1.0):
  ply_path = os.path.join(out_path, cluster_folder, ply_folder)
  os.makedirs(ply_path, exist_ok=True)
  # axes
  print(f'loading transforms from {os.path.join(out_path, "trans_pred.json")}')
  with open(os.path.join(out_path, 'trans_pred.json'), 'r') as json_file:
    trans = json.load(json_file)
  if mu is None:
    mu = np.load(os.path.join(out_path, 'mu_init.npy'))
  print(trans)
  print(mu)
  for i, trans_info in enumerate(trans):
    o = np.array(trans_info['axis']['o'])
    d = np.array(trans_info['axis']['d'])
    save_axis_mesh(d,
                   o,
                   os.path.join(ply_path, f'axis_{i}.ply'),
                   mu[i],
                   arrow_scale=arrow_scale)
  try:
    # pcd seg
    num_movable = len(mu)
    mask = np.load(os.path.join(out_path, 'mask_pre.npy'))
    part_indices = np.load(os.path.join(out_path, 'part_indices_pre.npy'))
    gaussians_st = get_gaussians(st_path, from_chk=True,
                                 from_pgsr=from_pgsr).cancel_grads()
    xyz = gaussians_st.get_xyz.detach().cpu().numpy()
    rgb = np.full(xyz.shape, 255)
    for k in np.arange(num_movable):
      rgb[(part_indices == k) & mask] = np.array(COLORS[k % len(COLORS)]) * 255
    storePly(os.path.join(ply_path, 'seg.ply'), xyz, rgb)
  except Exception as e:
    print('vis pcd seg failed:', e)


def refinement_demo(out_path: str,
                    st_path: str,
                    data_path: str,
                    num_movable: int = 0,
                    refine_depth_weight: float = None,
                    bws_st=None,
                    bws_ed=None):
  if num_movable == 0:
    num_movable = get_K(out_path)

  torch.autograd.set_detect_anomaly(False)
  print("[run] Load gaussians from:", st_path)
  gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr)
  amj = MPArtModelJoint(gaussians_st, num_movable)
  amj.set_dataset(source_path=os.path.realpath(data_path), model_path=out_path)
  amj.opt.depth_weight = refine_depth_weight
  t, r = amj.train(bws_st=bws_st, bws_ed=bws_ed)

  mask, part_indices = amj.canonical_gaussians.save_ply(
      os.path.join(out_path, f'point_cloud/iteration_99999/point_cloud.ply'),
      prune=True,
      auxiliary_attr=(amj.mask, amj.part_indices))
  gaussians_canonical = get_gaussians(out_path, from_chk=False, iters=99999)
  part_pcd = []
  for i in range(num_movable):
    gaussians_canonical[mask & (part_indices == i)].save_ply(
        os.path.join(out_path,
                     f'point_cloud/iteration_{31 + i}/point_cloud.ply'))
    part_pcd.append(
        gaussians_canonical[mask &
                            (part_indices == i)].get_xyz.detach().cpu().numpy())
  gaussians_canonical[~mask].save_ply(
      os.path.join(out_path, 'point_cloud/iteration_30/point_cloud.ply'))
  gaussians_canonical[part_indices == 0].save_ply(
      os.path.join(out_path, 'point_cloud/iteration_29/point_cloud.ply'))
  static_part = gaussians_canonical[~mask].get_xyz.detach().cpu().numpy()
  np.save(os.path.join(out_path, 't_final.npy'),
          [tt.detach().cpu().numpy() for tt in t])
  np.save(os.path.join(out_path, 'r_final.npy'),
          [rr.detach().cpu().numpy() for rr in r])
  np.save(os.path.join(out_path, 'mask_final.npy'), mask.detach().cpu().numpy())
  np.save(os.path.join(out_path, 'part_indices_final'),
          part_indices.detach().cpu().numpy())
  return {"Ps": static_part, "Pm": part_pcd}


def eval_demo(out_path: str,
              data_path: str,
              num_movable: int = 0,
              gt_num_movable: int = 0,
              reverse=True,
              iters=30):
  if num_movable == 0:
    num_movable = get_K(out_path)

  if iters == 30:
    t = np.load(os.path.join(out_path, 't_final.npy'))
    r = np.load(os.path.join(out_path, 'r_final.npy'))
  elif iters == 20:
    t = np.load(os.path.join(out_path, 't_pre.npy'))
    r = np.load(os.path.join(out_path, 'r_pre.npy'))
  else:
    print('wrong iters:', iters)
    return

  print(t)
  print(r)

  trans_pred = interpret_transforms(t, r)
  with open(os.path.join(out_path, 'trans_pred.json'), 'w') as outfile:
    json.dump(trans_pred, outfile, indent=4)

  if num_movable != gt_num_movable:
    print(
        f'Warning: num_movable ({num_movable}) != gt_num_movable ({gt_num_movable})'
    )
    # return

  with open(os.path.join(data_path, 'trans.json'), 'r') as json_file:
    trans = json.load(json_file)
  trans_gt = trans['trans_info']
  if isinstance(trans_gt, dict):
    trans_gt = [trans_gt]

  pcd_pred = get_pred_point_cloud(out_path, K=num_movable, iters=iters)
  pcd_gt = get_gt_point_clouds(os.path.join(data_path, 'gt/'),
                               K=num_movable,
                               reverse=reverse)

  metrics_axis = eval_axis_metrics(trans_pred,
                                   trans_gt,
                                   reverse=reverse,
                                   out_path=out_path)
  metrics_axis_stat = stat_axis_metrics(metrics_axis)
  metrics_cd = eval_geo_metrics(pcd_pred, pcd_gt)
  with open(os.path.join(out_path, 'metrics.json'), 'w') as outfile:
    json.dump(metrics_axis | metrics_axis_stat | metrics_cd, outfile, indent=4)
