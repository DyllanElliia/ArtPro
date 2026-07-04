import os
import json
import numpy as np

import torch
from pytorch3d.loss import chamfer_distance
from render import render_depth_for_pcd
from utils.general_utils import get_rotation_axis, rotation_matrix_from_axis_angle, sample_points_from_ply
from utils.loss_utils import sample_pts
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import get_pcd_from_depths
from scene.dataset_readers import storePly, fetchPly


def get_gt_motion_params(data_path: str, reverse=False):
  with open(os.path.join(data_path, 'trans.json'), 'r') as json_file:
    trans = json.load(json_file)
  trans_infos = trans['trans_info']
  if isinstance(trans_infos, dict):
    trans_infos = [trans_infos]
  for trans_info in trans_infos:
    r = np.eye(3)
    if trans_info['type'] == 'translate':
      direction = np.array(trans_info['axis']['d'])
      distance = trans_info['translate']['r'] - trans_info['translate']['l']
      t = direction * distance
    elif trans_info['type'] == 'rotate':
      c = np.array(trans_info['axis']['o'])
      n = np.array(trans_info['axis']['d'])
      theta = (trans_info['rotate']['r'] -
               trans_info['rotate']['l']) / 180 * np.pi
      r = rotation_matrix_from_axis_angle(n, theta)
      t = c - r @ c
      r = r.T
    else:
      assert False
    print('t:', t if not reverse else -r.T @ t)
    print('r:', r if not reverse else r.T)


def interpret_transforms(translations: np.ndarray,
                         rotations: np.ndarray) -> list[dict]:
  results = []
  for t, r in zip(translations, rotations):
    r = r.T
    trans_info = {}
    theta = np.arccos((np.trace(r) - 1) / 2)
    if theta < 0.15:
      trans_info['type'] = 'translate'
      trans_info['axis'] = {
          'o': [0., 0., 0.],
          'd': (t / np.linalg.norm(t)).tolist()
      }
      trans_info['translate'] = float(np.linalg.norm(t))
    else:
      o, d = get_rotation_axis(r, t)
      trans_info['type'] = 'rotate'
      trans_info['axis'] = {'o': o.tolist(), 'd': d.tolist()}
      trans_info['rotate'] = float(np.rad2deg(theta))

      r_chk = rotation_matrix_from_axis_angle(d, theta)
      if np.trace(np.dot(r.T, r_chk)) < 2.99792458:
        print('reversed:', np.trace(np.dot(r.T, r_chk)))
        trans_info['rotate'] = -trans_info['rotate']
    results.append(trans_info)
  return results


def sort_trans_gt(trans_pred: list[dict], trans_gt: list[dict],
                  reverse: bool) -> list[dict]:

  def eval_diff(pred_info: dict, gt_info: dict):
    difference = None
    if pred_info['type'] == gt_info['type'] and gt_info['type'] == 'rotate':
      # o_diff = np.array(pred_info['axis']['o']) - np.array(gt_info['axis']['o'])
      # o_diff /= np.linalg.norm(o_diff)
      # d_gt = np.array(gt_info['axis']['d'])
      # d_gt /= np.linalg.norm(d_gt)
      # angle_oo_axis = np.rad2deg(np.arccos(np.abs(o_diff @ d_gt)))
      # difference = angle_oo_axis + np.abs(abs(pred_info['rotate']) - abs(gt_info['rotate']))
      d_gt = np.array(gt_info['axis']['d'])
      d_pred = np.array(pred_info['axis']['d'])
      o_gt = np.array(gt_info['axis']['o'])
      o_pred = np.array(pred_info['axis']['o'])
      theta_gt = np.deg2rad(
          gt_info['rotate'] if not reverse else -gt_info['rotate'])
      theta_pred = np.deg2rad(pred_info['rotate'])
      r_diff = np.matmul(
          rotation_matrix_from_axis_angle(d_gt / np.linalg.norm(d_gt),
                                          theta_gt),
          rotation_matrix_from_axis_angle(d_pred / np.linalg.norm(d_pred),
                                          theta_pred).T)
      rot_diff = np.rad2deg(
          np.arccos(np.clip((np.trace(r_diff) - 1.0) * 0.5, a_min=-1, a_max=1)))
      pos_diff = line_distance(o_pred, d_pred, o_gt, d_gt)
      difference = rot_diff + pos_diff * 10
    if pred_info['type'] == gt_info['type'] and gt_info['type'] == 'translate':
      d_gt = np.array(gt_info['axis']['d'])
      d_pred = np.array(pred_info['axis']['d'])
      if reverse:
        d_gt *= -1
      # angle = np.arccos(d_gt @ d_pred)
      # arc_len = angle * gt_info['translate']
      # difference = arc_len + np.abs(abs(pred_info['translate']) - abs(gt_info['translate']))
      l_gt = gt_info['translate']
      l_pred = pred_info['translate']
      difference = np.linalg.norm(d_gt * l_gt - d_pred * l_pred)
    assert difference is not None
    return difference

  trans_gt_sorted = []
  org_trans_gt = trans_gt.copy()
  for pred in trans_pred:
    rm = None
    diff_min = 514
    for gt in trans_gt:
      if gt['type'] == pred['type']:
        diff = eval_diff(pred, gt)
        if diff < diff_min:
          diff_min = diff
          rm = gt
    trans_gt_sorted.append(rm)
    if rm is not None:
      trans_gt.remove(rm)
      if len(trans_gt) == 0:
        # print('[warning] gt axes less than pred axes!')
        trans_gt = org_trans_gt.copy()
  return trans_gt_sorted


def line_distance(a_o, a_d, b_o, b_d):
  normal = np.cross(a_d, b_d)
  normal_length = np.linalg.norm(normal)
  if normal_length < 1e-6:  # parallel
    return np.linalg.norm(np.cross(b_o - a_o, a_d))
  else:
    return np.abs(np.dot(normal, a_o - b_o)) / normal_length


def eval_axis_and_state(axis_a, axis_b, reverse):
  a_d, b_d = np.array(axis_a['axis']['d']), np.array(axis_b['axis']['d'])
  a_d /= np.linalg.norm(a_d)
  b_d /= np.linalg.norm(b_d)
  angle = np.rad2deg(
      np.arccos(np.dot(a_d, b_d) / np.linalg.norm(a_d) / np.linalg.norm(b_d)))
  angle = min(angle, 180 - angle)

  if axis_a['type'] == 'rotate':
    a_o, b_o = np.array(axis_a['axis']['o']), np.array(axis_b['axis']['o'])
    distance = line_distance(a_o, a_d, b_o, b_d)

    a_theta, b_theta = np.deg2rad(axis_a['rotate']), np.deg2rad(
        axis_b['rotate'])
    if reverse:
      b_theta = -b_theta
    # if (a_theta * a_d) @ (b_theta * b_d) < 0:
    #     b_theta = -b_theta
    a_r, b_r = rotation_matrix_from_axis_angle(
        a_d, a_theta), rotation_matrix_from_axis_angle(b_d, b_theta)
    r_diff = np.matmul(a_r, b_r.T)
    state = np.rad2deg(
        np.arccos(np.clip((np.trace(r_diff) - 1.0) * 0.5, a_min=-1, a_max=1)))
  else:
    distance = 0
    a_td, b_td = axis_a['translate'], axis_b['translate']
    a_t, b_t = a_td * a_d, b_td * b_d
    # if a_t @ b_t < 0:
    #     b_t = -b_t
    if reverse:
      b_t = -b_t
    state = np.linalg.norm(a_t - b_t)

  return angle, distance, state


def eval_axis_metrics(trans_pred: list[dict],
                      trans_gt: list[dict],
                      reverse: bool,
                      out_path=None) -> dict:
  for info in trans_gt:
    info[info['type']] = info[info['type']]['r'] - info[info['type']]['l']
  trans_gt = sort_trans_gt(trans_pred, trans_gt, reverse)
  if out_path is not None:
    with open(os.path.join(out_path, 'trans_gt.json'), 'w') as outfile:
      json.dump(trans_gt, outfile, indent=4)

  metric_dict = {'axes': []}
  for pred, gt in zip(trans_pred, trans_gt):
    if gt is None:
      metric_dict['axes'].append({
          'axis_angle': -1,
          'axis_dist': -1,
          'theta_diff': -1
      })
      continue
    angle, distance, state = eval_axis_and_state(pred, gt, reverse)
    metric_dict['axes'].append({
        'axis_angle': angle,
        'axis_dist': distance * 10,
        'theta_diff': state
    })
  return metric_dict


def stat_axis_metrics(metric_axis: dict) -> dict:
  metrics = metric_axis['axes']
  stat = {'axes_stat': {}}
  axis_angles = np.array([item['axis_angle'] for item in metrics])
  axis_dists = np.array([item['axis_dist'] for item in metrics])
  theta_diffs = np.array([item['theta_diff'] for item in metrics])
  stat['axes_stat']['mean'] = {
      "axis_angle": float(np.mean(axis_angles)),
      "axis_dist": float(np.mean(axis_dists)),
      "theta_diff": float(np.mean(theta_diffs)),
  }
  stat['axes_stat']['max'] = {
      "axis_angle": float(np.max(axis_angles)),
      "axis_dist": float(np.max(axis_dists)),
      "theta_diff": float(np.max(theta_diffs)),
  }
  return stat


############  axes 👆🏻     ############
############  geometry 👇🏻 ############


def get_gaussian_surface_pcd(model_path: str,
                             it: int,
                             n_samples: int = 10_000) -> np.ndarray:

  def from_gaussian_xyz():
    xyz = GaussianModel(0).load_ply(
        os.path.join(
            model_path,
            f'point_cloud/iteration_{it}/point_cloud.ply')).get_xyz.detach()
    return sample_pts(xyz, n_samples).cpu().numpy()

  def from_depth():
    pre_xyz = os.path.join(model_path, f'depth/ours_{it}.npy')
    try:
      xyz = np.load(pre_xyz)
      assert len(xyz) == n_samples
    except:
      print(f'\nProcessing part {it}:')
      cam_list = render_depth_for_pcd(model_path, it)
      xyz, _, _ = get_pcd_from_depths(cam_list, num_pts=n_samples)
      np.save(pre_xyz, xyz)
      storePly(pre_xyz.replace('.npy', '.ply'), xyz, np.zeros_like(xyz))
    return xyz

  def from_pgsr_mesh():
    return sample_points_from_ply(
        os.path.join(model_path, f'mesh/tsdf_fusion_{it}.ply'), n_samples)

  return from_gaussian_xyz()
  # return from_pgsr_mesh()


def get_pred_point_cloud(model_path: str,
                         iters=30,
                         K=1,
                         n_samples: int = 100_000) -> dict:
  static = get_gaussian_surface_pcd(model_path, iters, n_samples)
  movables = []
  for i in range(1, K + 1):
    movables.append(get_gaussian_surface_pcd(model_path, iters + i, n_samples))
  movable = np.concatenate(movables, axis=0)
  whole = get_gaussian_surface_pcd(model_path, 99999, n_samples)
  # whole = get_gaussian_surface_pcd(model_path, 15000, n_samples)
  return {
      'movables': movables,
      'movable': movable,
      'static': static,
      'whole': whole
  }


def get_gt_point_clouds(gt_dir: str,
                        K=1,
                        n_samples: int = 100_000,
                        reverse=True) -> dict:
  state = 'end' if reverse else 'start'
  whole = sample_points_from_ply(
      os.path.join(gt_dir, f'{state}/{state}_rotate.ply'), n_samples)
  static = sample_points_from_ply(
      os.path.join(gt_dir, f'{state}/{state}_static_rotate.ply'), n_samples)
  movables = []
  if K == 1:
    movable = sample_points_from_ply(
        os.path.join(gt_dir, f'{state}/{state}_dynamic_rotate.ply'), n_samples)
    movables.append(movable)
  else:
    try:
      j = 0
      for i in range(K):
        ply_file = os.path.join(gt_dir,
                                f'{state}/{state}_dynamic_{j}_rotate.ply')
        while not os.path.exists(ply_file):
          j += 1
          ply_file = os.path.join(gt_dir,
                                  f'{state}/{state}_dynamic_{j}_rotate.ply')
          if j > 99:
            break
        j += 1
        if j > 99:
          break
        movables.append(sample_points_from_ply(ply_file, n_samples))
      movable = np.concatenate(movables, axis=0)
    except:
      print(
          '[warning] failed to load multiple dynamic parts, load single part instead.'
      )
      movable = sample_points_from_ply(
          os.path.join(gt_dir, f'{state}/{state}_dynamic_rotate.ply'),
          n_samples)
      movables.append(movable)
  return {
      'movables': movables,
      'movable': movable,
      'static': static,
      'whole': whole
  }


def compute_chamfer_helper(recon_pts: np.ndarray, gt_pts: np.ndarray):
  recon_pts = torch.tensor(recon_pts, device='cuda',
                           dtype=torch.float).unsqueeze(0)
  gt_pts = torch.tensor(gt_pts, device='cuda', dtype=torch.float).unsqueeze(0)
  dist, _ = chamfer_distance(recon_pts, gt_pts, batch_reduction=None)
  dist = dist.item()
  return dist


def compute_chamfer(recon_pts: np.ndarray, gt_pts: np.ndarray) -> float:
  return (compute_chamfer_helper(recon_pts, gt_pts) +
          compute_chamfer_helper(gt_pts, recon_pts)) * 0.5 * 1000


def eval_geo_metrics(pred_pc: dict, gt_pc: dict) -> dict:
  metric_dict = {'chamfer_dynamics': []}
  for pred in pred_pc['movables']:
    min_cd = np.inf
    for gt in gt_pc['movables']:
      cd = compute_chamfer(pred, gt)
      min_cd = min(min_cd, cd)
    metric_dict['chamfer_dynamics'].append(min_cd)

  metric_dict['chamfer_dynamics_avg'] = np.array(
      metric_dict['chamfer_dynamics']).mean().tolist()
  metric_dict['chamfer_dynamic'] = compute_chamfer(pred_pc['movable'],
                                                   gt_pc['movable'])
  metric_dict['chamfer_static'] = compute_chamfer(pred_pc['static'],
                                                  gt_pc['static'])
  metric_dict['chamfer_whole'] = compute_chamfer(pred_pc['whole'],
                                                 gt_pc['whole'])
  return metric_dict
