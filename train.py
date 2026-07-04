from main import *
from os.path import join as pjoin

import shutil
import open3d as o3d
import json

from auto_merge.part_seg_utils import *
from auto_merge.tiny_main_utils import *
from auto_merge.merge_utils import *
from auto_merge.cal_trans_utils import best_to_art_Rct, best_to_o3d_Rt, estimate_constrained_motion_art, apply_motion_scale_to_rct
from auto_merge.auto_seg import debug_visual
from scene import BWScenes
from arguments import GroupParams


def set_seed(seed):
  import random
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def clear_folder(folder_path):
  if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')
  else:
    os.makedirs(folder_path, exist_ok=True)


def del_folder(folder_path):
  if os.path.exists(folder_path):
    print(f"Deleting folder: {folder_path}")
    shutil.rmtree(folder_path)


def init_stage(data_src_path,
               output_path,
               tau=-1,
               seg_knn=24,
               segment_level: str = "coarse"):
  cluster_init_path = os.path.join(output_path, 'clustering_0')
  for i in range(1, 20):
    del_folder(os.path.join(output_path, f'clustering_{i}'))

  # if normal pcd exists, load it
  normals_path = os.path.join(data_src_path, 'start', 'points3d_normals.ply')
  if os.path.exists(normals_path):
    print(f"[process] load existing normals from {normals_path}...")
    Pi = o3d.io.read_point_cloud(normals_path)
  else:
    Pi = o3d.io.read_point_cloud(
        os.path.join(data_src_path, 'start', 'points3d.ply'))
  Pj = o3d.io.read_point_cloud(
      os.path.join(data_src_path, 'end', 'points3d.ply'))
  print(f"[process] ensure normals...")
  has_valid_normals = False
  if Pi.has_normals():
    normals_arr = np.asarray(Pi.normals)
    mean_len = np.mean(np.linalg.norm(normals_arr, axis=1))
    print(f"  normal count: {len(normals_arr)}, mean length: {mean_len:.4f}")
    has_valid_normals = mean_len > 0.5  # 有效法向量长度应接近 1
  if not has_valid_normals:
    raise ValueError(
        f"Input point cloud normals are missing or invalid. Please provide valid normals or enable normal recomputation.\n data path: {data_src_path}"
    )
  else:
    print(f"[process] normals are valid.")
  # Pj = ensure_normals(Pj, max_nn=30, orient="consistent", k_consistent=30)
  Pi_normal = np.asarray(Pi.normals)
  # Pj_normal = np.asarray(Pj.normals)
  Pi = np.asarray(Pi.points, dtype=np.float32)
  Pj = np.asarray(Pj.points, dtype=np.float32)

  print(f"[process] load part features...")
  feat_path = os.path.join(data_src_path, 'start', 'partfield_features',
                           "part_feat_points3d.npy")
  if not os.path.exists(feat_path):
    print(f"[process] PartField features not found, running extraction...")
    from module.partfield_module import PartFieldModule
    _ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'checkpoint', 'model_objaverse.ckpt')
    _pf = PartFieldModule(checkpoint_path=_ckpt, device="cuda")
    feat_i = _pf.extract_features({"points": Pi})
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    np.save(feat_path, feat_i)
    print(f"[process] saved features to {feat_path}")
    del _pf
    torch.cuda.empty_cache()
  else:
    print(f"[process] loading features from {feat_path}...")
    feat_i = np.load(feat_path)
  segment_levels = {
      "fine": {
          "sim_hi": 0.85,
          "sim_mid": 0.80,
          "sim_low": 0.80
      },
      "medium": {
          "sim_hi": 0.80,
          "sim_mid": 0.75,
          "sim_low": 0.75
      },
      "default": {
          "sim_hi": 0.75,
          "sim_mid": 0.70,
          "sim_low": 0.70
      },
      "coarse": {
          "sim_hi": 0.70,
          "sim_mid": 0.65,
          "sim_low": 0.65
      },
  }
  print(
      f"[process] segment level: {segment_level}, seg params: {segment_levels[segment_level]}"
  )
  labels, parts, info = adaptive_part_segmentation_torch(
      Pi,
      Pj,
      feat_i,
      tau=tau,
      use_sigmoid_dist_norm=True,
      use_random_seeds=True,
      random_seed_count=100,
      sim_hi=segment_levels[segment_level]["sim_hi"],
      sim_mid=segment_levels[segment_level]["sim_mid"],
      sim_low=segment_levels[segment_level]["sim_low"],
      knn_k=seg_knn,
  )

  clear_folder(cluster_init_path)
  os.makedirs(os.path.join(cluster_init_path, 'clusters'), exist_ok=True)
  os.makedirs(os.path.join(cluster_init_path, 'debug'), exist_ok=True)
  debug_path = os.path.join(cluster_init_path, 'debug')
  debug_visual(Pi, labels, parts, info, debug_path, 'auto_seg', 'start')
  init_num_movable = len(parts)
  parts_pcd = []
  part_mask = np.zeros(len(Pi), dtype=bool)
  for pid, idx in enumerate(parts):
    idx = np.asarray(idx, dtype=int)
    part_mask[idx] = True
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pi[idx])
    pcd.normals = o3d.utility.Vector3dVector(Pi_normal[idx])
    o3d.io.write_point_cloud(
        os.path.join(cluster_init_path, 'clusters', f'points3d_{pid}.ply'), pcd)

    parts_pcd.append(Pi[idx])
  print(f"[process] init parts count: {init_num_movable}")
  static_idx = np.nonzero(~part_mask)[0]
  Ps = Pi[static_idx] if static_idx.size else np.empty(
      (0, Pi.shape[1]), dtype=Pi.dtype)
  Ps_o3d = o3d.geometry.PointCloud()
  Ps_o3d.points = o3d.utility.Vector3dVector(Ps)
  o3d.io.write_point_cloud(
      os.path.join(cluster_init_path, 'debug', 'points3d_static.ply'), Ps_o3d)
  pkg_ret = {
      'Pi': Pi,
      'Pj': Pj,
      'parts_idx': parts,
      'Pm': parts_pcd,
      'Ps': Ps,
  }
  # save pkg_ret to cluster_init_path/pkg_ret.npy
  # np.save(os.path.join(cluster_init_path, 'pkg_ret.npy'), pkg_ret)
  return init_num_movable, "clustering_0", pkg_ret


def merge_options(parts_results,
                  rel_tol=0.01,
                  merge_knn=4,
                  merge_method="v3",
                  am=None,
                  bws=None,
                  tau_merge=1e-2):
  parts_points, parts_idx, motions = filter_small_parts(
      parts_results['Pm'],
      parts_results['parts_idx'],
      parts_results['motions'],
      min_points=30)
  # parts_points, parts_idx, motions = filter_keep_largest_cluster(
  #     parts_points, parts_idx, motions, min_cluster_size=20)

  parts_points, parts_idx, motions = filter_static_parts_v3(
      parts_points,
      parts_idx,
      motions,
      translation_thresh=0.001,
      rotation_thresh=np.deg2rad(5.0))
  merge_records = None
  if merge_method == "both" and am is not None and bws is not None:
    # Depth occlusion causes the depth map-based merging operation to be not robust.
    # To solve this problem, we use cross-validation based on CD and depth to improve the robustness of the merging operation.
    parts_points, parts_idx, motions, merge_records, _ = neighbor_motion_merge_w_both_depth_cd(
        parts_points,
        parts_results['Pi'],
        parts_idx,
        motions,
        parts_results['Pj'],
        am,
        bws,
        neighbor_k=merge_knn,
        rel_tol=rel_tol,
        tau_merge=tau_merge,
    )
  else:
    parts_points, parts_idx, motions, merge_records, _ = neighbor_motion_merge_v3(
        parts_points,
        parts_results['Pi'],
        parts_idx,
        motions,
        parts_results['Pj'],
        neighbor_k=merge_knn,
        rel_tol=rel_tol,
    )
  parts_points, parts_idx, motions = filter_keep_largest_cluster(
      parts_points, parts_idx, motions, min_cluster_size=20)
  return parts_points, parts_idx, motions, merge_records, True


def run_steps(output_path,
              data_src_path,
              pretrain_st_path,
              pretrain_ed_path,
              num_movable,
              reverse,
              args={}):
  # use the same seed for all examples
  set_seed(42)
  gt_num_movable = num_movable
  # num_movable = 0
  print(
      f"[start] estimate art params of {data_src_path} with {gt_num_movable} parts."
  )
  print(f"  output path: {output_path}")
  print(f"  additional args: {args}")
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

  if not os.path.exists(os.path.join(pretrain_st_path, "chkpnt.pth")):
    train_single_demo(pretrain_st_path, os.path.join(data_src_path, 'start'))
    train_single_demo(pretrain_ed_path,
                      os.path.join(data_src_path, 'end'),
                      skip_train=True)

  # Pre-load BWScenes once to avoid reloading camera images in each cycle/refine
  def _make_bws(source_path, model_path, eval_flag):
    ds = GroupParams()
    ds.sh_degree = 0
    ds.images = "images"
    ds.resolution = -1
    ds.white_background = False
    ds.data_device = "cuda"
    ds.eval = eval_flag
    ds.source_path = source_path
    ds.model_path = model_path
    print(
        f"[make_bws] source_path: {source_path}, model_path: {model_path}, eval: {eval_flag}"
    )
    return BWScenes(ds, GaussianModel(0), is_new_gaussians=False)

  _data_real = os.path.realpath(data_src_path)
  bws_gmm = _make_bws(os.path.join(_data_real, 'end'),
                      output_path,
                      eval_flag=True)
  bws_joint_st = _make_bws(os.path.join(_data_real, 'start'),
                           output_path,
                           eval_flag=False)
  bws_joint_ed = _make_bws(os.path.join(_data_real, 'end'),
                           output_path,
                           eval_flag=False)
  print("[run_steps] Pre-loaded BWScenes for GMM and Joint refinement.")

  num_movable, cluster_folder, pkg_ret = init_stage(
      data_src_path,
      output_path,
      tau=args.tau,
      seg_knn=24,
      segment_level=args.segment_level)
  parts_points, parts_idx, motions = pkg_ret['Pm'], pkg_ret['parts_idx'], None
  Ps, Pj = pkg_ret['Ps'], pkg_ret['Pj']

  has_merge = True
  cur_loop = 0
  last_motion = []
  while has_merge:
    has_merge = False
    part_init_demo(output_path,
                   pretrain_st_path,
                   pretrain_ed_path,
                   cluster_path=cluster_folder,
                   num_movable=num_movable)
    bb_center, bb_axis, bb_extent = joint_init_demo(output_path,
                                                    pretrain_st_path,
                                                    cluster_path=cluster_folder,
                                                    num_movable=num_movable)
    # add motion initialization
    omega_limit = args.omega_limit
    translation_limit = args.translation_limit
    motion_scale = args.motion_scale
    last_motion_scale = args.last_motion_scale
    print(
        f"[process] motion init params: omega_limit={omega_limit}, translation_limit={translation_limit}, motion_scale={motion_scale}, last_motion_scale={last_motion_scale}"
    )

    motions = []
    pcd_init_transforms = []
    R_init, c_init, t_init = [], [], []
    r_json, t_json = [], []
    for i in tqdm(range(num_movable),
                  desc="Estimating initial motions for parts"):
      if cur_loop > 0 and last_motion_scale >= 0:
        R, c, t = last_motion[i]['R'], last_motion[i]['c'], last_motion[i]['t']
        R, c, t = apply_motion_scale_to_rct(R, c, t, last_motion_scale)
        R_vis, c_vis, t_vis = last_motion[i]['R'], last_motion[i][
            'c'], last_motion[i]['t']
      else:
        u0, u1, u2 = bb_axis[i]
        _, _, best = estimate_constrained_motion_art(
            parts_points[i],
            Ps,
            Pj,
            [u0, u1, u2],
            bb_center[i],
            bb_extent[i],
            omega_range=(-omega_limit * np.pi, omega_limit * np.pi),
            max_trans=translation_limit,
            refine=False,
            device="cuda",
        )
        R, c, t = best_to_art_Rct(best, motion_scale)
        R_vis, c_vis, t_vis = best_to_art_Rct(best, 1.0)
        torch.cuda.empty_cache()
      motions.append({'R': R, 't': t, 'c': c})
      R_init.append(R)
      c_init.append(c)
      t_init.append(t)
      pcd_init_transforms.append((parts_points[i] - c_vis) @ R_vis + c_vis +
                                 t_vis)
      r_json.append(R)
      t_json.append(-c @ R + c + t)

      # R_o, t_o = best_to_o3d_Rt(best, 1.0)
      # print(R)
      # print(R_o.T)
      # print(-c @ R + c + t)
      # print(t_o)
    trans_pred = interpret_transforms(t_json, r_json)
    with open(os.path.join(output_path, 'trans_pred.json'), 'w') as outfile:
      print(
          f"Writing transforms to {os.path.join(output_path, 'trans_pred.json')}..."
      )
      json.dump(trans_pred, outfile, indent=4)
    vis_axes_pp_demo(output_path,
                     cluster_folder,
                     pretrain_st_path,
                     ply_folder='init_ply',
                     arrow_scale=args.arrow_scale)
    # return
    visualize_pcd_parts(
        os.path.join(output_path, cluster_folder, 'debug',
                     "test_parts_init_transformed.ply"), pcd_init_transforms,
        Pj)
    np.save(os.path.join(output_path, 'R_init.npy'), R_init)
    np.save(os.path.join(output_path, 'c_init.npy'), c_init)
    np.save(os.path.join(output_path, 't_init.npy'), t_init)
    pcd_a = o3d.geometry.PointCloud()
    c_init = np.stack(c_init, axis=0)
    print("c_init shape:", c_init.shape, c_init)
    pcd_a.points = o3d.utility.Vector3dVector(c_init)
    o3d.io.write_point_cloud(
        os.path.join(output_path, cluster_folder, 'debug',
                     "test_parts_centers.ply"), pcd_a)

    # get_gt_motion_params(data_src_path, reverse=reverse)

    depth_weight = args.depth_weight
    if cur_loop != 0:
      depth_weight = 4.0
    parts_results = art_optim_demo(
        output_path,
        pretrain_st_path,
        pretrain_ed_path,
        data_src_path,
        cluster_folder=cluster_folder,
        num_movable=num_movable,
        gt_num=gt_num_movable,
        obb_collision_delta=args.obb_collision_delta,
        obb_collision_freeze_rot=args.collision_freeze_rotation,
        depth_weight=depth_weight,
        bws=bws_gmm,
    )
    visualize_pcd_parts(
        os.path.join(output_path, cluster_folder, "test_parts.ply"),
        parts_results['Pm'])

    parts_p_transforms = []
    parts_points = parts_results['Pm']
    curr_pcd_path = os.path.join(output_path, cluster_folder, 'curr_pcd')
    clear_folder(curr_pcd_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(parts_results['Ps'])
    o3d.io.write_point_cloud(os.path.join(curr_pcd_path, f'part_0.ply'), pcd)
    for pid, part in enumerate(parts_points):
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(part)
      o3d.io.write_point_cloud(os.path.join(curr_pcd_path, f'part_{pid+1}.ply'),
                               pcd)

    motions = parts_results['motions']
    t_json, r_json = [], []
    for i, part in enumerate(parts_points):
      R, t = motions[i]['R_old'], motions[i]['t_old']
      r_json.append(R)
      t_json.append(t)
      parts_p_transforms.append(part @ R + t)
    trans_pred = interpret_transforms(t_json, r_json)
    with open(os.path.join(output_path, 'trans_pred.json'), 'w') as outfile:
      print(
          f"Writing transforms to {os.path.join(output_path, 'trans_pred.json')}..."
      )
      json.dump(trans_pred, outfile, indent=4)
    clear_folder(os.path.join(output_path, cluster_folder, 'curr_ply'))
    vis_axes_pp_demo(output_path,
                     cluster_folder,
                     pretrain_st_path,
                     ply_folder='curr_ply',
                     arrow_scale=args.arrow_scale)
    visualize_pcd_parts(
        os.path.join(output_path, cluster_folder,
                     "test_parts_rt_transformed.ply"), parts_p_transforms,
        parts_results['Pj'])

    parts_points, parts_idx, last_motion, merge_records, _ = merge_options(
        parts_results,
        rel_tol=args.rel_tol,
        merge_knn=8,
        # merge_method=args.get('merge_method', 'depth'),
        merge_method=args.merge_method,
        am=parts_results.get('ArtModel'),
        bws=bws_gmm,
        tau_merge=args.tau_merge,
    )
    # save merge records to json for debugging and analysis
    with open(os.path.join(output_path, cluster_folder, 'merge_records.json'),
              'w') as outfile:
      print(
          f"Writing merge records to {os.path.join(output_path, cluster_folder, 'merge_records.json')}..."
      )
      json.dump(merge_records, outfile, indent=4)

    Ps, Pj = parts_results['Ps'], parts_results['Pj']

    Ps_o3d = o3d.geometry.PointCloud()
    Ps_o3d.points = o3d.utility.Vector3dVector(Ps)
    o3d.io.write_point_cloud(
        os.path.join(output_path, cluster_folder, 'static_points.ply'), Ps_o3d)

    # has_merge = False
    visualize_pcd_parts(
        os.path.join(output_path, cluster_folder, "test_parts_merge.ply"),
        parts_points)
    parts_p_transforms = []
    r_json, t_json = [], []
    for i, part in enumerate(parts_points):
      R, t = last_motion[i]['R_old'], last_motion[i]['t_old']
      r_json.append(R)
      t_json.append(t)
      R, c, t = last_motion[i]['R'], last_motion[i]['c'], last_motion[i]['t']
      # parts_p_transforms.append(part @ R + t)
      parts_p_transforms.append((part - c) @ R + c + t)
    visualize_pcd_parts(
        os.path.join(output_path, cluster_folder,
                     "test_parts_merge_transformed.ply"), parts_p_transforms,
        Pj)

    trans_pred = interpret_transforms(t_json, r_json)
    with open(os.path.join(output_path, cluster_folder, 'trans_pred.json'),
              'w') as outfile:
      print(
          f"Writing transforms to {os.path.join(output_path, cluster_folder, 'trans_pred.json')}..."
      )
      json.dump(trans_pred, outfile, indent=4)
    with open(os.path.join(output_path, 'trans_pred.json'), 'w') as outfile:
      json.dump(trans_pred, outfile, indent=4)

    torch.cuda.empty_cache()
    vis_axes_pp_demo(output_path,
                     cluster_folder,
                     pretrain_st_path,
                     mu=[part.mean(axis=0) for part in parts_points],
                     arrow_scale=args.arrow_scale)
    try:
      eval_demo(output_path,
                data_src_path,
                num_movable=len(parts_points),
                gt_num_movable=gt_num_movable,
                reverse=reverse,
                iters=20)
      shutil.copyfile(
          os.path.join(output_path, 'metrics.json'),
          os.path.join(output_path, cluster_folder, 'metrics_pre.json'))
    except Exception as e:
      print(f"[Warning] evaluation failed: {e}")
    # if current parts count is less than previous, a merge has occurred.
    # update num_movable and continue next iteration.
    if len(parts_points) < num_movable:
      has_merge = True
      num_movable = len(parts_points)
      print(
          f"[process] after merge, new parts count: {num_movable}, continue to next iteration."
      )

      cluster_folder = f'clustering_{cur_loop+1}'
      clear_folder(os.path.join(output_path, cluster_folder))
      os.makedirs(os.path.join(output_path, cluster_folder, 'clusters'),
                  exist_ok=True)
      os.makedirs(os.path.join(output_path, cluster_folder, 'debug'),
                  exist_ok=True)
      cur_loop += 1
      for pid, part in enumerate(parts_points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part)
        pcd = ensure_normals(pcd,
                             max_nn=30,
                             orient="consistent",
                             k_consistent=30)
        o3d.io.write_point_cloud(
            os.path.join(os.path.join(output_path, cluster_folder), 'clusters',
                         f'points3d_{pid}.ply'), pcd)
      print(f"[process] saved merged parts to {cluster_folder}")
      # 清理上一轮的结果以释放显存，为下一轮做准备
      del parts_results
      torch.cuda.empty_cache()

    else:
      print(
          f"[process] no more merge possible, finish auto-merge with {num_movable} parts."
      )
      print(f"[process] final more optimization...")

      parts_results = art_optim_demo(
          output_path,
          pretrain_st_path,
          pretrain_ed_path,
          data_src_path,
          cluster_folder=cluster_folder,
          num_movable=num_movable,
          gt_num=gt_num_movable,
          obb_collision_delta=args.obb_collision_delta,
          obb_collision_freeze_rot=args.collision_freeze_rotation,
          continue_training=True,
          curr_iter=6000,
          iteration=4000,
          am=parts_results['ArtModel'],
          depth_weight=args.final_depth_weight,
          bws=bws_gmm)
      parts_points, parts_idx, last_motion = parts_results['Pm'], parts_results[
          'parts_idx'], parts_results['motions']
      visualize_pcd_parts(
          os.path.join(output_path, cluster_folder, "test_parts_final.ply"),
          parts_points)

      parts_p_transforms = []
      r_json, t_json = [], []
      for i, part in enumerate(parts_points):
        R, c, t = last_motion[i]['R'], last_motion[i]['c'], last_motion[i]['t']
        parts_p_transforms.append((part - c) @ R + c + t)
        R, t = last_motion[i]['R_old'], last_motion[i]['t_old']
        r_json.append(R)
        t_json.append(t)

      visualize_pcd_parts(
          os.path.join(output_path, cluster_folder,
                       "test_parts_final_transformed.ply"), parts_p_transforms,
          Pj)
      trans_pred = interpret_transforms(t_json, r_json)
      with open(os.path.join(output_path, 'trans_pred.json'), 'w') as outfile:
        print(
            f"Writing final transforms to {os.path.join(output_path, 'trans_pred.json')}..."
        )
        json.dump(trans_pred, outfile, indent=4)
      torch.cuda.empty_cache()

      print('Final evaluation:')
    print('-----------------------------------')

  # eval_demo(out, data, num_movable=num_movable, reverse=reverse, iters=20)

  refine_depth_weight = args.refine_depth_weight
  P_sm = refinement_demo(output_path,
                         pretrain_st_path,
                         data_src_path,
                         num_movable=num_movable,
                         refine_depth_weight=refine_depth_weight,
                         bws_st=bws_joint_st,
                         bws_ed=bws_joint_ed)

  P = [P_sm['Ps']] + P_sm['Pm']
  P_out_path = os.path.join(output_path, cluster_folder, 'pcd')
  os.makedirs(P_out_path, exist_ok=True)
  for i, part in enumerate(P):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(part)
    o3d.io.write_point_cloud(os.path.join(P_out_path, f'part_{i}.ply'), pcd)
  torch.cuda.empty_cache()

  try:
    eval_demo(output_path,
              data_src_path,
              num_movable=num_movable,
              gt_num_movable=gt_num_movable,
              reverse=reverse,
              iters=30)
  except Exception as e:
    print(f"[Warning] refinement evaluation failed: {e}")

  vis_axes_pp_demo(output_path,
                   cluster_folder,
                   pretrain_st_path,
                   arrow_scale=args.arrow_scale)
  try:
    if num_movable == gt_num_movable:
      # copy out_path/metrics.json to output_path/cluster_folder/metrics.json
      shutil.copyfile(os.path.join(output_path, 'metrics.json'),
                      os.path.join(output_path, cluster_folder, 'metrics.json'))
    else:
      shutil.copyfile(
          os.path.join(output_path, 'metrics.json'),
          os.path.join(output_path, cluster_folder, 'metrics_ba.json'))
  except Exception as e:
    print(f"[Warning] final metrics copy failed: {e}")
  print(f"[done] all steps finished for {data_src_path}.")
  print(f"[done] results saved in {output_path}.")


def build_arg_parser():
  import argparse
  parser = argparse.ArgumentParser(
      description="Estimate articulation params for a single model.")
  # paths / required
  parser.add_argument(
      '--source',
      required=True,
      help="data source path, e.g. ./datasets/paris/blade_103706")
  parser.add_argument(
      '--output',
      required=True,
      help="output path, e.g. output_for_time/paris/blade_103706")
  parser.add_argument('--gt_movable_part',
                      type=int,
                      required=True,
                      help="ground-truth number of movable parts (num_movable)")
  parser.add_argument('--reverse',
                      action='store_true',
                      help="reverse the metrics motion axis")
  # ----- hyper-parameters, grouped by the stage/module that consumes them -----
  # These defaults are read directly by run_steps via args.<name>.

  # [segmentation] init_stage(): adaptive part segmentation
  parser.add_argument(
      '--segment_level',
      type=str,
      default='default',
      help="segmentation granularity: fine|medium|default|coarse")
  parser.add_argument('--tau',
                      type=float,
                      default=-2.0,
                      help="segmentation tau (distance normalization)")

  # [motion init] estimate_constrained_motion_art() / best_to_art_Rct()
  parser.add_argument(
      '--omega_limit',
      type=float,
      default=0.40,
      help="rotation search bound, in units of pi (+/- omega_limit*pi)")
  parser.add_argument(
      '--translation_limit',
      type=float,
      default=None,
      help="max translation for motion search (None = unbounded)")
  parser.add_argument('--motion_scale',
                      type=float,
                      default=0.5,
                      help="scale applied to the estimated motion at init")
  parser.add_argument(
      '--last_motion_scale',
      type=float,
      default=0.8,
      help="motion scale retained from last-loop motion as init; "
      "<0 disables reuse and falls back to estimating motion at init")

  # [articulation optim] art_optim_demo(): joint/articulation optimization
  parser.add_argument('--depth_weight',
                      type=float,
                      default=2.0,
                      help="depth loss weight during optimization")
  parser.add_argument('--final_depth_weight',
                      type=float,
                      default=4.0,
                      help="depth loss weight for the final extra optimization")
  parser.add_argument('--obb_collision_delta',
                      type=float,
                      default=None,
                      help="OBB collision margin (None = disabled)")
  parser.add_argument('--collision_freeze_rotation',
                      type=float,
                      default=None,
                      help="freeze rotation under collision (None = disabled)")

  # [part merge] merge_options(): neighbor-motion part merging
  parser.add_argument('--rel_tol',
                      type=float,
                      default=0.02,
                      help="relative tolerance for merging neighboring parts")
  parser.add_argument('--merge_method',
                      type=str,
                      default='both',
                      help="merge criterion: v3|depth|both")
  parser.add_argument('--tau_merge',
                      type=float,
                      default=5e-4,
                      help="motion-similarity threshold for merging")

  # [refinement] refinement_demo(): final joint refinement
  parser.add_argument('--refine_depth_weight',
                      type=float,
                      default=4.0,
                      help="depth loss weight during refinement")

  # [visualization] vis_axes_pp_demo(): articulation-axis arrows
  parser.add_argument('--arrow_scale',
                      type=float,
                      default=1.0,
                      help="scale of the drawn articulation-axis arrows")
  return parser


def main():
  args = build_arg_parser().parse_args()

  # derive single-frame pretrain paths from the output path:
  #   {dirname(output)}/single/{basename(output)}_st  (and _ed)
  # single_folder = os.path.join(os.path.dirname(args.output), 'single')
  # out_name = os.path.basename(args.output)
  # st = os.path.join(single_folder, out_name + '_st')
  # ed = os.path.join(single_folder, out_name + '_ed')
  out_name = os.path.basename(args.output)
  st = os.path.join(args.output, "pretrain", out_name + '_st')
  ed = os.path.join(args.output, "pretrain", out_name + '_ed')

  run_steps(args.output,
            args.source,
            st,
            ed,
            num_movable=args.gt_movable_part,
            reverse=args.reverse,
            args=args)


if __name__ == '__main__':
  main()
