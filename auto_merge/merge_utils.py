import copy
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
import math
from scipy.spatial import cKDTree

from .cal_trans_utils import best_to_o3d_T, residual_nn_torch


def _prepare_rct_merge_inputs(
    Pcd_parts: Sequence[np.ndarray],
    Pi: np.ndarray,
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    Pj: np.ndarray | None = None,
):
  if not (len(Pcd_parts) == len(parts_idx) == len(motions)):
    raise ValueError(
        "Pcd_parts, parts_idx, and motions must have the same length")

  Pi_arr = np.asarray(Pi, dtype=np.float32)
  Pj_arr = None if Pj is None else np.asarray(Pj, dtype=np.float32)
  parts_idx_arr = [np.asarray(idx, dtype=np.int32).copy() for idx in parts_idx]
  parts_points = [
      np.asarray(part, dtype=np.float32).copy() for part in Pcd_parts
  ]
  motions_copy = [copy.deepcopy(motion) for motion in motions]
  return Pi_arr, Pj_arr, parts_idx_arr, parts_points, motions_copy


def _apply_rct_motion(points_np: np.ndarray, motion: dict) -> np.ndarray:
  R = np.asarray(motion.get("R"), dtype=np.float64)
  c = np.asarray(motion.get("c"), dtype=np.float64)
  t = np.asarray(motion.get("t"), dtype=np.float64)
  if R.shape != (3, 3) or c.shape != (3,) or t.shape != (3,):
    raise ValueError("motion must contain R(3,3), c(3,), t(3,)")
  return ((points_np - c) @ R) + c + t


def _build_movable_knn_adjacency(
    Pi: np.ndarray,
    parts_idx: Sequence[np.ndarray],
    neighbor_k: int,
) -> List[set[int]]:
  adjacency: List[set[int]] = [set() for _ in parts_idx]
  if len(parts_idx) <= 1 or neighbor_k <= 0:
    return adjacency

  point_to_part = np.full(Pi.shape[0], -1, dtype=np.int32)
  for pid, idx in enumerate(parts_idx):
    point_to_part[idx] = pid

  movable_global_idx = np.flatnonzero(point_to_part >= 0)
  if movable_global_idx.size <= 1:
    return adjacency

  movable_pts = Pi[movable_global_idx]
  k_nn = min(neighbor_k + 1, movable_pts.shape[0])
  if k_nn <= 1:
    return adjacency

  tree = cKDTree(movable_pts)
  nn_local = tree.query(movable_pts, k=k_nn)[1]
  nn_local = np.atleast_2d(nn_local)

  for local_i, neighbors in enumerate(nn_local):
    src_part = int(point_to_part[movable_global_idx[local_i]])
    for local_j in np.atleast_1d(neighbors):
      local_j = int(local_j)
      if local_j == local_i or local_j >= movable_global_idx.size:
        continue
      dst_part = int(point_to_part[movable_global_idx[local_j]])
      if dst_part >= 0 and dst_part != src_part:
        adjacency[src_part].add(dst_part)
  return adjacency


def _merge_parts_in_place(
    Pi: np.ndarray,
    parts_points: List[np.ndarray],
    parts_idx: List[np.ndarray],
    donor: int,
    accept: int,
    *,
    merged_idx: np.ndarray | None = None,
) -> np.ndarray:
  if merged_idx is None:
    merged_idx = np.unique(
        np.concatenate([parts_idx[accept], parts_idx[donor]], axis=0))
  merged_idx = np.asarray(merged_idx, dtype=np.int32)
  parts_idx[accept] = merged_idx
  parts_points[accept] = Pi[merged_idx]
  return merged_idx


def _pop_merged_donor(donor: int, *arrays) -> None:
  for arr in arrays:
    arr.pop(donor)


def _run_neighbor_merge_loop(
    parts_points: List[np.ndarray],
    parts_idx: List[np.ndarray],
    motions: List[dict],
    *,
    Pi: np.ndarray,
    neighbor_k: int,
    max_iters: int,
    loop_label: str,
    try_merge_once,
    log_adjacency: bool = False,
    stop_message: str | None = "  no merge candidate found, stopping.",
):
  merge_records: List[dict] = []
  has_merged = False

  for iter_idx in range(max_iters):
    print(f"[{loop_label} iter={iter_idx}] parts count: {len(parts_idx)}")
    if len(parts_idx) <= 1:
      break

    adjacency = _build_movable_knn_adjacency(Pi, parts_idx, neighbor_k)
    if log_adjacency:
      print(f" - <{len(adjacency)}> adjacency built")

    merge_record = try_merge_once(adjacency)
    merge_records.append(merge_record)
    if merge_record is None:
      if stop_message:
        print(stop_message)
      break

    has_merged = True

  return merge_records, has_merged


def _create_cd_merge_context(
    Pj: np.ndarray,
    parts_points: Sequence[np.ndarray],
    motions: Sequence[dict],
    *,
    rel_tol: float,
    abs_tol: float,
    device: str,
):
  score_device = torch.device(device)
  target_torch = torch.from_numpy(Pj).to(score_device)

  @torch.no_grad()
  def _residual(points_np: np.ndarray) -> float:
    pts = torch.from_numpy(points_np.astype(np.float32,
                                            copy=False)).to(score_device)
    return float(residual_nn_torch(pts, target_torch))

  residuals: List[float] = []
  for pts, motion in zip(parts_points, motions):
    motion_residual = motion.get("residual")
    if motion_residual is None:
      motion_residual = _residual(_apply_rct_motion(pts, motion))
      motion["residual"] = float(motion_residual)
    residuals.append(float(motion_residual))

  def _evaluate(
      donor: int,
      accept: int,
      Pi: np.ndarray,
      parts_points: Sequence[np.ndarray],
      parts_idx: Sequence[np.ndarray],
      motions: Sequence[dict],
  ) -> dict:
    donor_threshold = residuals[donor] * (1.0 + rel_tol) + abs_tol
    donor_score = _residual(
        _apply_rct_motion(parts_points[donor], motions[accept]))

    _base = {
        "valid": False,
        "score": donor_score,
        "threshold": donor_threshold,
        "ratio": donor_score / max(donor_threshold, 1e-12),
        "donor": donor,
        "accept": accept,
    }
    if donor_score > donor_threshold:
      return _base

    merged_idx = np.unique(
        np.concatenate([parts_idx[accept], parts_idx[donor]], axis=0))
    merged_points = Pi[merged_idx]
    accept_score = _residual(_apply_rct_motion(merged_points, motions[accept]))
    accept_threshold = residuals[accept] * (1.0 + rel_tol) + abs_tol
    accept_limit = max(accept_threshold, donor_score)

    if accept_score > accept_limit:
      return _base

    return {
        "valid": True,
        "score": donor_score,
        "threshold": donor_threshold,
        "ratio": donor_score / max(donor_threshold, 1e-12),
        "accept_score": accept_score,
        "accept_threshold": accept_limit,
        "accept_ratio": accept_score / max(accept_limit, 1e-12),
        "merged_idx": np.asarray(merged_idx, dtype=np.int32),
        "donor": donor,
        "accept": accept,
    }

  def _commit(candidate: dict, motions: Sequence[dict]) -> float:
    donor = int(candidate["donor"])
    accept = int(candidate["accept"])
    accept_score = float(candidate["accept_score"])
    residuals[accept] = accept_score
    motions[accept]["residual"] = accept_score
    residuals.pop(donor)
    return accept_score

  def _commit_depth_only(
      donor: int,
      accept: int,
      parts_points: Sequence[np.ndarray],
      motions: Sequence[dict],
  ) -> float:
    """
      only update the accept part's residual and motion, without merging the points or indices.
    """

    new_accept_residual = _residual(
        _apply_rct_motion(parts_points[accept], motions[accept]))
    residuals[accept] = new_accept_residual
    motions[accept]["residual"] = new_accept_residual
    residuals.pop(donor)
    return new_accept_residual

  return {
      "evaluate": _evaluate,
      "commit": _commit,
      "commit_depth_only": _commit_depth_only,
      "num_parts": lambda: len(residuals),
  }


def neighbor_motion_merge_v3(
    Pcd_parts: Sequence[np.ndarray],
    Pi: np.ndarray,
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    Pj: np.ndarray,
    *,
    neighbor_k: int = 16,
    rel_tol: float = 0.05,
    abs_tol: float = 1e-4,
    max_iters: int = 40,
    device: str = "cuda",
):
  """
  merge neighboring parts based on motion similarity using point cloud residuals.
  """

  Pi, Pj_arr, parts_idx, parts_points, motions = _prepare_rct_merge_inputs(
      Pcd_parts, Pi, parts_idx, motions, Pj)
  if Pj_arr is None:
    raise ValueError("Pj must not be None")

  cd_ctx = _create_cd_merge_context(
      Pj_arr,
      parts_points,
      motions,
      rel_tol=rel_tol,
      abs_tol=abs_tol,
      device=device,
  )

  def _try_merge_once(adjacency: Sequence[set[int]]) -> dict | None:
    for donor in range(len(parts_idx)):
      neighbors = adjacency[donor]
      if not neighbors:
        continue

      donor_best_candidate = None
      for accept in neighbors:
        if accept == donor:
          continue
        cd_eval = cd_ctx["evaluate"](donor, accept, Pi, parts_points, parts_idx,
                                     motions)
        if not cd_eval["valid"]:
          continue
        if donor_best_candidate is None or cd_eval[
            "score"] < donor_best_candidate["score"]:
          donor_best_candidate = cd_eval

      if donor_best_candidate is None:
        continue

      donor = int(donor_best_candidate["donor"])
      accept = int(donor_best_candidate["accept"])
      _merge_parts_in_place(
          Pi,
          parts_points,
          parts_idx,
          donor,
          accept,
          merged_idx=donor_best_candidate["merged_idx"],
      )
      new_residual = cd_ctx["commit"](donor_best_candidate, motions)
      accept_after = accept - 1 if donor < accept else accept
      print(
          f" - Merged part {donor} into {accept} | new accept score: {new_residual:.6f}"
      )
      _pop_merged_donor(donor, parts_points, parts_idx, motions)
      return {
          "donor": donor,
          "accept_before": accept,
          "accept_after": accept_after,
          "score": float(donor_best_candidate["score"]),
      }

    return None

  merge_records, has_merged = _run_neighbor_merge_loop(
      parts_points,
      parts_idx,
      motions,
      Pi=Pi,
      neighbor_k=neighbor_k,
      max_iters=max_iters,
      loop_label="run",
      try_merge_once=_try_merge_once,
      log_adjacency=True,
      stop_message=None,
  )
  return parts_points, parts_idx, motions, merge_records, has_merged


def filter_small_parts(
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    min_points: int = 1000,
) -> tuple[Sequence[np.ndarray], Sequence[Iterable[int]], Sequence[dict]]:
  """Remove parts with fewer points than the specified threshold.

    Inputs:
          parts_points: List of point coordinates split by part.
          parts_idx: Index sets of each part in the original point cloud.
          motions: Motion estimation results corresponding to ``parts_points``.
          min_points: Minimum number of points threshold. Parts with fewer points will be removed.
    Returns:
          tuple: Filtered (parts_points, parts_idx, motions) lists.
    """
  filtered_points = []
  filtered_idx = []
  filtered_motions = []

  for pts, idx, motion in zip(parts_points, parts_idx, motions):
    if pts.shape[0] >= min_points:
      filtered_points.append(pts)
      filtered_idx.append(idx)
      filtered_motions.append(motion)
    else:
      print(f" - Filtered small part with {pts.shape[0]} points")

  return filtered_points, filtered_idx, filtered_motions


def filter_static_parts_v3(
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    *,
    translation_thresh: float = 1e-2,
    rotation_thresh: float = np.deg2rad(5.0),
) -> tuple[Sequence[np.ndarray], Sequence[Iterable[int]], Sequence[dict]]:
  """Filter out static parts based on motion parameters.

    Inputs:
          parts_points: List of point coordinates split by part.
          parts_idx: Index sets of each part in the original point cloud.
          motions: Motion estimation results corresponding to ``parts_points``.
          translation_thresh: Translation norm threshold below which a part is considered static.
          rotation_thresh: Rotation angle threshold below which a part is considered static.
    Returns:
          tuple: Filtered (parts_points, parts_idx, motions) lists.
  """

  def _is_static(motion: dict) -> bool:
    R = np.asarray(motion.get("R"), dtype=np.float64)
    c = np.asarray(motion.get("c"), dtype=np.float64)
    t = np.asarray(motion.get("t"), dtype=np.float64)
    if R.shape != (3, 3) or c.shape != (3,) or t.shape != (3,):
      # If motion parameters are invalid, consider the part as static.
      return False

    trans_norm = float(np.linalg.norm(t))
    trace = float(np.clip(np.trace(R), -1.0, 3.0))
    rot_angle = math.acos(max(min((trace - 1.0) * 0.5, 1.0), -1.0))
    return trans_norm < translation_thresh and rot_angle < rotation_thresh

  filtered_points = []
  filtered_idx = []
  filtered_motions = []

  for pts, idx, motion in zip(parts_points, parts_idx, motions):
    if _is_static(motion):
      print(f" - Filtered static part (v3) with motion: {motion}")
      continue
    filtered_points.append(pts)
    filtered_idx.append(idx)
    filtered_motions.append(motion)

  return filtered_points, filtered_idx, filtered_motions


def filter_keep_largest_cluster(
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    *,
    radius: float = 0.06,
    min_cluster_size: int = 100,
) -> tuple[Sequence[np.ndarray], Sequence[Iterable[int]], Sequence[dict]]:
  """Keep the largest connected cluster in each part to suppress outliers.
  
    Inputs:
          parts_points: List of point coordinates split by part.
          parts_idx: Index sets of each part in the original point cloud.
          motions: Motion estimation results corresponding to ``parts_points``.
          radius: Radius for neighbor search to define connectivity.
          min_cluster_size: Minimum number of points for a cluster to be kept.
  """

  if radius <= 0:
    raise ValueError("radius must be positive")

  filtered_points: List[np.ndarray] = []
  filtered_idx: List[np.ndarray] = []
  filtered_motions: List[dict] = []

  for pts, idx, motion in zip(parts_points, parts_idx, motions):
    pts_arr = np.asarray(pts, dtype=np.float32)
    idx_arr = np.asarray(idx, dtype=np.int32)

    if pts_arr.shape[0] < min_cluster_size:
      filtered_points.append(pts_arr)
      filtered_idx.append(idx_arr)
      filtered_motions.append(motion)
      continue

    tree = cKDTree(pts_arr)
    neighbor_lists = tree.query_ball_tree(tree, r=radius)

    visited = np.zeros(len(neighbor_lists), dtype=bool)
    components: List[List[int]] = []

    for start in range(len(neighbor_lists)):
      if visited[start]:
        continue
      queue = deque([start])
      visited[start] = True
      component: List[int] = []

      while queue:
        current = queue.popleft()
        component.append(current)
        for nb in neighbor_lists[current]:
          if nb >= len(neighbor_lists) or visited[nb]:
            continue
          visited[nb] = True
          queue.append(nb)

      components.append(component)

    if not components:
      filtered_points.append(pts_arr)
      filtered_idx.append(idx_arr)
      filtered_motions.append(motion)
      continue

    largest_component = max(components, key=len)

    if len(largest_component) < min_cluster_size or len(
        largest_component) == pts_arr.shape[0]:
      filtered_points.append(pts_arr)
      filtered_idx.append(idx_arr)
      filtered_motions.append(motion)
      continue

    kept_points = pts_arr[largest_component]
    kept_idx = idx_arr[largest_component]
    print(
        f" - Filtered {pts_arr.shape[0] - kept_points.shape[0]} outlier points; kept {kept_points.shape[0]}"
    )

    filtered_points.append(kept_points)
    filtered_idx.append(kept_idx)
    filtered_motions.append(motion)

  return filtered_points, filtered_idx, filtered_motions


def _make_scoring_am_copy(am):
  """Create a simple ArtModel, only used for depth scoring (deform + render)
  """
  from scene.gaussian_model import GaussianModel
  from torch import nn as _nn

  sc = copy.copy(am)  # Shallow copy

  # 1. copy the Gaussians (deep copy of parameters, but features/scaling are shared)
  gs = am.gaussians
  gs_copy = GaussianModel(gs.max_sh_degree)
  gs_copy.active_sh_degree = gs.active_sh_degree
  gs_copy._xyz = gs._xyz.clone()
  gs_copy._rotation = gs._rotation.clone()
  gs_copy._opacity = gs._opacity.clone()
  # features / scaling are read-only in deform+render, so they can be shared
  gs_copy._features_dc = gs._features_dc
  gs_copy._features_rest = gs._features_rest
  gs_copy._scaling = gs._scaling
  sc.gaussians = gs_copy

  # 2. copy motion parameters (these will be repeatedly modified during scoring)
  sc._column_vec1 = [
      _nn.Parameter(p.data.clone(), requires_grad=False)
      for p in am._column_vec1
  ]
  sc._column_vec2 = [
      _nn.Parameter(p.data.clone(), requires_grad=False)
      for p in am._column_vec2
  ]
  sc._t = [_nn.Parameter(p.data.clone(), requires_grad=False) for p in am._t]
  sc._c = [_nn.Parameter(p.data.clone(), requires_grad=False) for p in am._c]

  # 3. copy pcd deformation cache (deform will write into these)
  if am.pcds_deformed_all is not None:
    sc.pcds_deformed_all = am.pcds_deformed_all.clone()
  sc.pcds_deformed = [
      t.clone() if isinstance(t, torch.Tensor) else t for t in am.pcds_deformed
  ]

  # 4. reset ppp cache
  sc.ppp = None

  return sc


# ===========================================================================
# both-merge (depth + CD)
# ===========================================================================

# Corss verification: when only one metric triggers a merge, the other metric is allowed to have a "counter strength" upper limit (in units of its own threshold). Depth has stronger discriminative power for parallel structures (such as side-by-side drawers), so when depth triggers, give CD a stricter tolerance (5x), and when CD triggers, give depth a wider tolerance (10x).
_CV_CD_FACTOR = 5.0  # if depth triggers, CD's counter tolerance
_CV_DEPTH_FACTOR = 10.0  # if CD triggers, depth's counter tolerance

# --- dataclasses -------------------------------------------------------------


@dataclass
class ChamferEvaluation:
  """Chamfer (CD) residual-based evaluation result for a single (donor → accept) merge pair.
  """
  is_mergeable: bool
  donor_score: float  # donor point cloud residual under accept motion
  donor_threshold: float  # donor residual threshold
  donor_ratio: float  # donor_score / donor_threshold
  accept_score: float = float(
      "nan")  # merged accept point cloud residual under accept motion
  accept_threshold: float = float("nan")
  accept_ratio: float = float("nan")
  merged_point_indices: np.ndarray | None = None  # accept ∪ donor global indices


@dataclass
class ChamferMergeState:
  """Chamfer (CD) residual-based merge state, used for evaluating and committing merges.
  """
  target_points: torch.Tensor  # target point cloud Pj, already moved to the computation device
  part_residuals: List[float]  # residuals of each part under its own motion
  relative_tolerance: float
  absolute_tolerance: float
  device: torch.device


@dataclass
class DepthEvaluation:
  """Depth-render evaluation result for a single (donor → accept) merge pair."""
  is_mergeable: bool
  score: float  # average depth score variation across top-k views
  threshold: float
  ratio: float  # score / top_baseline_mean
  top_baseline_mean: float  # baseline mean across top-k views
  max_view_delta: float  # maximum single-view score variation across all views


@dataclass
class DepthMergeState:
  """Depth-render merge state, used for evaluating and committing merges.
  """
  scoring_model: object  # lightweight copy of _make_scoring_am_copy(am) (used only for scoring)
  render: object  # gaussian_renderer.render
  pipeline: object  # am.pipe
  background: object  # bws.background_black
  sampled_cameras: list
  view_valid_masks: List[
      torch.Tensor]  # GT depth valid pixel masks for each view
  view_valid_pixel_counts: List[
      float]  # valid pixel counts for each view (used for pixel-wise averaging)
  am_part_id_groups: List[List[int]]  # Gaussian group IDs under each part
  effective_top_k: int
  tau_merge: float
  depth_error_clip: float  # maximum single-pixel depth error, to prevent domination by outliers


@dataclass
class MergeCandidate:
  """A merge candidate that has passed evaluation (and cross-validation) and is eligible for selection in the current round."""
  donor: int
  accept: int
  criteria: List[str]  # ["depth"] / ["cd"] / ["depth", "cd"]
  depth_eval: DepthEvaluation | None  # non-None only if depth criterion is triggered
  cd_eval: ChamferEvaluation | None  # non-None only if CD criterion is triggered
  sort_key: tuple  # smaller is better, see merge_candidate_sort_key


# --- chamfer（CD ------------------------------------------------


def _chamfer_residual_under_motion(
    points: np.ndarray,
    motion: dict,
    target_points: torch.Tensor,
    device: torch.device,
) -> float:
  """points residual under motion, compared to target_points."""
  transformed = _apply_rct_motion(points, motion)
  pts = torch.from_numpy(transformed.astype(np.float32, copy=False)).to(device)
  return float(residual_nn_torch(pts, target_points))


def create_chamfer_merge_state(
    target_points_np: np.ndarray,
    parts_points: Sequence[np.ndarray],
    motions: Sequence[dict],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    device: str,
) -> ChamferMergeState:
  """Initialize chamfer merge state and compute/cache residuals for parts missing them."""
  compute_device = torch.device(device)
  target_points = torch.from_numpy(target_points_np).to(compute_device)

  part_residuals: List[float] = []
  for points, motion in zip(parts_points, motions):
    residual = motion.get("residual")
    if residual is None:
      residual = _chamfer_residual_under_motion(points, motion, target_points,
                                                compute_device)
      motion["residual"] = float(residual)
    part_residuals.append(float(residual))

  return ChamferMergeState(
      target_points=target_points,
      part_residuals=part_residuals,
      relative_tolerance=relative_tolerance,
      absolute_tolerance=absolute_tolerance,
      device=compute_device,
  )


def evaluate_chamfer_merge(
    state: ChamferMergeState,
    donor: int,
    accept: int,
    Pi: np.ndarray,
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[np.ndarray],
    motions: Sequence[dict],
) -> ChamferEvaluation:
  """Evaluate whether a chamfer merge from donor to accept is valid.

  First, check if the residual of the donor point set under the accept motion
  is within the donor threshold. If it passes, then check if the residual of
  the merged accept point set under the accept motion is within the accept
  threshold (the threshold is at least relaxed to accommodate the donor_score,
  preventing the merge from being blocked by the donor's own residual).
  """
  rel = state.relative_tolerance
  abs_tol = state.absolute_tolerance

  donor_threshold = state.part_residuals[donor] * (1.0 + rel) + abs_tol
  donor_score = _chamfer_residual_under_motion(parts_points[donor],
                                               motions[accept],
                                               state.target_points,
                                               state.device)
  donor_ratio = donor_score / max(donor_threshold, 1e-12)

  if donor_score > donor_threshold:
    return ChamferEvaluation(
        is_mergeable=False,
        donor_score=donor_score,
        donor_threshold=donor_threshold,
        donor_ratio=donor_ratio,
    )

  merged_point_indices = np.unique(
      np.concatenate([parts_idx[accept], parts_idx[donor]], axis=0))
  merged_points = Pi[merged_point_indices]
  accept_score = _chamfer_residual_under_motion(merged_points, motions[accept],
                                                state.target_points,
                                                state.device)
  accept_threshold = max(state.part_residuals[accept] * (1.0 + rel) + abs_tol,
                         donor_score)
  accept_ratio = accept_score / max(accept_threshold, 1e-12)

  return ChamferEvaluation(
      is_mergeable=accept_score <= accept_threshold,
      donor_score=donor_score,
      donor_threshold=donor_threshold,
      donor_ratio=donor_ratio,
      accept_score=accept_score,
      accept_threshold=accept_threshold,
      accept_ratio=accept_ratio,
      merged_point_indices=np.asarray(merged_point_indices, dtype=np.int32),
  )


def chamfer_residual_for_merged_accept(
    state: ChamferMergeState,
    accept: int,
    parts_points: Sequence[np.ndarray],
    motions: Sequence[dict],
) -> float:
  """Chamfer residual of the merged accept point set under the accept motion.

  This is used only when the current merge is triggered solely by depth (CD is not triggered,
  so ChamferEvaluation does not have a ready-made accept_score). _merge_parts_in_place must
  have been called before, so that parts_points[accept] is the merged point set.
  """
  return _chamfer_residual_under_motion(parts_points[accept], motions[accept],
                                        state.target_points, state.device)


def commit_chamfer_merge(
    state: ChamferMergeState,
    donor: int,
    accept: int,
    new_accept_residual: float,
    motions: Sequence[dict],
) -> None:
  """Commit the merge by updating the chamfer state: write the new residual for accept and pop donor."""
  state.part_residuals[accept] = new_accept_residual
  motions[accept]["residual"] = new_accept_residual
  state.part_residuals.pop(donor)


# --- depth-render metrics ----------------------------------------------------


def create_depth_merge_state(
    parts_idx: Sequence[np.ndarray],
    motions: Sequence[dict],
    am,
    bws,
    *,
    tau_merge: float,
    top_k_views: int,
) -> DepthMergeState | None:
  """Initialize depth-render merge state; return None if no cameras with GT depth are available.

  Scoring is done on a lightweight copy of the original am (without modifying the original am).
  Each logical part holds a "group" of am_pids (Gaussian groups), initially one-to-one;
  after merging, the entire group under the donor is merged into the accept, ensuring that
  the transitive closure of merges remains consistent in depth geometry.
  """
  from gaussian_renderer import render as gs_render

  train_cameras = bws.scene_black.getTrainCameras()
  depth_cameras = [cam for cam in train_cameras if cam.image_depth is not None]
  if not depth_cameras:
    return None

  scoring_model = _make_scoring_am_copy(am)
  print("[depth-merge] Created lightweight scoring copy of ArtModel.")

  ppp_np = am.get_ppp().detach().cpu().numpy()
  part_assignment = np.argmax(ppp_np, axis=1)
  am_part_id_groups: List[List[int]] = [
      [int(part_assignment[idx[0]])] for idx in parts_idx
  ]

  # Sample one depth camera every 4 frames to control scoring overhead.
  sampled_cameras = [
      depth_cameras[i] for i in range(len(depth_cameras)) if i % 4 == 0
  ]
  n_sampled = len(sampled_cameras)
  effective_top_k = min(top_k_views, n_sampled) if n_sampled > 0 else 1
  print(f"[depth-merge] sampled views: {n_sampled}, "
        f"top_k_views: {effective_top_k}")

  # Keep all valid GT depth pixels (only exclude gt <= 0.01). Also record the number of valid pixels per view for pixel-wise averaging (numerical stability, comparable across views/iterations).
  view_valid_masks: List[torch.Tensor] = []
  view_valid_pixel_counts: List[float] = []
  for cam in sampled_cameras:
    gt_depth = cam.image_depth.cuda().float()
    valid = gt_depth > 0.01
    view_valid_masks.append(valid)
    view_valid_pixel_counts.append(float(valid.sum().item()))

  return DepthMergeState(
      scoring_model=scoring_model,
      render=gs_render,
      pipeline=am.pipe,
      background=bws.background_black,
      sampled_cameras=sampled_cameras,
      view_valid_masks=view_valid_masks,
      view_valid_pixel_counts=view_valid_pixel_counts,
      am_part_id_groups=am_part_id_groups,
      effective_top_k=effective_top_k,
      tau_merge=tau_merge,
      # A single error pixel can dominate the entire score after being amplified by exp,
      # so we clip the error to an upper limit.
      depth_error_clip=1.0,
  )


def _set_part_group_motion(
    state: DepthMergeState,
    am_pids: Sequence[int],
    motion: dict,
) -> None:
  """Set the motion parameters for a group of Gaussians (am_pids) to the given motion."""
  R = np.asarray(motion["R"], dtype=np.float32)
  c = np.asarray(motion["c"], dtype=np.float32)
  t = np.asarray(motion["t"], dtype=np.float32)
  model = state.scoring_model
  for am_pid in am_pids:
    device0 = model._column_vec1[am_pid].device
    model._column_vec1[am_pid].data.copy_(
        torch.as_tensor(R[:, 0], dtype=torch.float32, device=device0))
    model._column_vec2[am_pid].data.copy_(
        torch.as_tensor(R[:, 1], dtype=torch.float32, device=device0))
    model._c[am_pid].data.copy_(
        torch.as_tensor(c, dtype=torch.float32, device=device0))
    model._t[am_pid].data.copy_(
        torch.as_tensor(t, dtype=torch.float32, device=device0))


def _apply_all_part_motions(
    state: DepthMergeState,
    motions: Sequence[dict],
) -> None:
  """把每个 part 当前的运动写入其名下所有 Gaussian 分组。"""
  for part_index, am_pids in enumerate(state.am_part_id_groups):
    _set_part_group_motion(state, am_pids, motions[part_index])


@torch.no_grad()
def _render_depth_score_per_view(state: DepthMergeState) -> List[float]:
  """Render each sampled view according to the current motion of the scoring_model and return the depth score per view.

  Each view's score = mean of (exp(clamp(|render - gt|)) - 1) over valid pixels.
  """
  model = state.scoring_model
  model.ppp = None
  model.deform(-1)

  per_view: List[float] = []
  for cam_index, cam in enumerate(state.sampled_cameras):
    package = state.render(cam,
                           model.gaussians,
                           state.pipeline,
                           state.background,
                           opacity_filter=0.01)
    depth = package["depth"]
    gt_depth = cam.image_depth.cuda().float()
    valid = state.view_valid_masks[cam_index]
    error = torch.abs(depth - gt_depth).clamp(max=state.depth_error_clip)
    per_pixel = (torch.exp(error) - 1.0) * valid
    denominator = max(state.view_valid_pixel_counts[cam_index], 1.0)
    per_view.append(per_pixel.sum().item() / denominator)
  return per_view


def compute_depth_baseline_per_view(
    state: DepthMergeState,
    motions: Sequence[dict],
) -> List[float]:
  """Compute the baseline depth score per view using the current motion of each part."""
  _apply_all_part_motions(state, motions)
  baseline_per_view = _render_depth_score_per_view(state)
  baseline_total = sum(baseline_per_view)
  baseline_mean = baseline_total / max(len(baseline_per_view), 1)
  print(f"  baseline depth score: total={baseline_total:.4f}, "
        f"per-view mean={baseline_mean:.4f}")
  return baseline_per_view


def evaluate_depth_merge(
    state: DepthMergeState,
    donor: int,
    accept: int,
    motions: Sequence[dict],
    baseline_per_view: Sequence[float],
) -> DepthEvaluation:
  """Evaluate whether the depth merge from donor to accept is valid.

  Temporarily replace the donor's motion with the accept's motion, re-render, and take the average
  of the top-k views with the largest score increase as the variation; if the variation is below
  the threshold, the merge is valid (only consider the views with the largest increase to avoid
  diluting the discriminative signal of small parts across all views).
  """
  donor_am_pids = state.am_part_id_groups[donor]
  donor_motion = motions[donor]

  _set_part_group_motion(state, donor_am_pids, motions[accept])
  test_per_view = _render_depth_score_per_view(state)
  _set_part_group_motion(state, donor_am_pids, donor_motion)  # Restore

  per_view_deltas = [
      test_score - base_score
      for test_score, base_score in zip(test_per_view, baseline_per_view)
  ]
  views_by_delta_desc = sorted(
      range(len(per_view_deltas)),
      key=lambda idx: per_view_deltas[idx],
      reverse=True,
  )
  top_views = views_by_delta_desc[:state.effective_top_k]
  variation = sum(
      per_view_deltas[idx] for idx in top_views) / state.effective_top_k
  top_baseline_mean = sum(
      baseline_per_view[idx] for idx in top_views) / state.effective_top_k

  absolute_threshold = state.tau_merge * state.depth_error_clip
  threshold = max(state.tau_merge * top_baseline_mean, absolute_threshold)

  return DepthEvaluation(
      is_mergeable=variation < threshold,
      score=variation,
      threshold=threshold,
      ratio=variation / max(top_baseline_mean, 1e-12),
      top_baseline_mean=top_baseline_mean,
      max_view_delta=per_view_deltas[views_by_delta_desc[0]],
  )


def commit_depth_merge(
    state: DepthMergeState,
    donor: int,
    accept: int,
    motions: Sequence[dict],
) -> None:
  """Merge all Gaussian groups under the donor into the accept, align to the accept's current motion, and then pop the donor."""
  donor_am_pids = state.am_part_id_groups[donor]
  state.am_part_id_groups[accept].extend(donor_am_pids)
  _set_part_group_motion(state, donor_am_pids, motions[accept])
  state.am_part_id_groups.pop(donor)


def cleanup_depth_merge_state(state: DepthMergeState) -> None:
  """Release the GPU memory occupied by depth scores."""
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --- Candidate evaluation, cross-validation, and selection --------------------------------------------


def merge_candidate_sort_key(
    depth_eval: DepthEvaluation | None,
    cd_eval: ChamferEvaluation | None,
) -> tuple:
  """Sort key for merge candidates, smaller is better.

  Compare (worst metric ratio, best metric ratio, -number of satisfied metrics) in order:
  first look at the worst metric, then the best, and in case of a tie, prefer candidates
  that satisfy both depth and CD. The passed evals are all triggered (valid) metrics, so
  cd_eval's accept_ratio is always valid.
  """
  ratios: List[float] = []
  criteria_count = 0
  if depth_eval is not None:
    ratios.append(float(depth_eval.ratio))
    criteria_count += 1
  if cd_eval is not None:
    ratios.append(float(cd_eval.donor_ratio))
    ratios.append(float(cd_eval.accept_ratio))
    criteria_count += 1
  return (min(ratios), max(ratios), -criteria_count)


def cross_validation_rejects_merge(
    donor: int,
    accept: int,
    depth_eval: DepthEvaluation,
    cd_eval: ChamferEvaluation,
) -> bool:
  """Check if the other metric strongly disagrees when only one metric triggers a merge.
  Returns True if the merge should be rejected.

  If both metrics trigger, no cross-validation is performed (returns False). Depth has
  stronger discriminative power for parallel structures (e.g., side-by-side drawers),
  so when depth triggers, CD is given a stricter tolerance (_CV_CD_FACTOR), and when CD
  triggers, depth is given a looser tolerance (_CV_DEPTH_FACTOR).
  """
  if cd_eval.is_mergeable and not depth_eval.is_mergeable:
    # CD triggers: depth variation must not exceed _CV_DEPTH_FACTOR × depth threshold.
    depth_veto_limit = _CV_DEPTH_FACTOR * depth_eval.threshold
    if depth_eval.score > depth_veto_limit:
      print(f"  Skip merge donor {donor} → accept {accept}: "
            f"CD pass but depth strongly disagrees "
            f"(score={depth_eval.score:.4f} > {depth_veto_limit:.4f})")
      return True

  if depth_eval.is_mergeable and not cd_eval.is_mergeable:
    # Depth triggers: CD's donor score must not exceed _CV_CD_FACTOR × CD donor threshold.
    cd_veto_limit = _CV_CD_FACTOR * cd_eval.donor_threshold
    if cd_eval.donor_score > cd_veto_limit:
      print(f"  Skip merge donor {donor} → accept {accept}: "
            f"depth pass but CD strongly disagrees "
            f"(score={cd_eval.donor_score:.4f} > {cd_veto_limit:.4f})")
      return True

  return False


def _evaluate_merge_pair(
    donor: int,
    accept: int,
    cd_state: ChamferMergeState,
    depth_state: DepthMergeState,
    baseline_per_view: Sequence[float],
    Pi: np.ndarray,
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[np.ndarray],
    motions: Sequence[dict],
) -> MergeCandidate | None:
  """Evaluate a single (donor → accept) merge pair.

  Returns None in two cases: both metrics do not trigger; or a single metric triggers but is vetoed by cross-validation. Otherwise, returns a MergeCandidate that can participate in this round of selection (criteria records which metrics are triggered).
  """
  depth_eval = evaluate_depth_merge(depth_state, donor, accept, motions,
                                    baseline_per_view)
  cd_eval = evaluate_chamfer_merge(cd_state, donor, accept, Pi, parts_points,
                                   parts_idx, motions)

  if not depth_eval.is_mergeable and not cd_eval.is_mergeable:
    return None
  if cross_validation_rejects_merge(donor, accept, depth_eval, cd_eval):
    return None

  depth_candidate = depth_eval if depth_eval.is_mergeable else None
  cd_candidate = cd_eval if cd_eval.is_mergeable else None
  criteria = (["depth"] if depth_candidate is not None else
              []) + (["cd"] if cd_candidate is not None else [])

  log_msg = (
      f"  Test merge donor {donor} → accept {accept} | "
      f"criteria: {'+'.join(criteria)} | "
      f"depth score: {depth_eval.score:.6f} / {depth_eval.threshold:.6f}")
  if cd_candidate is not None:
    log_msg += f" | cd score: {cd_candidate.donor_score:.6f}"
  print(log_msg)

  return MergeCandidate(
      donor=donor,
      accept=accept,
      criteria=criteria,
      depth_eval=depth_candidate,
      cd_eval=cd_candidate,
      sort_key=merge_candidate_sort_key(depth_candidate, cd_candidate),
  )


def _collect_merge_candidates(
    adjacency: Sequence[set[int]],
    cd_state: ChamferMergeState,
    depth_state: DepthMergeState,
    baseline_per_view: Sequence[float],
    Pi: np.ndarray,
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[np.ndarray],
    motions: Sequence[dict],
) -> List[MergeCandidate]:
  """Iterate over all adjacent pairs and return the list of merge candidates for this round."""
  candidates: List[MergeCandidate] = []
  for donor in range(len(parts_idx)):
    for accept in adjacency[donor]:
      if accept == donor:
        continue
      candidate = _evaluate_merge_pair(donor, accept, cd_state, depth_state,
                                       baseline_per_view, Pi, parts_points,
                                       parts_idx, motions)
      if candidate is not None:
        candidates.append(candidate)
  return candidates


def _assert_part_lists_consistent(
    parts_points: Sequence[np.ndarray],
    parts_idx: Sequence[np.ndarray],
    motions: Sequence[dict],
    cd_state: ChamferMergeState,
    depth_state: DepthMergeState,
) -> None:
  """Ensure the outer four lists and the internal lists of cd/depth states are of equal length and order.
  Any omission or synchronized pop will cause overall misalignment."""
  n_parts = len(parts_idx)
  assert (len(parts_points) == n_parts == len(motions) == len(
      cd_state.part_residuals) == len(depth_state.am_part_id_groups)), (
          f"part list length mismatch after merge: "
          f"parts_points={len(parts_points)}, parts_idx={n_parts}, "
          f"motions={len(motions)}, "
          f"cd_residuals={len(cd_state.part_residuals)}, "
          f"am_part_id_groups={len(depth_state.am_part_id_groups)}")


def _commit_merge_candidate(
    candidate: MergeCandidate,
    cd_state: ChamferMergeState,
    depth_state: DepthMergeState,
    Pi: np.ndarray,
    parts_points: List[np.ndarray],
    parts_idx: List[np.ndarray],
    motions: List[dict],
) -> None:
  """Execute a merge candidate: merge point sets, synchronize depth/chamfer states, pop donor, and check list consistency."""
  donor = candidate.donor
  accept = candidate.accept

  depth_msg = (f" | depth score: {candidate.depth_eval.score:.6f}"
               if candidate.depth_eval is not None else "")
  cd_msg = (f" | cd score: {candidate.cd_eval.donor_score:.6f}"
            if candidate.cd_eval is not None else "")
  donor_n = len(parts_idx[donor])
  accept_n = len(parts_idx[accept])
  print(f"  Merging part {donor} into {accept} | "
        f"criteria: {'+'.join(candidate.criteria)}{depth_msg}{cd_msg}")
  print(f"    points before merge | donor {donor}: {donor_n} | "
        f"accept {accept}: {accept_n} | sum: {donor_n + accept_n}")

  merged_point_indices = (None if candidate.cd_eval is None else
                          candidate.cd_eval.merged_point_indices)
  _merge_parts_in_place(Pi,
                        parts_points,
                        parts_idx,
                        donor,
                        accept,
                        merged_idx=merged_point_indices)
  merged_n = len(parts_idx[accept])
  print(f"    points after merge  | accept {accept}: {merged_n} | "
        f"deduped: {donor_n + accept_n - merged_n}")
  commit_depth_merge(depth_state, donor, accept, motions)

  # accept the new chamfer residual: if CD triggered, it was already computed during evaluation;
  # otherwise (depth-only trigger), recompute for the merged accept point set.
  if candidate.cd_eval is not None:
    new_accept_residual = candidate.cd_eval.accept_score
  else:
    new_accept_residual = chamfer_residual_for_merged_accept(
        cd_state, accept, parts_points, motions)
  commit_chamfer_merge(cd_state, donor, accept, new_accept_residual, motions)

  _pop_merged_donor(donor, parts_points, parts_idx, motions)
  _assert_part_lists_consistent(parts_points, parts_idx, motions, cd_state,
                                depth_state)


@torch.no_grad()
def neighbor_motion_merge_w_both_depth_cd(
    Pcd_parts: Sequence[np.ndarray],
    Pi: np.ndarray,
    parts_idx: Sequence[Iterable[int]],
    motions: Sequence[dict],
    Pj: np.ndarray,
    am,
    bws,
    *,
    neighbor_k: int = 8,
    rel_tol: float = 0.05,
    abs_tol: float = 1e-4,
    tau_merge: float = 1e-3,
    max_iters: int = 40,
    top_k_views: int = 8,
    device: str = "cuda",
):
  """Neighbor motion merge using both depth and CD metrics.

  Each round, a kNN adjacency graph is constructed over all movable parts, and adjacent pairs are evaluated
  for depth-render and chamfer merge metrics. A pair is considered a candidate if either metric triggers,
  but single-metric triggers require cross-validation by the other metric. The best candidate in each round
  is selected based on sort_key and merged, until no candidates remain or the number of parts drops to 1.

  Returns:
      tuple: (parts_points, parts_idx, motions, merge_records, has_merged),
             where merge_records is always an empty list.
  """
  if am is None or bws is None:
    print(
        "[both-merge] WARNING: missing depth context, falling back to CD merge (v3)."
    )
    return neighbor_motion_merge_v3(
        Pcd_parts,
        Pi,
        parts_idx,
        motions,
        Pj,
        neighbor_k=neighbor_k,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_iters=max_iters,
        device=device,
    )

  Pi, Pj_arr, parts_idx, parts_points, motions = _prepare_rct_merge_inputs(
      Pcd_parts, Pi, parts_idx, motions, Pj)
  if Pj_arr is None:
    raise ValueError("Pj cannot be None")

  cd_state = create_chamfer_merge_state(
      Pj_arr,
      parts_points,
      motions,
      relative_tolerance=rel_tol,
      absolute_tolerance=abs_tol,
      device=device,
  )
  depth_state = create_depth_merge_state(
      parts_idx,
      motions,
      am,
      bws,
      tau_merge=tau_merge,
      top_k_views=top_k_views,
  )
  if depth_state is None:
    print(
        "[both-merge] WARNING: no cameras with GT depth, falling back to CD merge (v3)."
    )
    return neighbor_motion_merge_v3(
        Pcd_parts,
        Pi,
        parts_idx,
        motions,
        Pj,
        neighbor_k=neighbor_k,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_iters=max_iters,
        device=device,
    )

  has_merged = False
  try:
    for iteration in range(max_iters):
      print(f"[both-merge iter={iteration}] parts count: {len(parts_idx)}")
      if len(parts_idx) <= 1:
        break

      adjacency = _build_movable_knn_adjacency(Pi, parts_idx, neighbor_k)
      baseline_per_view = compute_depth_baseline_per_view(depth_state, motions)
      candidates = _collect_merge_candidates(adjacency, cd_state, depth_state,
                                             baseline_per_view, Pi,
                                             parts_points, parts_idx, motions)
      if not candidates:
        print("  no merge candidate found, stopping.")
        break

      best_candidate = min(candidates, key=lambda c: c.sort_key)
      _commit_merge_candidate(best_candidate, cd_state, depth_state, Pi,
                              parts_points, parts_idx, motions)
      has_merged = True
  finally:
    cleanup_depth_merge_state(depth_state)

  merge_records: List[dict] = []
  return parts_points, parts_idx, motions, merge_records, has_merged
