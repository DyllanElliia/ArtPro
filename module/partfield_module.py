"""PartField 3D part-feature extraction + clustering.

Wraps the PartField PVCNN+triplane model for direct inference without the
Lightning trainer.  Operates on raw point-cloud input, returning per-point
(per-point features) results.

Typical usage
-------------
::

    from module.partfield_module import PartFieldModule
    from sklearn.cluster import KMeans

    module = PartFieldModule(
        checkpoint_path="./checkpoints/model_objaverse.ckpt",
        device="cuda:0",
        n_clusters=8,
    )

    # --- point-cloud input (k-means default) ---
    result = module.seg_3d({"points": pts_np})

    # --- adaptive part segmentation (IoU-merge, auto part count) ---
    result = module.seg_3d({"points": pts_np}, seg_fn=module._iou_seg_fn)

    # --- custom seg_fn (e.g. agglomerative clustering) ---
    from sklearn.cluster import AgglomerativeClustering
    def my_seg(features, data3d):
        feat_norm = features / (np.linalg.norm(features, -1, keepdims=True) + 1e-8)
        return AgglomerativeClustering(n_clusters=8).fit_predict(feat_norm)
    result = module.seg_3d({"points": pts_np}, seg_fn=my_seg)

    labels   = result["labels"]    # (N,) int32
    features = result["features"]  # (N, 448) float32
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from module.seg3d_base import Seg3D_Base

# ── PartField repo root (added to sys.path on first import) ───────────────────
_PARTFIELD_ROOT = Path(
    __file__).resolve().parent.parent / "third_party" / "PartField"


def _ensure_partfield_path() -> None:
  if str(_PARTFIELD_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARTFIELD_ROOT))


# ── Default config matching configs/final/demo.yaml ───────────────────────────


def _build_default_cfg():
  _ensure_partfield_path()
  from partfield.config.defaults import _C as base_cfg  # noqa: PLC0415
  cfg = base_cfg.clone()
  cfg.defrost()
  cfg.triplane_channels_low = 128
  cfg.triplane_channels_high = 512
  cfg.triplane_resolution = 128
  cfg.n_point_per_face = 1000
  cfg.n_sample_each = 10_000
  cfg.is_pc = False
  cfg.use_pvcnnonly = True
  cfg.use_2d_feat = False
  cfg.pvcnn.z_triplane_channels = 256
  cfg.pvcnn.z_triplane_resolution = 128
  cfg.dataset.val_batch_size = 1
  cfg.dataset.val_num_workers = 0
  cfg.freeze()
  return cfg


# ── helpers ───────────────────────────────────────────────────────────────────


def _bbox_normalize_params(pts: np.ndarray):
  """Return ``(center, scale)`` that maps pts bounding box into ``[-0.9, 0.9]``."""
  bbmin = pts.min(0)
  bbmax = pts.max(0)
  center = (bbmin + bbmax) * 0.5
  scale = 2.0 * 0.9 / (bbmax - bbmin).max()
  return center, scale


# ── module ────────────────────────────────────────────────────────────────────


class PartFieldModule(Seg3D_Base):
  """PartField part-feature extractor.

  Parameters
  ----------
  checkpoint_path : str
      Path to the ``.ckpt`` Lightning checkpoint.
  device : str
      Compute device string, e.g. ``"cuda:0"``.
  n_clusters : int
      Number of clusters for the default k-means ``seg_fn``.
  n_surface_pts : int
      Number of surface samples used to build the triplane representation.
  """

  def __init__(
      self,
      checkpoint_path: str,
      device: str = "cuda",
      n_clusters: int = 10,
      n_surface_pts: int = 100_000,
      adaptive_kwargs: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.n_clusters = n_clusters
    self.n_surface_pts = n_surface_pts
    # Extra keyword args forwarded to ``adaptive_part_segmentation_torch``
    # when using :meth:`_iou_seg_fn` (e.g. ``{"sim_hi": 0.8, "max_parts": 30}``).
    # NOTE: that routine's spatial thresholds (``seed_dbscan_eps_space``,
    # ``merge_space_thresh`` …) are metric, so coords are passed through
    # unmodified — tune these here for non-metric clouds.
    self.adaptive_kwargs = dict(adaptive_kwargs) if adaptive_kwargs else {}
    super().__init__(checkpoint_path, device)

  # ── Seg3D_Base hooks ──────────────────────────────────────────────────

  def _build_model(self) -> torch.nn.Module:
    _ensure_partfield_path()
    from partfield.model_trainer_pvcnn_only_demo import Model  # noqa: PLC0415
    cfg = _build_default_cfg()
    model = Model(cfg)
    ckpt = torch.load(
        str(self.checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
      print(f"[PartFieldModule] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
      print(
          f"[PartFieldModule] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …"
      )
    return model

  @torch.no_grad()
  def extract_features(self, data3d: Dict[str, Any]) -> np.ndarray:
    """Return per-point PartField features (point-cloud input only).

    Parameters
    ----------
    data3d : dict
        Point-cloud mode: ``{"points": (N, 3)}``
        → returns ``(N, 448)`` per-point features for **all** N points.

    Returns
    -------
    features : (N, 448) float32 np.ndarray

    Raises
    ------
    ValueError
        If mesh input (``vertices``/``faces``) is supplied — this module
        only supports point clouds.
    """
    _ensure_partfield_path()
    from partfield.model.PVCNN.encoder_pc import sample_triplane_feat  # noqa: PLC0415

    if "vertices" in data3d or "faces" in data3d:
      raise ValueError(
          "PartFieldModule supports point-cloud input only; got mesh keys "
          "(vertices/faces). Pass data3d={'points': (N, 3)} instead.")
    if "points" not in data3d:
      raise ValueError("PartFieldModule expects data3d={'points': (N, 3)}.")

    pts = np.asarray(data3d["points"], dtype=np.float32)
    center, scale = _bbox_normalize_params(pts)
    pts_norm = (pts - center) * scale  # (N, 3) all points normalised
    # Use a subsample for triplane building (cap at n_surface_pts)
    if len(pts_norm) > self.n_surface_pts:
      rng = np.random.default_rng(0)
      idx = rng.choice(len(pts_norm), self.n_surface_pts, replace=False)
      surface_pts = pts_norm[idx]
    else:
      surface_pts = pts_norm
    # Query at ALL normalised positions so every input point gets a label
    query_pts = pts_norm

    # ── triplane forward ───────────────────────────────────────────────
    pc_t = torch.tensor(surface_pts, dtype=torch.float32,
                        device=self.device).unsqueeze(0)
    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
      pc_feat = self.model.pvcnn(pc_t, pc_t)
      planes = self.model.triplane_transformer(pc_feat)
    _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
    part_planes = part_planes.float(
    )  # autocast may yield fp16; grid_sample needs fp32

    # ── sample at query positions ──────────────────────────────────────
    # sample in chunks to avoid OOM on large point clouds
    chunk = self.model.cfg.n_sample_each
    q_all = torch.tensor(query_pts, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
    N_q = q_all.shape[1]
    feat_chunks = []
    for start in range(0, N_q, chunk):
      q_chunk = q_all[:, start:start + chunk]
      f_chunk = sample_triplane_feat(part_planes, q_chunk)  # (1, M, C)
      feat_chunks.append(f_chunk)
    features = torch.cat(feat_chunks, dim=1)  # (1, N_q, C)
    return features.squeeze(0).cpu().float().numpy()  # (N_q, 448)

  def seg_3d(
      self,
      data3d: Dict[str, Any],
      seg_fn: Optional[Callable] = None,
  ) -> Dict[str, Any]:
    features = self.extract_features(data3d)
    # fn = seg_fn if seg_fn is not None else self._default_seg_fn
    fn = seg_fn if seg_fn is not None else self._iou_seg_fn
    labels = fn(features, data3d)
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "features": features,
    }

  # ── built-in seg_fn ───────────────────────────────────────────────────

  def _default_seg_fn(
      self,
      features: np.ndarray,
      data3d: Dict[str, Any],
  ) -> np.ndarray:
    """K-means clustering on L2-normalised features."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    feat_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) +
                            1e-8)
    km = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
    return km.fit_predict(feat_norm).astype(np.int32)

  def _iou_seg_fn(
      self,
      features: np.ndarray,
      data3d: Dict[str, Any],
  ) -> np.ndarray:
    """Adaptive part segmentation as a ``seg_fn`` (point-cloud only).

    Bridges :func:`module.utils.part_seg_utils.adaptive_part_segmentation_torch`
    to the ``seg_fn(features, data3d) -> labels`` contract.  Unlike
    :meth:`_default_seg_fn`, the part count is discovered automatically by
    region-growing followed by prototype + overlap (IoU) merging, rather than
    fixed to ``n_clusters``.  Label ``0`` denotes background (un-grown points).

    Expects point-cloud input ``{"points": (N, 3)}``: ``extract_features``
    queries every point, so ``data3d["points"]`` aligns 1:1 with ``features``
    and serves directly as the region-growing coordinates ``Pi`` (left in its
    original metric space — the routine's distance thresholds are metric).

    Optional ``data3d["prompt"]`` ``(M, 3)`` seeds the growth at user-supplied
    points (the routine maps each to its nearest ``Pi``); otherwise FPS seeds
    are used.  Raw (un-normalised) features are forwarded — the adaptive
    routine L2-normalises internally.
    """
    from module.utils.part_seg_utils import (  # noqa: PLC0415
        adaptive_part_segmentation_torch,)

    if "points" not in data3d:
      raise ValueError("_iou_seg_fn supports point-cloud input only; "
                       "expected data3d['points'].")
    Pi = np.ascontiguousarray(data3d["points"], dtype=np.float32)
    if Pi.shape[0] != features.shape[0]:
      raise ValueError(
          f"_iou_seg_fn: points ({Pi.shape[0]}) and features "
          f"({features.shape[0]}) length mismatch.")

    Fi = np.ascontiguousarray(features, dtype=np.float32)
    prompt = data3d.get("prompt")
    if prompt is not None:
      prompt = np.asarray(prompt, dtype=np.float32)

    labels, _parts, _info = adaptive_part_segmentation_torch(
        Pi,
        Fi,
        prompt_i=prompt,
        **self.adaptive_kwargs,
    )
    return np.asarray(labels, dtype=np.int32)
