import os

import numpy as np
import torch
import open3d as o3d
from tqdm.auto import tqdm

import pyvista as pv

from pathlib import Path
from typing import Iterable, Optional, Union

ArrayLike = Union[np.ndarray, Iterable[Iterable[float]]]


def export_vectors_to_ply(
    vectors: ArrayLike,
    out_path: Union[str, Path],
    mode: str = "lines",  # "normals" | "lines"
    scale: float = 1.0,
    origin: Iterable[float] = (0.0, 0.0, 0.0),
    color: Optional[Iterable[float]] = None,
    binary: bool = True,
) -> str:

  vectors = np.asarray(vectors, dtype=np.float64)
  assert vectors.ndim == 2 and vectors.shape[1] == 3, "vectors 需为 (n,3)"

  out_path = str(Path(out_path))

  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  nonzero = norms[:, 0] > 1e-12
  safe_norms = np.where(nonzero[:, None], norms, 1.0)
  dirs = vectors / safe_norms
  dirs[~nonzero] = np.array([0.0, 0.0, 1.0])

  if mode == "normals":
    pts = vectors * float(scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(dirs)
    if color is not None:
      color = np.asarray(color, dtype=np.float64).reshape(1, 3)
      colors = np.repeat(color, pts.shape[0], axis=0)
      pcd.colors = o3d.utility.Vector3dVector(colors.clip(0, 1))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=not binary)
    return out_path

  elif mode == "lines":
    o_raw = np.asarray(origin, dtype=np.float64)
    if o_raw.ndim == 1:
      o_raw = o_raw.reshape(1, 3)
    elif o_raw.ndim != 2 or o_raw.shape[1] != 3:
      raise ValueError("origin 需要是 (3,) 或 (n,3)")

    n = vectors.shape[0]
    if o_raw.shape[0] == 1:
      origins = np.repeat(o_raw, n, axis=0)
    elif o_raw.shape[0] == n:
      origins = o_raw
    else:
      raise ValueError("origin 行数需与向量数量一致")

    verts = np.empty((n * 2, 3), dtype=np.float64)
    verts[0::2] = origins
    verts[1::2] = origins + vectors * float(scale)
    lines = np.column_stack([np.arange(0, 2 * n, 2),
                             np.arange(1, 2 * n, 2)]).astype(np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    if color is not None:
      color = np.asarray(color, dtype=np.float64).reshape(1, 3).clip(0, 1)
      ls.colors = o3d.utility.Vector3dVector(np.repeat(color, n, axis=0))
    o3d.io.write_line_set(out_path, ls, write_ascii=not binary)
    return out_path

  else:
    raise ValueError("mode 只能是 'normals' 或 'lines'")
