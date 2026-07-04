import os

import numpy as np
import torch
import open3d as o3d
from tqdm.auto import tqdm

import pyvista as pv


def calculate_obb_pv(
    pointcloud: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


#
# [2]
#
from pathlib import Path
from typing import Iterable, Optional, Union

ArrayLike = Union[np.ndarray, Iterable[Iterable[float]]]


def export_vectors_to_ply(
    vectors: ArrayLike,
    out_path: Union[str, Path],
    mode: str = "lines",  # "normals" | "lines"
    scale: float = 1.0,  # vector scale factor
    origin: Iterable[float] = (
        0.0, 0.0, 0.0),  # only valid for mode="lines", can be (n,3)
    color: Optional[Iterable[float]] = None,  # RGB(0~1), optional
    binary: bool = True,  # whether to write binary PLY
) -> str:
  """
    Export (n,3) vectors to PLY for visualization in MeshLab.
    - mode="normals": Use point cloud + vertex normals to represent vectors (recommended; MeshLab displays normals most stably)
      Point positions = vectors * scale; normals = normalized directions
    - mode="lines":   Use LineSet to draw lines from origin to origin + vectors*scale
      origin can be a single (3,) or the same number of (n,3) as vectors

    Returns the written file path as a string.
    """

  vectors = np.asarray(vectors, dtype=np.float64)
  assert vectors.ndim == 2 and vectors.shape[1] == 3, "vectors must be (n,3)"

  out_path = str(Path(out_path))

  # directions and lengths
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  nonzero = norms[:, 0] > 1e-12
  safe_norms = np.where(nonzero[:, None], norms, 1.0)
  dirs = vectors / safe_norms  # normalized directions
  dirs[~nonzero] = np.array(
      [0.0, 0.0, 1.0])  # placeholder direction for zero vectors to avoid NaN

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
    # Use lines to display: each vector is a line from origin to origin + v*scale
    o_raw = np.asarray(origin, dtype=np.float64)
    if o_raw.ndim == 1:
      o_raw = o_raw.reshape(1, 3)
    elif o_raw.ndim != 2 or o_raw.shape[1] != 3:
      raise ValueError("origin must be (3,) or (n,3)")

    n = vectors.shape[0]
    if o_raw.shape[0] == 1:
      origins = np.repeat(o_raw, n, axis=0)
    elif o_raw.shape[0] == n:
      origins = o_raw
    else:
      raise ValueError("origin must have the same number of rows as vectors")

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
    raise ValueError("mode must be 'normals' or 'lines'")


def normalize(v):
  n = np.linalg.norm(v)
  return v if n < 1e-12 else v / n


def rodrigues_rotation_matrix(u, theta):
  """Rodrigues rotation matrix, rotate around unit vector u by theta"""
  u = normalize(u)
  K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
  I = np.eye(3)
  return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def best_to_o3d_Rt(best, scale=1.0):
  """
    Convert the best dictionary from estimate_constrained_motion to R, tvec
    - Translation: X' = X + t*u  ->  R = I,  tvec = t*u
    - Rotation: X' = R(X - a) + a -> R = Rodrigues(u, omega), tvec = a - R@a
    """
  if best["mode"] == "trans":
    u = np.asarray(best["u"])
    t = np.asarray(best["t"]) * scale
    R = np.eye(3)
    tvec = t * u
  elif best["mode"] == "rot":
    u = np.asarray(best["u"])
    omega = best["omega"] * scale
    a = np.asarray(best["a_point"])
    R = rodrigues_rotation_matrix(u, omega)
    tvec = a - R @ a
  else:
    raise ValueError(f"Unknown mode: {best['mode']}")
  return R, tvec


def best_to_o3d_Rct(best, scale=1.0):
  """
    Convert the best dictionary from estimate_constrained_motion to R, c, tvec
    - Translation: X' = X + t*u  ->  R = I,  tvec = t*u, c defaults to zero vector
    - Rotation: X' = R(X - c) + c + t -> R = Rodrigues(u, omega), t defaults to zero vector
    """
  mode = best["mode"]

  def _scalar_or_vec_t(t_value, axis):
    t_arr = np.asarray(t_value) * scale
    if t_arr.shape == (3,):
      return t_arr
    return float(t_arr) * axis

  if mode == "trans":
    u = np.asarray(best["u"])
    t = best.get("t", 0.0)
    R = np.eye(3)
    tvec = _scalar_or_vec_t(t, u)
    a = np.asarray(best.get("a_point", np.zeros(3)))
  elif mode == "rot":
    u = np.asarray(best["u"])
    omega = best.get("omega", 0.0)
    a = np.asarray(best.get("a_point", np.zeros(3)))
    R = rodrigues_rotation_matrix(u, omega * scale)
    tvec = np.zeros(3)

  else:
    raise ValueError(f"Unknown mode: {mode}")

  return R, a, tvec


def best_to_art_Rct(best, scale=1.0):
  """
    Convert the best dictionary from estimate_constrained_motion to R, c, tvec
    - Translation: X' = X + t*u  ->  R = I,  tvec = t*u, c defaults to zero vector
    - Rotation: X' = (X - c)R + c + t -> R = Rodrigues(u, omega), t defaults to zero vector
    """
  R_o3d, c, t = best_to_o3d_Rct(best, scale=scale)

  return R_o3d.T, c, t


def best_to_o3d_T(best):
  R, tvec = best_to_o3d_Rt(best)

  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = tvec
  return T


def _rotation_matrix_to_axis_angle(R, eps=1e-8):
  R = np.asarray(R)
  trace = np.trace(R)
  if trace > 0.0:
    s = np.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (R[2, 1] - R[1, 2]) / s
    qy = (R[0, 2] - R[2, 0]) / s
    qz = (R[1, 0] - R[0, 1]) / s
  else:
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
      s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], eps)) * 2.0
      qw = (R[2, 1] - R[1, 2]) / s
      qx = 0.25 * s
      qy = (R[0, 1] + R[1, 0]) / s
      qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
      s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], eps)) * 2.0
      qw = (R[0, 2] - R[2, 0]) / s
      qx = (R[0, 1] + R[1, 0]) / s
      qy = 0.25 * s
      qz = (R[1, 2] + R[2, 1]) / s
    else:
      s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], eps)) * 2.0
      qw = (R[1, 0] - R[0, 1]) / s
      qx = (R[0, 2] + R[2, 0]) / s
      qy = (R[1, 2] + R[2, 1]) / s
      qz = 0.25 * s
  quat = np.array([qw, qx, qy, qz], dtype=np.float64)
  norm = np.linalg.norm(quat)
  if norm < eps:
    return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
  quat /= norm
  qw, qx, qy, qz = quat
  qw = np.clip(qw, -1.0, 1.0)
  angle = 2.0 * np.arccos(qw)
  sin_half = np.sqrt(max(1.0 - qw * qw, 0.0))
  if sin_half < eps:
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  else:
    axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half
  axis_norm = np.linalg.norm(axis)
  if axis_norm < eps:
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  else:
    axis = axis / axis_norm
  return axis, angle


def apply_motion_scale_to_rct(R, c, t, motion_scale):
  if motion_scale is None or np.isclose(motion_scale, 1.0):
    return R, c, t
  R_arr = np.asarray(R)
  t_arr = np.asarray(t)
  dtype_R = R_arr.dtype
  dtype_t = t_arr.dtype
  axis, angle = _rotation_matrix_to_axis_angle(R_arr)
  scaled_angle = angle * motion_scale
  R_scaled = rodrigues_rotation_matrix(axis, scaled_angle)
  R_scaled = R_scaled.astype(dtype_R, copy=False)
  t_scaled = (t_arr.astype(np.float64) * motion_scale).astype(dtype_t,
                                                              copy=False)
  c_arr = np.asarray(c)
  if np.issubdtype(c_arr.dtype, np.floating):
    c_arr = c_arr.astype(dtype_R, copy=False)
  return R_scaled, c_arr, t_scaled


#
# [3]
#
def to_torch(x, device, dtype=torch.float32):
  if isinstance(x, np.ndarray):
    return torch.from_numpy(x).to(device=device, dtype=dtype)
  return x.to(device=device, dtype=dtype)


def normalize_torch(v):
  n = torch.linalg.norm(v)
  return v if n < 1e-12 else v / n


def plane_basis_torch(u):
  u = normalize_torch(u)
  tmp = (torch.tensor([1.0, 0.0, 0.0], device=u.device, dtype=u.dtype)
         if torch.abs(u[0]) < 0.9 else torch.tensor(
             [0.0, 1.0, 0.0], device=u.device, dtype=u.dtype))
  e1 = torch.nn.functional.normalize(torch.cross(u, tmp), dim=0)
  e2 = torch.cross(u, e1)
  return e1, e2


def project_to_plane_coords_torch(X, u):
  e1, e2 = plane_basis_torch(u)
  X2 = torch.stack([X @ e1, X @ e2], dim=1)  # (N,2)
  return X2, e1, e2


def kabsch_2d_torch(X, Y, weights=None):
  # X,Y: (N,2)
  if weights is None:
    w = torch.ones((X.shape[0],), device=X.device, dtype=X.dtype)
  else:
    w = weights.reshape(-1).to(device=X.device, dtype=X.dtype)
  wsum = torch.clamp(w.sum(), min=1e-12)
  Xc = (w[:, None] * X).sum(dim=0) / wsum
  Yc = (w[:, None] * Y).sum(dim=0) / wsum
  X0, Y0 = X - Xc, Y - Yc
  H = X0.T @ (w[:, None] * Y0)  # (2,2)
  U, S, Vh = torch.linalg.svd(H)
  R = Vh.T @ U.T
  if torch.det(R) < 0:
    Vh[1, :] *= -1
    R = Vh.T @ U.T
  t = Yc - R @ Xc
  return R, t


def rodrigues_rotation_matrix_torch(u, theta):
  u = normalize_torch(u)
  ux, uy, uz = u
  K = torch.tensor([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
                   device=u.device,
                   dtype=u.dtype)
  I = torch.eye(3, device=u.device, dtype=u.dtype)
  theta_t = torch.as_tensor(theta, device=u.device, dtype=u.dtype)
  s = torch.sin(theta_t)
  c = torch.cos(theta_t)
  return I + s * K + (1 - c) * (K @ K)


def transform_rotation_about_axis_torch(P, u, theta, a_point):
  R = rodrigues_rotation_matrix_torch(u, theta)
  return (P - a_point) @ R.T + a_point


# ---------- Torch KNN ----------
def knn_nn_torch(P, Q, batch_P=8192, chunk_Q=8192):
  """
    Return (Q_match, dists); both are torch tensors (Q_match on the same device as P)
    - P: (N,3), Q: (M,3)
    - batch_P / chunk_Q control memory usage
    """
  try:
    from pytorch3d.ops import knn_points

    d2, idx, _ = knn_points(P[None], Q[None], K=1, return_nn=False)  # (1,N,1)
    idx = idx[0, :, 0]
    d2 = d2[0, :, 0]
    return Q[idx], torch.sqrt(torch.clamp(d2, min=0))
  except Exception:
    pass

  device = P.device
  dtype = P.dtype
  N, M = P.shape[0], Q.shape[0]
  nn_idx = torch.empty(N, dtype=torch.long, device=device)
  nn_d2 = torch.full((N,), float("inf"), device=device, dtype=dtype)

  for i in range(0, N, batch_P):
    P_b = P[i:i + batch_P]  # (B,3)
    B = P_b.shape[0]
    best_d2 = torch.full((B,), float("inf"), device=device, dtype=dtype)
    best_idx = torch.zeros((B,), dtype=torch.long, device=device)
    p_norm = (P_b * P_b).sum(dim=1)  # (B,)

    for j in range(0, M, chunk_Q):
      Q_c = Q[j:j + chunk_Q]  # (m,3)
      q_norm = (Q_c * Q_c).sum(dim=1)  # (m,)
      # d^2 = ||p||^2 + ||q||^2 - 2 p·q
      d2 = p_norm[:, None] + q_norm[None, :] - 2.0 * (P_b @ Q_c.T)  # (B,m)
      d2 = torch.clamp(d2, min=0)
      d2_min, idx_local = d2.min(dim=1)  # (B,)
      mask = d2_min < best_d2
      best_d2[mask] = d2_min[mask]
      best_idx[mask] = idx_local[mask] + j

    nn_d2[i:i + B] = best_d2
    nn_idx[i:i + B] = best_idx

  Q_match = Q[nn_idx]
  return Q_match, torch.sqrt(nn_d2)


def residual_nn_torch(P1, P, **knn_kwargs):
  Q, _ = knn_nn_torch(P1, P, **knn_kwargs)
  d2 = ((P1 - Q)**2).sum(dim=1)
  # return d2.mean()
  return d2.sqrt().sqrt().mean()

  # l1_dists = torch.abs(P1 - Q).sum(dim=1)
  # return l1_dists.mean()


def bbox_center_size_torch(P):
  m = P.min(dim=0).values
  M = P.max(dim=0).values
  return 0.5 * (m + M), (M - m)


# ======== SO3 toolkit ========
def skew_matrix_torch(v: torch.Tensor) -> torch.Tensor:
  # v: (...,3) -> (...,3,3)
  vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
  O = torch.zeros_like(vx)
  return torch.stack(
      [
          torch.stack([O, -vz, vy], dim=-1),
          torch.stack([vz, O, -vx], dim=-1),
          torch.stack([-vy, vx, O], dim=-1),
      ],
      dim=-2,
  )


def axis_angle_to_matrix_torch(phi: torch.Tensor) -> torch.Tensor:
  # phi: (3,) or (B,3)
  single = phi.ndim == 1
  if single:
    phi = phi[None, :]
  theta = torch.linalg.norm(phi, dim=-1, keepdim=True)  # (B,1)
  K = skew_matrix_torch(phi / torch.clamp(theta, min=1e-12))
  I = torch.eye(3, device=phi.device,
                dtype=phi.dtype).expand(phi.shape[0], 3, 3)
  s = torch.sin(theta)[..., None]  # (B,1,1)
  c = torch.cos(theta)[..., None]
  R = I + s * K + (1 - c) * (K @ K)
  return R[0] if single else R


def skew_batch_torch(X: torch.Tensor) -> torch.Tensor:
  # X: (N,3) -> (N,3,3)
  x, y, z = X[:, 0], X[:, 1], X[:, 2]
  O = torch.zeros_like(x)
  return torch.stack(
      [
          torch.stack([O, -z, y], dim=-1),
          torch.stack([z, O, -x], dim=-1),
          torch.stack([-y, x, O], dim=-1),
      ],
      dim=-2,
  )


# ======== Coarse Search ========
@torch.no_grad()
def search_translation_coarse_torch(Pm, P, u, t_range, coarse=40, **knn_kwargs):
  tmin, tmax = t_range
  ts = torch.linspace(tmin, tmax, coarse, device=Pm.device, dtype=Pm.dtype)
  best_R, best_t = float("inf"), 0
  for t in ts.tolist():
    Rv = residual_nn_torch(Pm + t * u, P, **knn_kwargs).item()
    if Rv < best_R:
      best_R, best_t = Rv, t
  return float(best_t), float(best_R)


@torch.no_grad()
def search_rotation_coarse_torch(Pm,
                                 P,
                                 u,
                                 a_points,
                                 omega_range,
                                 coarse=40,
                                 **knn_kwargs):
  omin, omax = omega_range
  omegas = torch.linspace(omin, omax, coarse, device=Pm.device, dtype=Pm.dtype)
  best_R, best_omega, best_a = float("inf"), 0, None
  for a in a_points:
    for omega in omegas.tolist():
      Pm_t = transform_rotation_about_axis_torch(Pm, u, float(omega), a)
      Rv = residual_nn_torch(Pm_t, P, **knn_kwargs).item()
      if Rv < best_R:
        best_R, best_omega, best_a = Rv, omega, a.clone()
  return float(
      best_omega), best_a if best_a is not None else a_points[0], float(best_R)


# ======== Refine: Axial Translation (Least Squares GN/LM, optimize u and t)========
@torch.no_grad()
def refine_translation_ls_torch(
    Pm,
    P,
    u_init,
    t_init,
    steps=15,
    lm_lambda=1e-1,
    lm_up=10.0,
    lm_down=0.3,
    **knn_kwargs,
):
  u = normalize_torch(u_init.clone())
  t = torch.as_tensor(float(t_init), device=Pm.device, dtype=Pm.dtype)
  I3 = torch.eye(3, device=Pm.device, dtype=Pm.dtype)

  def loss_of(u_, t_):
    Pm_t = Pm + t_ * u_
    Q, _ = knn_nn_torch(Pm_t, P, **knn_kwargs)
    r = Pm_t - Q
    return (r * r).sum() / Pm.shape[0], Q, r

  L, Q, r = loss_of(u, t)
  lam = torch.as_tensor(lm_lambda, device=Pm.device, dtype=Pm.dtype)

  for _ in range(steps):
    # Orthogonal basis
    tmp = (torch.tensor([1.0, 0.0, 0.0], device=Pm.device, dtype=Pm.dtype)
           if torch.abs(u[0]) < 0.9 else torch.tensor(
               [0.0, 1.0, 0.0], device=Pm.device, dtype=Pm.dtype))
    e1 = torch.nn.functional.normalize(torch.cross(u, tmp), dim=0)
    e2 = torch.cross(u, e1)

    # 3x3 constant Jacobian column vectors
    B1 = t * e1  # for α
    B2 = t * e2  # for β
    B3 = u  # for δt

    r_sum = r.sum(dim=0)  # (3,)
    # H, g
    N = Pm.shape[0]
    H = torch.stack([
        torch.stack([N * (B1 @ B1), N * (B1 @ B2), N * (B1 @ B3)]),
        torch.stack([N * (B2 @ B1), N * (B2 @ B2), N * (B2 @ B3)]),
        torch.stack([N * (B3 @ B1), N * (B3 @ B2), N * (B3 @ B3)]),
    ])
    g = torch.stack([B1 @ r_sum, B2 @ r_sum, B3 @ r_sum])

    # LM damping
    H_damped = H + lam * torch.eye(3, device=Pm.device, dtype=Pm.dtype)
    try:
      delta = -torch.linalg.solve(H_damped, g)  # [α, β, δt]
    except RuntimeError:
      delta = -torch.linalg.pinv(H_damped) @ g

    alpha, beta, dt = delta
    u_candidate = normalize_torch(u + alpha * e1 + beta * e2)
    t_candidate = t + dt

    L_new, Q_new, r_new = loss_of(u_candidate, t_candidate)
    if L_new < L:
      u, t, Q, r, L = u_candidate, t_candidate, Q_new, r_new, L_new
      lam = lam * lm_down
    else:
      lam = lam * lm_up

  Pm_final = Pm + t * u
  R_final = residual_nn_torch(Pm_final, P, **knn_kwargs).item()
  return u, float(t), float(R_final)


# ======== Refine: Rotation About Axis (Least Squares GN/LM, optimize φ=ωu and a)========
@torch.no_grad()
def refine_rotation_ls_torch(
    Pm,
    P,
    u_init,
    omega_init,
    a_init,
    steps=15,
    lm_lambda=1e-1,
    lm_up=10.0,
    lm_down=0.3,
    omega_range=None,
    **knn_kwargs,
):

  # Parameterize with axis-angle vector φ = ω u; allow slight adjustment of φ's direction (i.e., update u)
  phi = float(omega_init) * normalize_torch(u_init.clone())
  a = a_init.clone()
  lam = torch.as_tensor(lm_lambda, device=Pm.device, dtype=Pm.dtype)
  N = Pm.shape[0]
  I3 = torch.eye(3, device=Pm.device, dtype=Pm.dtype)

  def forward_and_loss(phi_, a_):
    R = axis_angle_to_matrix_torch(phi_)
    Pm_t = (Pm - a_) @ R.T + a_
    Q, _ = knn_nn_torch(Pm_t, P, **knn_kwargs)
    r = Pm_t - Q  # (N,3)
    loss = (r * r).sum() / N
    return loss, R, Pm_t, Q, r

  L, R, Pm_t, Q, r = forward_and_loss(phi, a)

  for _ in range(steps):
    # Current axis direction (used to remove the unidentifiable component of a along the axis)
    th = torch.linalg.norm(phi)
    u_cur = ((phi / torch.clamp(th, min=1e-12))
             if th > 1e-12 else normalize_torch(u_init.clone()))

    # Per-point Y = R(X - a) = Pm_t - a
    Y = Pm_t - a  # (N,3)
    S_y = skew_batch_torch(Y)  # (N,3,3)

    # Jacobian wrt φ: δr ≈ - [Y]_x R δφ    (left-multiplicative perturbation, world-frame axis-angle)
    A = -torch.einsum("nij,jk->nik", S_y, R)  # (N,3,3)

    # Jacobian wrt a: δr ≈ (I - R) δa
    B = I3 - R  # (3,3)  constant (point-independent)

    # Assemble normal equations H δ = -g, H = Σ JᵀJ, g = Σ Jᵀ r
    H_aa = torch.einsum("nji,njk->ik", A, A)  # Σ A_iᵀ A_i   (3,3)
    A_T_sum = A.transpose(1, 2).sum(dim=0)  # Σ A_iᵀ       (3,3)
    H_ab = A_T_sum @ B  # Σ A_iᵀ B     (3,3)
    H_bb = N * (B.T @ B)  # Σ Bᵀ B = N BᵀB (3,3)

    g_a = torch.einsum("nji,nj->i", A, r)  # Σ A_iᵀ r_i   (3,)
    r_sum = r.sum(dim=0)  # Σ r_i        (3,)
    g_b = B.T @ r_sum  # Σ Bᵀ r_i     (3,)

    H_top = torch.cat([H_aa, H_ab], dim=1)  # (3,6)
    H_bot = torch.cat([H_ab.T, H_bb], dim=1)  # (3,6)
    H = torch.cat([H_top, H_bot], dim=0)  # (6,6)
    g = torch.cat([g_a, g_b], dim=0)  # (6,)

    # Levenberg-Marquardt damping
    H_damped = H + lam * torch.eye(6, device=Pm.device, dtype=Pm.dtype)
    try:
      delta = -torch.linalg.solve(H_damped, g)
    except RuntimeError:
      delta = -torch.linalg.pinv(H_damped) @ g

    dphi = delta[:3]
    da = delta[3:]

    # Remove the unidentifiable component of da along the axis ((I - R)u = 0)
    da = da - (da @ u_cur) * u_cur

    # Prevent excessively large rotation steps (10°)
    max_step = np.deg2rad(10.0)
    dphi_norm = torch.linalg.norm(dphi)
    if dphi_norm > max_step:
      dphi = dphi * (max_step / dphi_norm)

    # Lightweight backtracking line search to improve acceptance rate and prevent λ from increasing indefinitely
    accepted = False
    alpha = 1.0
    for _ls in range(5):
      phi_cand = phi + alpha * dphi
      a_cand = a + alpha * da

      # Angle range clipping (only limit the upper bound; the lower bound is absorbed by the direction)
      if omega_range is not None:
        omin, omax = omega_range
        th_cand = torch.linalg.norm(phi_cand)
        if th_cand > (omax + 1e-6):
          phi_cand = phi_cand * (omax / th_cand)

      L_new, R_new, Pm_t_new, Q_new, r_new = forward_and_loss(phi_cand, a_cand)
      if L_new < L:
        phi, a = phi_cand, a_cand
        R, Pm_t, Q, r, L = R_new, Pm_t_new, Q_new, r_new, L_new
        lam = lam * lm_down
        accepted = True
        break
      alpha *= 0.5

    if not accepted:
      lam = lam * lm_up  # Step back, increase damping

  omega = float(torch.linalg.norm(phi))
  u = normalize_torch(u_init.clone()) if omega < 1e-8 else (phi / omega)
  R_final = residual_nn_torch(Pm_t, P, **knn_kwargs).item()
  return u, omega, a, float(R_final)


@torch.no_grad()
def estimate_constrained_motion_torch(
    Pm_np,
    P_np,
    axes_np,
    center,
    extent,
    device="cuda",
    dtype=torch.float32,
    omega_range=(-np.pi * 0.45, np.pi * 0.45),
    max_trans=0.4,
    downsample_keep=100_000,
    knn_batch_P=20000,
    knn_chunk_Q=50000,
):
  print(f"P_m shape: {Pm_np.shape}, P shape: {P_np.shape}")
  dev = torch.device(device)
  Pm = to_torch(Pm_np, dev, dtype)
  P = to_torch(P_np, dev, dtype)
  axes = [to_torch(np.asarray(u), dev, dtype) for u in axes_np]

  if not torch.is_tensor(center):
    center = to_torch(np.asarray(center), dev, dtype)
  if not torch.is_tensor(extent):
    extent = to_torch(np.asarray(extent), dev, dtype).reshape(-1, 1)
  extent = 2.0 * extent  # convert half-extent to full extent

  # Downsampling
  if Pm.shape[0] > downsample_keep:
    print(f"Downsampling Pm from {Pm.shape[0]} to {downsample_keep}")
    idx = torch.randperm(Pm.shape[0], device=dev)[:downsample_keep]
    Pm = Pm[idx]
  MAX_DOWN_SAMPLED = int(downsample_keep)
  if P.shape[0] > MAX_DOWN_SAMPLED:
    print(f"Downsampling P from {P.shape[0]} to {MAX_DOWN_SAMPLED}")
    r2 = ((P - center)**2).sum(dim=1)
    k = min(MAX_DOWN_SAMPLED, P.shape[0])
    _, idx = torch.topk(r2, k=k, largest=False, sorted=False)
    P = P[idx]

  def kk():
    return dict(batch_P=knn_batch_P, chunk_Q=knn_chunk_Q)

  base_R = residual_nn_torch(Pm, P, **kk()).item()
  results = []

  # Axis index to perpendicular direction index
  e_ij = [[1, 2], [0, 2], [0, 1]]

  for i, u0 in enumerate(axes):
    print(f"--- Processing axis {i}: {u0.detach().cpu().numpy()} ---")
    u0 = normalize_torch(u0)

    # Translation: coarse search t, then least squares refinement (u, t)
    tmin, tmax = -max_trans, max_trans
    t_seed, R_seed = search_translation_coarse_torch(Pm,
                                                     P,
                                                     u0, (tmin, tmax),
                                                     coarse=50,
                                                     **kk())
    print(f"Axis {i} (trans) seed: t={t_seed:.4f}, R={R_seed:.6f}")
    u_opt, t_opt, R_t = refine_translation_ls_torch(Pm,
                                                    P,
                                                    u0,
                                                    t_seed,
                                                    steps=15,
                                                    lm_lambda=1e-3,
                                                    **kk())
    print(f"Axis {i} (trans): t={t_opt:.4f}, R={R_t:.6f}")
    results.append(
        dict(
            mode="trans",
            axis_idx=i,
            u=u_opt.detach().cpu().numpy().tolist(),
            t=t_opt,
            residual=R_t,
            range=(tmin, tmax),
            t_seed=t_seed,
        ))

    # Rotation: construct a candidates, coarse search ω, then least squares refinement (u, ω, a)
    e_idx1, e_idx2 = e_ij[i]
    e1 = axes[e_idx1]
    e2 = axes[e_idx2]
    a0_plane = center - (center @ u0) * u0
    # Take a candidates along the two perpendicular axes of the bbox (approximate corners)
    d1 = 0.5 * (torch.abs(extent[e_idx1]).item()) * e1
    d2 = 0.5 * (torch.abs(extent[e_idx2]).item()) * e2
    a_points = [
        a0_plane,
        a0_plane + d1,
        a0_plane - d1,
        a0_plane + d2,
        a0_plane - d2,
    ]

    omega0, a0, R_r0 = search_rotation_coarse_torch(Pm,
                                                    P,
                                                    u0,
                                                    a_points,
                                                    omega_range,
                                                    coarse=50,
                                                    **kk())
    print(f"Axis {i} (rot) seed: ω={omega0:.4f} rad, R={R_r0:.6f}")
    u_rot, omega_opt, a_opt, R_r = refine_rotation_ls_torch(
        Pm,
        P,
        u0,
        omega0,
        a0,
        steps=20,
        lm_lambda=1e-3,
        omega_range=omega_range,
        **kk(),
    )

    print(f"Axis {i} (rot): ω={omega_opt:.4f} rad, R={R_r:.6f}")

    # Filter out very small angles
    if abs(omega_opt) >= 5.0 * np.pi / 180.0:
      results.append(
          dict(
              mode="rot",
              axis_idx=i,
              u=u_rot.detach().cpu().numpy().tolist(),
              omega=omega_opt,
              a_point=a_opt.detach().cpu().numpy().tolist(),
              residual=R_r,
              omega_init=omega0,
              a_init=a0.detach().cpu().numpy().tolist(),
          ))
    else:
      print("(Rotation angle too small, ignore)")

  best = sorted(results, key=lambda d: d["residual"])[0]
  return base_R, results, best


@torch.no_grad()
def estimate_constrained_motion_art(
    Pm_np,
    Ps_np,
    P_np,
    axes_np,
    center,
    extent,
    device="cuda",
    dtype=torch.float32,
    omega_range=(-np.pi * 0.45, np.pi * 0.45),
    max_trans=None,
    downsample_keep=100_000,
    knn_batch_P=20000,
    knn_chunk_Q=50000,
    refine=True,
    max_refine_a_shift=0.15,
):
  print(
      f"P_m shape: {Pm_np.shape}, P_s shape: {Ps_np.shape}, P shape: {P_np.shape}"
  )
  dev = torch.device(device)
  Pm = to_torch(Pm_np, dev, dtype)
  Ps = to_torch(Ps_np, dev, dtype)
  P = to_torch(P_np, dev, dtype)
  axes = [to_torch(np.asarray(u), dev, dtype) for u in axes_np]

  if not torch.is_tensor(center):
    center = to_torch(np.asarray(center), dev, dtype)
  if not torch.is_tensor(extent):
    extent = to_torch(np.asarray(extent), dev, dtype).reshape(-1, 1)
  extent = 2.0 * extent  # convert half-extent to full extent

  if max_trans is None:
    max_trans = 1.0 * torch.linalg.norm(extent).item()

  if Pm.shape[0] > downsample_keep:
    print(f"Downsampling Pm from {Pm.shape[0]} to {downsample_keep}")
    idx = torch.randperm(Pm.shape[0], device=dev)[:downsample_keep]
    Pm = Pm[idx]
  MAX_DOWN_SAMPLED = int(downsample_keep)
  if P.shape[0] > MAX_DOWN_SAMPLED:
    print(f"Downsampling P from {P.shape[0]} to {MAX_DOWN_SAMPLED}")
    r2 = ((P - center)**2).sum(dim=1)
    k = min(MAX_DOWN_SAMPLED, P.shape[0])
    _, idx = torch.topk(r2, k=k, largest=False, sorted=False)
    P = P[idx]
  if Ps.shape[0] > MAX_DOWN_SAMPLED:
    print(f"Downsampling Ps from {Ps.shape[0]} to {MAX_DOWN_SAMPLED}")
    r2 = ((Ps - center)**2).sum(dim=1)
    k = min(MAX_DOWN_SAMPLED, Ps.shape[0])
    _, idx = torch.topk(r2, k=k, largest=False, sorted=False)
    Ps = Ps[idx]

  def kk():
    return dict(batch_P=knn_batch_P, chunk_Q=knn_chunk_Q)

  base_R = residual_nn_torch(Pm, P, **kk()).item()
  results = []
  e_ij = [[1, 2], [0, 2], [0, 1]]

  for i, u0 in enumerate(axes):
    print(f"--- Processing axis {i}: {u0.detach().cpu().numpy()} ---")
    u0 = normalize_torch(u0)

    e_idx1, e_idx2 = e_ij[i]
    e1 = axes[e_idx1]
    e2 = axes[e_idx2]
    a0_plane = center - (center @ u0) * u0
    d1 = 0.5 * (torch.abs(extent[e_idx1]).item()) * e1
    d2 = 0.5 * (torch.abs(extent[e_idx2]).item()) * e2
    a_candidates = [
        a0_plane,
        a0_plane + d1,
        a0_plane - d1,
        a0_plane + d2,
        a0_plane - d2,
    ]

    if Ps.shape[0] > 0:
      a_stack = torch.stack([a.clone() for a in a_candidates], dim=0)
      dmat = torch.cdist(a_stack, Ps)
      min_dists = dmat.min(dim=1).values
      best_idx = torch.argmin(min_dists)
      selected_a = a_candidates[int(best_idx)]
      print(
          f"Axis {i} (rot) select a idx {int(best_idx)}, min dist {float(min_dists[best_idx]):.6f}"
      )
    else:
      selected_a = a_candidates[0]
      print(f"Axis {i} (rot) Ps empty, using first a candidate")

    tmin, tmax = -max_trans, max_trans
    t_seed, R_seed = search_translation_coarse_torch(Pm,
                                                     P,
                                                     u0, (tmin, tmax),
                                                     coarse=40,
                                                     **kk())
    print(f"Axis {i} (trans) seed: t={t_seed:.4f}, R={R_seed:.6f}")

    if refine:
      u_opt, t_opt, R_t = refine_translation_ls_torch(Pm,
                                                      P,
                                                      u0,
                                                      t_seed,
                                                      steps=10,
                                                      lm_lambda=1e-3,
                                                      **kk())
      print(f"Axis {i} (trans): t={t_opt:.4f}, R={R_t:.6f}")
    else:
      u_opt = u0
      t_opt = float(t_seed)
      Pm_t = Pm + t_opt * u_opt
      R_t = residual_nn_torch(Pm_t, P, **kk()).item()
      print(f"Axis {i} (trans no refine): t={t_opt:.4f}, R={R_t:.6f}")
    if t_opt >= tmin + 0.1 and t_opt <= tmax + 0.1:
      results.append(
          dict(
              mode="trans",
              axis_idx=i,
              u=u_opt.detach().cpu().numpy().tolist(),
              t=t_opt,
              a_point=selected_a.detach().cpu().numpy().tolist(),
              residual=R_t,
              range=(tmin, tmax),
              t_seed=t_seed,
              refined=refine,
          ))
    else:
      print("(Translation out of range, ignore)")

    omega0, a0, R_r0 = search_rotation_coarse_torch(Pm,
                                                    P,
                                                    u0, [selected_a],
                                                    omega_range,
                                                    coarse=50,
                                                    **kk())
    print(f"Axis {i} (rot) seed: ω={omega0:.4f} rad, R={R_r0:.6f}")

    if refine:
      u_rot, omega_opt, a_opt, R_r = refine_rotation_ls_torch(
          Pm,
          P,
          u0,
          omega0,
          a0,
          steps=10,
          lm_lambda=1e-3,
          omega_range=omega_range,
          **kk())

      a_shift = torch.linalg.norm(a_opt - a0).item()
      if np.isnan(R_r) or np.isinf(R_r):
        print(f"Axis {i} (rot) refined R is invalid ({R_r}), skip")
        continue
      if a_shift > max_refine_a_shift:
        print(
            f"Axis {i} (rot) refined a shift {a_shift:.6f} exceeds {max_refine_a_shift}, skip"
        )
        continue
      print(f"Axis {i} (rot): ω={omega_opt:.4f} rad, R={R_r:.6f}")
    else:
      u_rot = u0
      omega_opt = float(omega0)
      a_opt = a0
      Pm_rot = transform_rotation_about_axis_torch(Pm, u_rot, omega_opt, a_opt)
      R_r = residual_nn_torch(Pm_rot, P, **kk()).item()
      print(f"Axis {i} (rot no refine): ω={omega_opt:.4f} rad, R={R_r:.6f}")

    if abs(omega_opt) >= 10.0 * np.pi / 180.0 and abs(
        omega_opt) <= omega_range[1] + 0.5 * np.pi:
      results.append(
          dict(
              mode="rot",
              axis_idx=i,
              u=u_rot.detach().cpu().numpy().tolist(),
              omega=omega_opt,
              a_point=a_opt.detach().cpu().numpy().tolist(),
              residual=R_r,
              omega_init=omega0,
              a_init=a0.detach().cpu().numpy().tolist(),
              refined=refine,
          ))
    else:
      print("(Rotation angle too small or out of range, ignore)")

  if len(results) == 0:
    raise RuntimeError("No valid motion hypotheses found.")

  best = sorted(results, key=lambda d: d["residual"])[0]

  print(
      f"Best motion: {best['mode']} on axis {best['axis_idx']}, residual={best['residual']:.6f}"
  )

  return base_R, results, best
