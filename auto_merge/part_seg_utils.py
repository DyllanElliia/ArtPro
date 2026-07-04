import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import deque
from scipy.spatial import cKDTree

from tqdm.auto import tqdm

import open3d as o3d
import torch
from pytorch3d.ops import knn_points

import numpy as np

# -------------------------
# Toolkit
# -------------------------


def knn_median_scale_p3d(
    Pi: np.ndarray,
    knn_k: int,
    use_cuda: bool = True,
    chunk_size: int | None = None,
    return_numpy: bool = True,
):
  """KNN + median scale
    Inputs:
      - Pi: (N,3) numpy
      - knn_k: int, number of neighbors
      - use_cuda: bool, whether to use GPU
      - chunk_size: int | None, chunk size for processing large N
      - return_numpy: bool, whether to return numpy arrays
    Outputs:
      - d_knn: (N, k) Euclidean distances (including self, first column ≈ 0)
      - nn_idx: (N, k) indices of neighbors in Pi
      - per_node_med: (N,) median distance excluding self
    """
  assert Pi.ndim == 2 and Pi.shape[1] == 3, "Pi must be (N,3)"
  N = Pi.shape[0]
  if N == 0:
    raise ValueError("Pi is empty")
  # As in the original code: k cannot exceed N
  k = int(min(knn_k, N))
  if k < 1:
    raise ValueError("knn_k must be >= 1")

  device = (torch.device("cuda") if
            (use_cuda and torch.cuda.is_available()) else torch.device("cpu"))

  # Target set (entire cloud, as the library)
  P_all = (torch.as_tensor(Pi, dtype=torch.float32,
                           device=device).view(1, N, 3).contiguous())

  d_list = []
  i_list = []
  med_list = []

  # Query set is chunked by rows (target is not chunked) to prevent GPU memory overflow
  if chunk_size is None:
    chunk_size = N  # No chunking

  start = 0
  while start < N:
    end = min(start + chunk_size, N)
    Q = P_all[:, start:end, :]  # (1, n_chunk, 3)

    # knn_points return **squared distances**; sorted in ascending order, the first is usually self (≈0)
    # Here K=k, behavior is consistent with sklearn (including self)
    d2, idx, _ = knn_points(Q, P_all, K=k, return_nn=False)  # (1, n_chunk, k)
    d = torch.sqrt(torch.clamp(d2, min=0.0))  # Euclidean distance

    # Compute median excluding self
    if k >= 2:
      # d[:, :, 0] ~ self distance (≈0), aligns with original np.median(d_knn[:, 1:], axis=1)
      med = torch.median(d[0, :, 1:], dim=-1).values  # (n_chunk,)
    else:
      # k==1 has no "neighbors", can set to 0 or nan; here set to 0 for consistency
      med = torch.zeros((end - start,), dtype=d.dtype, device=d.device)

    d_list.append(d[0])  # (n_chunk, k)
    i_list.append(idx[0])  # (n_chunk, k)
    med_list.append(med)  # (n_chunk,)

    start = end

  d_knn = torch.cat(d_list, dim=0)  # (N, k)
  nn_idx = torch.cat(i_list, dim=0)  # (N, k)
  per_node_med = torch.cat(med_list, dim=0)  # (N,)

  if return_numpy:
    return (
        d_knn.detach().cpu().numpy(),
        nn_idx.detach().cpu().numpy().astype(np.int64),
        per_node_med.detach().cpu().numpy(),
    )
  else:
    return d_knn, nn_idx, per_node_med


def _l2norm(x, eps=1e-12):
  return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _cos(a: np.ndarray, b: np.ndarray):
  return float(np.dot(a, b))


def _pair_min_dist(A: np.ndarray, B: np.ndarray) -> float:
  if A.size == 0 or B.size == 0:
    return np.inf
  t = cKDTree(B)
  d, _ = t.query(A, k=1)
  return float(np.min(d))


def _fps_xyz(points: np.ndarray,
             m: int,
             start_idx: int | None = None) -> np.ndarray:
  N = points.shape[0]
  if m >= N:
    return np.arange(N, dtype=int)
  if start_idx is None:
    centroid = np.mean(points, axis=0, keepdims=True)
    d2 = np.sum((points - centroid)**2, axis=1)
    start_idx = int(np.argmax(d2))

  selected = np.empty(m, dtype=int)
  selected[0] = start_idx

  # Maintain the minimum squared distance from each point to the "selected set"
  min_d2 = np.full(N, np.inf, dtype=float)

  last = points[start_idx]
  diff = points - last
  min_d2 = np.minimum(min_d2, np.sum(diff * diff, axis=1))

  for i in range(1, m):
    idx = int(np.argmax(min_d2))
    selected[i] = idx
    last = points[idx]
    diff = points - last
    min_d2 = np.minimum(min_d2, np.sum(diff * diff, axis=1))

  return selected


def _fps_xyz_torch(points: torch.Tensor,
                   m: int,
                   start_idx: int | None = None) -> torch.Tensor:
  """
  Farthest Point Sampling (FPS) in PyTorch
  """
  N = points.shape[0]
  device = points.device
  if m >= N:
    return torch.arange(N, dtype=torch.long, device=device)
  if start_idx is None:
    centroid = points.mean(dim=0, keepdim=True)
    d2 = ((points - centroid)**2).sum(dim=1)
    start_idx = int(d2.argmax().item())

  selected = torch.empty(m, dtype=torch.long, device=device)
  selected[0] = start_idx

  # Maintain the minimum squared distance from each point to the "selected set"
  min_d2 = torch.full((N,), float('inf'), dtype=torch.float32, device=device)

  last = points[start_idx]
  diff = points - last
  min_d2 = torch.minimum(min_d2, (diff * diff).sum(dim=1))

  for i in range(1, m):
    idx = int(min_d2.argmax().item())
    selected[i] = idx
    last = points[idx]
    diff = points - last
    min_d2 = torch.minimum(min_d2, (diff * diff).sum(dim=1))

  return selected


def adaptive_part_segmentation_torch(
    Pi: np.ndarray,
    Pj: np.ndarray,
    Fi: np.ndarray,
    tau: float = -3.0,  # Compatible with use_sigmoid_dist_norm=True
    seed_dbscan_eps_space:
    float = 0.001,  # Small spatial radius: sticky neighbor seeds
    seed_dbscan_min_samples: int = 3,
    knn_k: int = 24,
    gate_factor:
    float = 2.0,  # Edge length gating: maximum allowed edge length factor (relative to each point's kNN median distance)
    sim_hi: float = 0.75,  # Three-stage threshold
    sim_mid: float = 0.70,
    sim_low: float = 0.70,
    min_neighbors_mid:
    int = 2,  # Mid-stage requires at least m neighbors already in the region
    min_part_size: int = 20,
    max_parts: int = 50,
    merge_sim_thresh:
    float = 0.92,  # (Optional) Merge threshold based on prototype features
    merge_space_thresh:
    float = 0.02,  # (Optional) Merge threshold based on span (meters)
    close_holes: bool = True,  # Soft close switch
    use_sigmoid_dist_norm:
    bool = True,  # Distance logit normalization (compatible with tau=-3)
    overlap_merge_ratio:
    float = 0.80,  # Final overlap merge: if 80% of points in A are within B's neighbors, merge A→B
    overlap_radius_factor: float = 1.5,  # Overlap neighbor radius factor
    # ★ Number of extra seeds sampled per cluster n (final n+1 seeds)
    extra_seeds_per_cluster: int = 10,
    # ★ Extra seed selection strategy: 'fps' (Farthest Point Sampling in feature space) or 'random'
    extra_seed_mode: str = "fps",
    # ★ Random seed mode
    use_random_seeds: bool = False,
    random_seed_count: int = 0,
    verbose: bool = True,
):
  """
  Adaptive part segmentation based on point cloud and features (PyTorch version)
  """
  assert Pi.ndim == 2 and Pi.shape[1] == 3
  N = Pi.shape[0]
  assert Fi.shape[0] == N, "Fi length must match Pi"

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Convert inputs to torch tensors on the appropriate device
  Pi_t = torch.from_numpy(Pi.astype(np.float32, copy=False)).to(device)
  Pj_t = torch.from_numpy(Pj.astype(np.float32, copy=False)).to(device)
  Fi_t = torch.from_numpy(Fi.astype(np.float32, copy=False)).to(device)

  # PartField features: L2 normalization per point
  Fi_norm_t = Fi_t / (torch.norm(Fi_t, dim=1, keepdim=True) + 1e-12)

  # -------------------------
  # 1) Seed selection
  # -------------------------
  # Use knn_points to compute the nearest neighbor distance from Pi to Pj
  d2_fwd, _, _ = knn_points(Pi_t.unsqueeze(0),
                            Pj_t.unsqueeze(0),
                            K=1,
                            return_nn=False)
  d_fwd = torch.sqrt(d2_fwd[0, :, 0].clamp(min=0.0))  # (N,)

  if use_sigmoid_dist_norm:
    d_max = d_fwd.max() + 1e-12
    d = d_fwd / d_max
    d = d.clamp(1e-6, 1 - 1e-6)
    d_logit = torch.log(d / (1 - d))
    seed_mask = d_logit > tau
    seed_indices_t = torch.nonzero(seed_mask, as_tuple=False).view(-1)
    dist_used_t = d_logit
    print(
        f"d_max={d_max.item():.6f} L2 threadhold={d_max.item()*torch.sigmoid(torch.tensor(tau)).item():.6f}"
    )
  else:
    seed_mask = d_fwd > tau
    seed_indices_t = torch.nonzero(seed_mask, as_tuple=False).view(-1)
    dist_used_t = d_fwd

  if verbose:
    print(f"[torch] N={N}, seeds={seed_indices_t.numel()} (tau={tau:.4f})")

  info = {
      "dists": dist_used_t.cpu().numpy(),
      "seed_indices": seed_indices_t.cpu().numpy(),
      "curr_tau": tau,
  }
  if seed_indices_t.numel() == 0:
    return np.zeros(N, np.int32), [], info

  # -------------------------
  # 2) Generate cluster_seeds (DBSCAN / Random)
  # -------------------------
  # DBSCAN requires sklearn on CPU, this is the only place we need to temporarily convert to numpy
  seed_indices_np = seed_indices_t.cpu().numpy()

  if not use_random_seeds:
    db = DBSCAN(
        eps=seed_dbscan_eps_space,
        min_samples=seed_dbscan_min_samples,
        metric="euclidean",
    )
    seed_xyz_np = Pi_t[seed_indices_t].cpu().numpy()
    seed_cluster_labels = db.fit_predict(seed_xyz_np)
    info["seed_dbscan_labels"] = seed_cluster_labels
    clusters = [c for c in np.unique(seed_cluster_labels) if c >= 0]

    cluster_seeds_list = []  # Store torch tensors
    if len(clusters) == 0:
      for si in seed_indices_np:
        cluster_seeds_list.append(
            torch.tensor([si], dtype=torch.long, device=device))
    else:
      for c in clusters:
        members = seed_indices_np[seed_cluster_labels == c]
        cluster_seeds_list.append(
            torch.from_numpy(members.astype(np.int64)).to(device))

    if len(cluster_seeds_list) > max_parts:
      cluster_seeds_list.sort(key=lambda x: -x.numel())
      cluster_seeds_list = cluster_seeds_list[:max_parts]
  else:
    if random_seed_count <= 0 or random_seed_count > seed_indices_t.numel():
      random_seed_count = seed_indices_t.numel()
    xyz = Pi_t[seed_indices_t]  # (S,3)
    sel_local = _fps_xyz_torch(xyz, random_seed_count)
    selected_seeds = seed_indices_t[sel_local]

    cluster_seeds_list = [
        selected_seeds[i:i + 1] for i in range(selected_seeds.numel())
    ]
    info["seed_dbscan_labels"] = None

  # Save cluster_seeds to info (convert to numpy for compatibility)
  info["cluster_seeds"] = [cs.cpu().numpy() for cs in cluster_seeds_list]
  if verbose:
    print(f"[torch] initial seed groups: {len(cluster_seeds_list)}")

  # -------------------------
  # 3) Build graph: kNN + edge gating + mutual neighbors (pure PyTorch)
  # -------------------------
  print(f"[torch] building graph with kNN + edge gating + mutual neighbors...")
  d_knn, nn_idx, per_node_med = knn_median_scale_p3d(Pi,
                                                     knn_k,
                                                     use_cuda=True,
                                                     chunk_size=200_000,
                                                     return_numpy=False)
  d_knn = d_knn.to(device)
  nn_idx = nn_idx.to(device).long()
  per_node_med = per_node_med.to(device)

  # Vectorized construction of edge mask
  K = nn_idx.shape[1]
  gate_thresh = gate_factor * (per_node_med + 1e-12)  # (N,)

  # allow_mask: edges within the gating threshold
  allow_mask = d_knn <= gate_thresh.unsqueeze(1)  # (N, k)

  # mutual_mask: i is a neighbor of j and j is a neighbor of i
  i_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1,
                                                                 K)  # (N, K)
  mutual_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
  print(f" - computing mutual neighbor mask...")
  for pos in range(K):
    j_indices = nn_idx[:,
                       pos]  # (N,) - neighbor j of each point i at position pos
    j_neighbors = nn_idx[j_indices]  # (N, K)
    is_mutual = (j_neighbors == i_indices[:, :1]).any(dim=1)  # (N,)
    mutual_mask[:, pos] = is_mutual

  # Exclude self
  self_mask = nn_idx == torch.arange(N, device=device).unsqueeze(1)
  mutual_mask = mutual_mask & ~self_mask

  edge_mask = mutual_mask & allow_mask

  # Precompute neighbor index tensor (for batch operations)
  # Count how many valid neighbors each point has
  neigh_counts = edge_mask.sum(dim=1)  # (N,)
  max_neighbors = int(neigh_counts.max().item()) if N > 0 else 0
  max_neighbors = max(max_neighbors,
                      1)  # At least 1 to avoid empty tensor issues

  print(f" - preparing neighbor tensor with max_neighbors={max_neighbors}...")
  # Vectorized construction of neigh_tensor (avoid per-point loop)
  neigh_tensor = torch.full((N, max_neighbors),
                            -1,
                            dtype=torch.long,
                            device=device)
  # Get all valid edge (row, col) indices
  rows, cols = torch.where(
      edge_mask)  # rows: point indices, cols: positions in nn_idx
  if rows.numel() > 0:
    # Compute the position of each valid neighbor within its row (using cumsum)
    # First, compute the starting offset for each row
    col_offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
    col_offsets[1:] = neigh_counts.cumsum(0)
    # Per-row position of valid neighbors: pos_in_row = global_index - row_start_offset
    pos_in_row = torch.arange(rows.numel(), device=device) - col_offsets[rows]
    neigh_tensor[rows, pos_in_row] = nn_idx[rows, cols]

  # -------------------------
  # 4) Three-stage adaptive diffusion + region growing
  # -------------------------
  labels = torch.zeros(N, dtype=torch.int32, device=device)
  best_score = torch.full((N,),
                          -float('inf'),
                          dtype=torch.float32,
                          device=device)
  parts = []  # Store torch tensors
  part_id = 0

  def region_grow_torch(seed_ids_t: torch.Tensor, sim_threshold: float,
                        phase: str, in_part: torch.Tensor) -> tuple:
    """
    PyTorch parallelized region growing
    Returns (newly added point indices tensor, updated in_part)
    
    Logic is consistent with the original numpy version:
    - Seed points: added to in_part and frontier regardless of existing labels
    - Only update labels if labels[s] == 0
    - During BFS diffusion, check accept conditions and best_score for each neighbor
    """
    nonlocal part_id

    # Remove duplicate seeds
    seed_ids_t = torch.unique(seed_ids_t)

    # Compute prototype
    proto = Fi_norm_t[seed_ids_t].mean(dim=0)
    proto = proto / (torch.norm(proto) + 1e-12)

    # Precompute similarity of all points with the prototype
    all_scores = Fi_norm_t @ proto  # (N,)

    newly = []

    # Process seeds: add to in_part regardless of existing labels; update labels only if labels==0
    in_part[seed_ids_t] = True
    for s in seed_ids_t:
      s_int = s.item()
      if labels[s_int] == 0:
        score = all_scores[s_int]
        if score > best_score[s_int]:
          best_score[s_int] = score
          labels[s_int] = part_id
          newly.append(s_int)

    # BFS parallel diffusion
    # Note: frontier initially contains all seeds (regardless of existing labels), consistent with the original version
    frontier = seed_ids_t.clone()
    max_iters = N  # Prevent infinite loop

    for _ in range(max_iters):
      if frontier.numel() == 0:
        break

      # Collect neighbors of all frontier points
      frontier_neighbors = neigh_tensor[frontier]  # (F, max_neighbors)
      frontier_neighbor_counts = neigh_counts[frontier]  # (F,)

      # Flatten and deduplicate
      valid_mask = torch.arange(
          max_neighbors,
          device=device).unsqueeze(0) < frontier_neighbor_counts.unsqueeze(1)
      all_neighbors = frontier_neighbors[valid_mask]  # (total_valid,)

      if all_neighbors.numel() == 0:
        break

      unique_neighbors = torch.unique(all_neighbors)
      # Exclude invalid indices
      unique_neighbors = unique_neighbors[unique_neighbors >= 0]
      # Exclude points already in in_part (consistent with original `if not in_part[nb]`)
      unique_neighbors = unique_neighbors[~in_part[unique_neighbors]]

      if unique_neighbors.numel() == 0:
        break

      # Batch compute scores
      scores = all_scores[unique_neighbors]

      # Compute neighborhood agreement based on phase
      if phase in ("mid", "soft"):
        # Batch compute how many neighbors of each candidate are in in_part
        cand_neighbors = neigh_tensor[unique_neighbors]  # (C, max_neighbors)
        cand_counts = neigh_counts[unique_neighbors]  # (C,)

        # Vectorized count of neighbors in in_part
        valid_cand_mask = torch.arange(
            max_neighbors, device=device).unsqueeze(0) < cand_counts.unsqueeze(
                1)  # (C, max_neighbors)
        # Set invalid positions to 0 (a valid index)
        safe_neighbors = cand_neighbors.clone()
        safe_neighbors[~valid_cand_mask] = 0
        # Query in_part status
        neighbor_in_part = in_part[safe_neighbors]  # (C, max_neighbors)
        # Only count valid positions
        neighbor_in_part = neighbor_in_part & valid_cand_mask
        agree_counts = neighbor_in_part.sum(dim=1)  # (C,)
      else:
        agree_counts = torch.zeros(unique_neighbors.numel(),
                                   dtype=torch.long,
                                   device=device)

      # Determine acceptance criteria based on phase
      if phase == "hard":
        accept = scores >= sim_threshold
      elif phase == "mid":
        accept = (scores >= sim_threshold) & (agree_counts >= min_neighbors_mid)
      else:  # 'soft'
        accept = (scores >= sim_threshold) & (agree_counts >= max(
            2 * min_neighbors_mid, 3))

      # Further check if it improves best_score
      better = scores > best_score[unique_neighbors]
      final_accept = accept & better

      accepted_indices = unique_neighbors[final_accept]

      if accepted_indices.numel() == 0:
        break

      # Update state
      best_score[accepted_indices] = scores[final_accept]
      labels[accepted_indices] = part_id
      in_part[accepted_indices] = True

      newly.append(accepted_indices)
      frontier = accepted_indices

    # Merge all newly added points
    if len(newly) > 0:
      # newly may contain a mix of integers and tensors
      int_items = [x for x in newly if isinstance(x, int)]
      tensor_items = [x for x in newly if isinstance(x, torch.Tensor)]
      all_newly = []
      if int_items:
        all_newly.append(
            torch.tensor(int_items, dtype=torch.long, device=device))
      all_newly.extend(tensor_items)
      if all_newly:
        return torch.cat(all_newly) if len(
            all_newly) > 1 else all_newly[0], in_part
    return torch.tensor([], dtype=torch.long, device=device), in_part

  # Extra sampling function
  def select_seeds_from_cluster_torch(seed_members_t: torch.Tensor,
                                      n_extra: int,
                                      mode: str = "fps") -> torch.Tensor:
    if seed_members_t.numel() == 0:
      return seed_members_t
    if seed_members_t.numel() == 1 or n_extra <= 0:
      return seed_members_t[:1]

    feats = Fi_norm_t[seed_members_t]
    proto = feats.mean(dim=0)
    proto = proto / (torch.norm(proto) + 1e-12)
    sims = feats @ proto
    core_local = sims.argmax().item()
    core_idx = seed_members_t[core_local].item()

    cand_mask = seed_members_t != core_idx
    cand_t = seed_members_t[cand_mask]
    if cand_t.numel() <= n_extra:
      return torch.cat(
          [torch.tensor([core_idx], dtype=torch.long, device=device), cand_t])

    if mode == "random":
      perm = torch.randperm(cand_t.numel(), device=device)[:n_extra]
      sel_extra = cand_t[perm]
      return torch.cat([
          torch.tensor([core_idx], dtype=torch.long, device=device), sel_extra
      ])

    # FPS in feature space
    selected = [core_idx]  # Store Python int
    cand_feats = Fi_norm_t[cand_t]  # (C, F)
    selected_feats = Fi_norm_t[core_idx].unsqueeze(0)  # (1, F)

    while len(selected) < n_extra + 1 and cand_t.numel() > 0:
      # Compute the maximum similarity of each candidate to the selected points
      sims_mat = cand_feats @ selected_feats.T  # (C, S)
      max_sim, _ = sims_mat.max(dim=1)  # (C,)
      # Select the point with the minimum similarity (farthest in feature space)
      j = max_sim.argmin().item()
      new_idx = cand_t[j].item()
      selected.append(new_idx)
      selected_feats = torch.cat(
          [selected_feats, Fi_norm_t[new_idx].unsqueeze(0)], dim=0)
      # Remove the selected point
      mask = torch.ones(cand_t.numel(), dtype=torch.bool, device=device)
      mask[j] = False
      cand_t = cand_t[mask]
      cand_feats = cand_feats[mask]

    return torch.tensor(selected, dtype=torch.long, device=device)

  # Main growth loop
  if not use_random_seeds:
    for seed_members_t in tqdm(cluster_seeds_list,
                               desc="Growing parts (torch)"):
      if torch.all(labels[seed_members_t] > 0):
        continue

      initial_seeds = select_seeds_from_cluster_torch(
          seed_members_t, n_extra=extra_seeds_per_cluster, mode=extra_seed_mode)

      part_id += 1
      in_part = torch.zeros(N, dtype=torch.bool, device=device)

      # Phase 1: hard
      new1, in_part = region_grow_torch(initial_seeds, sim_hi, "hard", in_part)
      if new1.numel() == 0:
        part_id -= 1
        continue

      # Phase 2: mid
      new2, in_part = region_grow_torch(new1, sim_mid, "mid", in_part)

      # Phase 3: soft
      seeds_for_soft = new2 if new2.numel() > 0 else new1
      _, in_part = region_grow_torch(seeds_for_soft, sim_low, "soft", in_part)

      idxs = torch.where(labels == part_id)[0]
      if idxs.numel() < min_part_size:
        labels[idxs] = 0
        best_score[idxs] = -float('inf')
        part_id -= 1
        continue

      parts.append(idxs)
      if part_id >= max_parts:
        if verbose:
          print("[torch] reached max_parts limit")
        break
  else:
    for seed_members_t in tqdm(cluster_seeds_list,
                               desc="Growing parts (torch random)"):
      if torch.all(labels[seed_members_t] > 0):
        continue

      part_id += 1
      in_part = torch.zeros(N, dtype=torch.bool, device=device)

      new1, in_part = region_grow_torch(seed_members_t, sim_hi, "hard", in_part)
      if new1.numel() == 0:
        part_id -= 1
        continue

      new2, in_part = region_grow_torch(new1, sim_mid, "mid", in_part)
      seeds_for_soft = new2 if new2.numel() > 0 else new1
      _, in_part = region_grow_torch(seeds_for_soft, sim_low, "soft", in_part)

      idxs = torch.where(labels == part_id)[0]
      if idxs.numel() < min_part_size:
        labels[idxs] = 0
        best_score[idxs] = -float('inf')
        part_id -= 1
        continue

      parts.append(idxs)
      if part_id >= max_parts:
        if verbose:
          print("[torch] reached max_parts limit")
        break

  if verbose:
    print(f"[torch] after growth, parts={len(parts)}")

  # -------------------------
  # 5) Post-merging (similar prototypes + spatial proximity)
  # -------------------------
  def compute_proto_torch(idx_t: torch.Tensor) -> torch.Tensor:
    return Fi_norm_t[idx_t].mean(dim=0)

  print(f"[torch] post-merging parts...")
  merged = True
  while merged:
    merged = False
    K_parts = len(parts)
    if K_parts <= 1:
      break

    # Batch compute all prototypes
    protos_list = [compute_proto_torch(p) for p in parts]
    protos_t = torch.stack(protos_list, dim=0)  # (K, F)
    protos_t = protos_t / (torch.norm(protos_t, dim=1, keepdim=True) + 1e-12)

    # Compute similarity matrix
    sim_matrix = protos_t @ protos_t.T  # (K, K)

    to_delete = set()
    for a in range(K_parts):
      if a in to_delete:
        continue
      for b in range(a + 1, K_parts):
        if b in to_delete:
          continue
        if sim_matrix[a, b].item() < merge_sim_thresh:
          continue
        # Compute minimum distance
        Pa = Pi_t[parts[a]]
        Pb = Pi_t[parts[b]]
        # Use knn_points to compute minimum distance
        d2, _, _ = knn_points(Pa.unsqueeze(0),
                              Pb.unsqueeze(0),
                              K=1,
                              return_nn=False)
        dmin = torch.sqrt(d2.min()).item()
        if dmin <= merge_space_thresh:
          parts[a] = torch.unique(torch.cat([parts[a], parts[b]]))
          to_delete.add(b)
          merged = True

    if merged and to_delete:
      new_parts = []
      new_labels = torch.zeros_like(labels)
      pid = 0
      for i, p in enumerate(parts):
        if i in to_delete:
          continue
        pid += 1
        new_parts.append(p)
        new_labels[p] = pid
      labels = new_labels
      parts = new_parts
      part_id = len(parts)

  # -------------------------
  # 6) Soft hole closing (fill small holes)
  # -------------------------
  print(f"[torch] closing holes...")
  # Vectorized soft hole closing (fill small holes)
  if close_holes and len(parts) > 0:
    new_labels = labels.clone()
    for pid, idxs_t in enumerate(parts, start=1):
      # Vectorized collection of boundary points: get neighbors of all points in the part
      part_neighbors = neigh_tensor[idxs_t]  # (P, max_neighbors)
      part_counts = neigh_counts[idxs_t]  # (P,)

      # Create valid neighbor mask
      valid_neighbor_mask = torch.arange(
          max_neighbors, device=device).unsqueeze(0) < part_counts.unsqueeze(1)

      # Get all valid neighbors (flattened)
      all_neighbors = part_neighbors[valid_neighbor_mask]  # (total_valid,)
      all_neighbors = all_neighbors[all_neighbors
                                    >= 0]  # Exclude invalid indices

      if all_neighbors.numel() == 0:
        continue

      # Filter neighbors that do not belong to the current part (i.e., boundary points)
      neighbor_labels_flat = labels[all_neighbors]
      boundary_mask = neighbor_labels_flat != pid
      boundary_points = all_neighbors[boundary_mask]
      boundary_t = torch.unique(boundary_points)

      if boundary_t.numel() == 0:
        continue

      # batch voting
      boundary_neighbors = neigh_tensor[boundary_t]  # (B, max_neighbors)
      boundary_counts = neigh_counts[boundary_t]  # (B,)
      valid_mask = torch.arange(
          max_neighbors,
          device=device).unsqueeze(0) < boundary_counts.unsqueeze(1)
      safe_neighbors = boundary_neighbors.clone()
      safe_neighbors[~valid_mask] = 0
      neighbor_labels = labels[safe_neighbors]  # (B, max_neighbors)
      votes = ((neighbor_labels == pid) & valid_mask).sum(dim=1)  # (B,)

      take_mask = votes >= max(3, min_neighbors_mid + 1)
      take = boundary_t[take_mask]
      new_labels[take] = pid
      if take.numel() > 0:
        parts[pid - 1] = torch.unique(torch.cat([parts[pid - 1], take]))
    labels = new_labels

  # Filter small parts
  print(f"[torch] filtering small parts...")
  final_parts = []
  new_labels = torch.zeros_like(labels)
  pid = 0
  for p in parts:
    if p.numel() >= min_part_size:
      pid += 1
      final_parts.append(p)
      new_labels[p] = pid
  labels = new_labels

  # -------------------------
  # 7) Final overlap-merge - if 80% of points in A overlap with B, merge A->B
  # -------------------------
  print(f"[torch] final overlap-merge...")
  if len(final_parts) > 1:

    def overlap_fraction_torch(A_idx_t: torch.Tensor,
                               B_idx_t: torch.Tensor) -> float:
      if A_idx_t.numel() == 0 or B_idx_t.numel() == 0:
        return 0.0

      RA = per_node_med[A_idx_t].median().item() * overlap_radius_factor
      # Use knn_points to compute nearest neighbor distance from A to B
      Pa = Pi_t[A_idx_t].unsqueeze(0)  # (1, |A|, 3)
      Pb = Pi_t[B_idx_t].unsqueeze(0)  # (1, |B|, 3)
      d2, _, _ = knn_points(Pa, Pb, K=1, return_nn=False)
      d = torch.sqrt(d2[0, :, 0].clamp(min=0.0))  # (|A|,)
      return float((d <= RA).float().mean().item())

    changed = True
    while changed:
      changed = False
      K_parts = len(final_parts)
      if K_parts <= 1:
        break

      to_delete = set()
      for a in tqdm(range(K_parts), desc="Overlap-merge (torch)"):
        if a in to_delete:
          continue
        for b in range(K_parts):
          if a == b or b in to_delete:
            continue
          frac = overlap_fraction_torch(final_parts[a], final_parts[b])
          if frac >= overlap_merge_ratio:
            final_parts[b] = torch.unique(
                torch.cat([final_parts[b], final_parts[a]]))
            to_delete.add(a)
            changed = True
            break

      if changed and to_delete:
        new_parts = []
        new_labels = torch.zeros_like(labels)
        pid = 0
        for i, p in enumerate(final_parts):
          if i in to_delete:
            continue
          pid += 1
          new_parts.append(p)
          new_labels[p] = pid
        final_parts = new_parts
        labels = new_labels

  if verbose:
    print(f"[torch] finished, detected parts: {len(final_parts)}")

  labels_np = labels.cpu().numpy()
  best_score_np = best_score.cpu().numpy()
  final_parts_np = [p.cpu().numpy() for p in final_parts]

  info["parts_idx"] = final_parts_np
  info["best_score"] = best_score_np
  return labels_np, final_parts_np, info


import matplotlib.pyplot as plt


def visualize_parts(path_path, Pi, labels):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(Pi)
  max_label = int(labels.max())
  colors = np.zeros((Pi.shape[0], 3))
  import matplotlib.pyplot as plt

  cmap = plt.get_cmap("tab20")
  for i in range(1, max_label + 1):
    colors[labels == i] = cmap((i - 1) % 20)[:3]
  colors[labels == 0] = np.array([0.6, 0.6, 0.6])  # background gray
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.io.write_point_cloud(path_path, pcd)


def visualize_pcd_parts(path_path, pcd_parts, Pj=None):
  pcd = o3d.geometry.PointCloud()
  all_points = np.concatenate(pcd_parts, axis=0)
  colors = np.zeros((all_points.shape[0], 3))
  import matplotlib.pyplot as plt

  cmap = plt.get_cmap("tab20")
  start = 0
  for i, part in enumerate(pcd_parts):
    end = start + part.shape[0]
    colors[start:end] = cmap(i % 20)[:3]
    start = end
  if Pj is not None:
    # Mark Pj points as black
    all_points = np.concatenate([all_points, Pj], axis=0)
    colors = np.concatenate([colors, 0.3 * np.ones((Pj.shape[0], 3))], axis=0)
  pcd.points = o3d.utility.Vector3dVector(all_points)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.io.write_point_cloud(path_path, pcd)


def visualize_cluster_seeds(path_path, Pi, cluster_seeds):
  all_seed_indices = np.concatenate(cluster_seeds)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(Pi[all_seed_indices])
  max_label = len(cluster_seeds)
  colors = np.zeros((all_seed_indices.shape[0], 3))
  import matplotlib.pyplot as plt

  cmap = plt.get_cmap("tab20")
  for i in range(max_label):
    colors[np.isin(all_seed_indices, cluster_seeds[i])] = cmap(i % 20)[:3]
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.io.write_point_cloud(path_path, pcd)


def visualize_cd_parts(path_path, Pi, indices):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(Pi)
  colors = np.zeros((Pi.shape[0], 3))
  colors[indices] = np.array([1.0, 0.0, 0.0])  # red for changed
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.io.write_point_cloud(path_path, pcd)


def visualize_dists(path_path, dists, tau):
  plt.figure(figsize=(6, 4))
  plt.hist(dists, bins=50, color="blue", alpha=0.7)
  plt.axvline(x=tau,
              color="red",
              linestyle="--",
              label=f"Threshold (tau={tau})")
  plt.title("Histogram of Distances")
  plt.xlabel("Distance")
  plt.ylabel("Frequency")
  plt.grid(True)
  plt.savefig(path_path)
