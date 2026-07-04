#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct

import torch

from utils.graphics_utils import getWorld2View, getProjectionMatrix

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CameraModel = collections.namedtuple("CameraModel",
                                     ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera",
                                ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([
    (camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS
])
CAMERA_MODEL_NAMES = dict([
    (camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS
])


def qvec2rotmat(qvec):
  return np.array([[
      1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
      2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
      2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
  ],
                   [
                       2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                       1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                       2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
                   ],
                   [
                       2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                       2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                       1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
                   ]])


def rotmat2qvec(R):
  Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
  K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
  eigvals, eigvecs = np.linalg.eigh(K)
  qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
  if qvec[0] < 0:
    qvec *= -1
  return qvec


class Image(BaseImage):

  def qvec2rotmat(self):
    return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
  """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
  data = fid.read(num_bytes)
  return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
  """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
  xyzs = None
  rgbs = None
  errors = None
  num_points = 0
  with open(path, "r") as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      line = line.strip()
      if len(line) > 0 and line[0] != "#":
        num_points += 1

  xyzs = np.empty((num_points, 3))
  rgbs = np.empty((num_points, 3))
  errors = np.empty((num_points, 1))
  count = 0
  with open(path, "r") as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      line = line.strip()
      if len(line) > 0 and line[0] != "#":
        elems = line.split()
        xyz = np.array(tuple(map(float, elems[1:4])))
        rgb = np.array(tuple(map(int, elems[4:7])))
        error = np.array(float(elems[7]))
        xyzs[count] = xyz
        rgbs[count] = rgb
        errors[count] = error
        count += 1

  return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
  """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

  with open(path_to_model_file, "rb") as fid:
    num_points = read_next_bytes(fid, 8, "Q")[0]

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))

    for p_id in range(num_points):
      binary_point_line_properties = read_next_bytes(
          fid, num_bytes=43, format_char_sequence="QdddBBBd")
      xyz = np.array(binary_point_line_properties[1:4])
      rgb = np.array(binary_point_line_properties[4:7])
      error = np.array(binary_point_line_properties[7])
      track_length = read_next_bytes(fid, num_bytes=8,
                                     format_char_sequence="Q")[0]
      track_elems = read_next_bytes(fid,
                                    num_bytes=8 * track_length,
                                    format_char_sequence="ii" * track_length)
      xyzs[p_id] = xyz
      rgbs[p_id] = rgb
      errors[p_id] = error
  return xyzs, rgbs, errors


def read_intrinsics_text(path):
  """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
  cameras = {}
  with open(path, "r") as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      line = line.strip()
      if len(line) > 0 and line[0] != "#":
        elems = line.split()
        camera_id = int(elems[0])
        model = elems[1]
        assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
        width = int(elems[2])
        height = int(elems[3])
        params = np.array(tuple(map(float, elems[4:])))
        cameras[camera_id] = Camera(id=camera_id,
                                    model=model,
                                    width=width,
                                    height=height,
                                    params=params)
  return cameras


def read_extrinsics_binary(path_to_model_file):
  """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
  images = {}
  with open(path_to_model_file, "rb") as fid:
    num_reg_images = read_next_bytes(fid, 8, "Q")[0]
    for _ in range(num_reg_images):
      binary_image_properties = read_next_bytes(
          fid, num_bytes=64, format_char_sequence="idddddddi")
      image_id = binary_image_properties[0]
      qvec = np.array(binary_image_properties[1:5])
      tvec = np.array(binary_image_properties[5:8])
      camera_id = binary_image_properties[8]
      image_name = ""
      current_char = read_next_bytes(fid, 1, "c")[0]
      while current_char != b"\x00":  # look for the ASCII 0 entry
        image_name += current_char.decode("utf-8")
        current_char = read_next_bytes(fid, 1, "c")[0]
      num_points2D = read_next_bytes(fid, num_bytes=8,
                                     format_char_sequence="Q")[0]
      x_y_id_s = read_next_bytes(fid,
                                 num_bytes=24 * num_points2D,
                                 format_char_sequence="ddq" * num_points2D)
      xys = np.column_stack([
          tuple(map(float, x_y_id_s[0::3])),
          tuple(map(float, x_y_id_s[1::3]))
      ])
      point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
      images[image_id] = Image(id=image_id,
                               qvec=qvec,
                               tvec=tvec,
                               camera_id=camera_id,
                               name=image_name,
                               xys=xys,
                               point3D_ids=point3D_ids)
  return images


def read_intrinsics_binary(path_to_model_file):
  """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
  cameras = {}
  with open(path_to_model_file, "rb") as fid:
    num_cameras = read_next_bytes(fid, 8, "Q")[0]
    for _ in range(num_cameras):
      camera_properties = read_next_bytes(fid,
                                          num_bytes=24,
                                          format_char_sequence="iiQQ")
      camera_id = camera_properties[0]
      model_id = camera_properties[1]
      model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
      width = camera_properties[2]
      height = camera_properties[3]
      num_params = CAMERA_MODEL_IDS[model_id].num_params
      params = read_next_bytes(fid,
                               num_bytes=8 * num_params,
                               format_char_sequence="d" * num_params)
      cameras[camera_id] = Camera(id=camera_id,
                                  model=model_name,
                                  width=width,
                                  height=height,
                                  params=np.array(params))
    assert len(cameras) == num_cameras
  return cameras


def read_extrinsics_text(path):
  """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
  images = {}
  with open(path, "r") as fid:
    while True:
      line = fid.readline()
      if not line:
        break
      line = line.strip()
      if len(line) > 0 and line[0] != "#":
        elems = line.split()
        image_id = int(elems[0])
        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        camera_id = int(elems[8])
        image_name = elems[9]
        elems = fid.readline().split()
        xys = np.column_stack(
            [tuple(map(float, elems[0::3])),
             tuple(map(float, elems[1::3]))])
        point3D_ids = np.array(tuple(map(int, elems[2::3])))
        images[image_id] = Image(id=image_id,
                                 qvec=qvec,
                                 tvec=tvec,
                                 camera_id=camera_id,
                                 name=image_name,
                                 xys=xys,
                                 point3D_ids=point3D_ids)
  return images


def read_colmap_bin_array(path):
  """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
  with open(path, "rb") as fid:
    width, height, channels = np.genfromtxt(fid,
                                            delimiter="&",
                                            max_rows=1,
                                            usecols=(0, 1, 2),
                                            dtype=int)
    fid.seek(0)
    num_delimiter = 0
    byte = fid.read(1)
    while True:
      if byte == b"&":
        num_delimiter += 1
        if num_delimiter >= 3:
          break
      byte = fid.read(1)
    array = np.fromfile(fid, np.float32)
  array = array.reshape((width, height, channels), order="F")
  return np.transpose(array, (1, 0, 2)).squeeze()


def extract_world_pts(depth_map: np.ndarray,
                      R: np.array,
                      T: np.array,
                      FovY: np.array,
                      FovX: np.array,
                      image_width: int,
                      image_height: int,
                      color_map: np.ndarray = None):
  """Returns (world_coords, colors, normals) as torch tensors on _DEVICE."""
  zfar = 100.0
  znear = 0.01

  # Build 4×4 matrices on _DEVICE
  proj = getProjectionMatrix(znear, zfar, FovX, FovY, blender_convention=True)
  inverse_projection = torch.inverse(proj).to(_DEVICE)

  w2c_colmap = getWorld2View(R, T)  # numpy (4,4)
  c2w_colmap = np.linalg.inv(w2c_colmap)
  c2w_blender = torch.as_tensor(c2w_colmap, dtype=torch.float32, device=_DEVICE)
  c2w_blender[:3, 1:3] *= -1
  inverse_view = c2w_blender

  # Depth map → flat tensor
  depth = torch.as_tensor(depth_map, dtype=torch.float32, device=_DEVICE)
  depth_flat = depth.reshape(-1)

  # Coordinate grids, matching np.meshgrid(arange(W), arange(H)) row-major order
  y_coords, x_coords = torch.meshgrid(
      torch.arange(image_height, dtype=torch.float32, device=_DEVICE),
      torch.arange(image_width, dtype=torch.float32, device=_DEVICE),
      indexing='ij')
  x_flat = x_coords.reshape(-1)
  y_flat = y_coords.reshape(-1)

  # Compute world coords for ALL pixels (needed for finite-diff normals)
  all_valid = depth_flat > 1e-6
  depth_for_proj = -depth_flat.clone()
  depth_for_proj[~all_valid] = -1.0  # dummy for invalid pixels
  world_coords_all = xyd_to_world_coords(x_flat, y_flat, depth_for_proj,
                                         image_width, image_height,
                                         inverse_projection, inverse_view)

  # Reshape to (H, W, 3) for normal computation via finite differences
  world_pts_map = world_coords_all.reshape(image_height, image_width, 3)
  valid_mask_2d = all_valid.reshape(image_height, image_width)
  world_pts_map[~valid_mask_2d] = float('nan')

  # Compute normals: cross product of forward differences along x and y
  nan3 = float('nan')
  dx = torch.full_like(world_pts_map, nan3)
  dy = torch.full_like(world_pts_map, nan3)
  dx[:, :-1, :] = world_pts_map[:, 1:, :] - world_pts_map[:, :-1, :]
  dy[:-1, :, :] = world_pts_map[1:, :, :] - world_pts_map[:-1, :, :]
  # (right,down) basis is left-handed vs -Z camera; swap operands to face camera
  cross = torch.linalg.cross(dy, dx, dim=-1)
  normals_map = cross / cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)

  normals_flat = normals_map.reshape(-1, 3)

  # Keep only pixels beyond the median depth (same filter as before)
  thresh = torch.quantile(depth_flat[all_valid], 0.5)
  valid_indices = depth_flat > thresh

  world_coords = world_coords_all[valid_indices]
  normals_valid = torch.nan_to_num(normals_flat[valid_indices], nan=0.0)

  if color_map is not None:
    colors = torch.as_tensor(
        np.array(color_map, dtype=np.float32).reshape(-1, 3), device=_DEVICE)
    colors = colors[valid_indices]
  else:
    colors = torch.zeros((0, 3), dtype=torch.float32, device=_DEVICE)

  return world_coords, colors, normals_valid


def xyd_to_world_coords(x_valid, y_valid, depth_valid, image_width,
                        image_height, inverse_projection, inverse_view):
  """All inputs are torch tensors on _DEVICE; returns (N,3) tensor on _DEVICE."""
  # Normalized device coordinates (NDC)
  ndc_x = (2.0 * x_valid / image_width) - 1.0
  ndc_y = 1.0 - (2.0 * y_valid / image_height)
  ndc_z = torch.full_like(ndc_x, -1.0)
  ndc_homogeneous = torch.stack([ndc_x, ndc_y, ndc_z, torch.ones_like(ndc_x)], dim=1)

  # Camera space coordinates
  camera_homogeneous = ndc_homogeneous @ inverse_projection.T
  camera_homogeneous = camera_homogeneous / camera_homogeneous[:, 3:4]  # perspective divide
  # Scale xyz by the actual depth value
  camera_homogeneous[:, :3] *= (depth_valid / (camera_homogeneous[:, 2] + 1e-8)).unsqueeze(1)

  # World space coordinates
  world_coords = (camera_homogeneous @ inverse_view.T)[:, :3]
  return world_coords


def get_pcd_from_depths(
    cam_infos: list,
    num_pts: int = 100_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  world_coords_list = []
  colors_list = []
  normals_list = []
  for cam_info in cam_infos:
    pts, rgb, nrm = extract_world_pts(
        depth_map=cam_info.image_d,
        R=cam_info.R,
        T=cam_info.T,
        FovX=cam_info.FovX,
        FovY=cam_info.FovY,
        image_width=cam_info.width,
        image_height=cam_info.height,
        color_map=cam_info.image,
    )
    world_coords_list.append(pts)
    colors_list.append(rgb)
    normals_list.append(nrm)
    print(f"done with train camera {cam_info.uid}: got {pts.shape[0]} pts")
  print('done.\n')

  world_coords = torch.cat(world_coords_list, dim=0)
  colors = torch.cat(colors_list, dim=0)
  normals = torch.cat(normals_list, dim=0)
  N = world_coords.shape[0]

  if num_pts >= N:
    return world_coords.cpu().numpy(), colors.cpu().numpy(), normals.cpu().numpy()

  idx = torch.randperm(N, device=_DEVICE)[:num_pts]
  colors = colors[idx] if colors.shape[0] == N else colors
  normals = normals[idx]
  world_coords = world_coords[idx]
  return world_coords.cpu().numpy(), colors.cpu().numpy(), normals.cpu().numpy()


def proj_corr_to_world(cam_info, corr: np.ndarray) -> np.ndarray:
  zfar = 100.0
  znear = 0.01

  proj = getProjectionMatrix(znear, zfar, cam_info.FovX, cam_info.FovY,
                             blender_convention=True)
  inverse_projection = torch.inverse(proj).to(_DEVICE)

  w2c_colmap = getWorld2View(cam_info.R, cam_info.T)
  c2w_colmap = np.linalg.inv(w2c_colmap)
  c2w_blender = torch.as_tensor(c2w_colmap, dtype=torch.float32, device=_DEVICE)
  c2w_blender[:3, 1:3] *= -1
  inverse_view = c2w_blender

  depth_map = cam_info.image_d
  x, y = corr[:, 0], corr[:, 1]
  d = -depth_map[y, x]
  x_t = torch.as_tensor(x, dtype=torch.float32, device=_DEVICE)
  y_t = torch.as_tensor(y, dtype=torch.float32, device=_DEVICE)
  d_t = torch.as_tensor(d, dtype=torch.float32, device=_DEVICE)
  world_coords = xyd_to_world_coords(x_t, y_t, d_t, cam_info.width, cam_info.height,
                                     inverse_projection, inverse_view)
  return world_coords.cpu().numpy()
