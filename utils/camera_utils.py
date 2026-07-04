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

import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

import json

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image_d = None
    if cam_info.image_d is not None:
        assert 0.999 < resolution_scale < 1.001
        gt_image_d = torch.from_numpy(cam_info.image_d)
        if len(gt_image_d.shape) == 3:
            gt_image_d = gt_image_d[:, :, :1].permute(2, 0, 1)
        else:
            gt_image_d = gt_image_d.unsqueeze(dim=-1).permute(2, 0, 1)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, image_d=gt_image_d, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def create_transforms_on_sphere(
    output: str, radius: float, angle_x: float, num_points: int=100, num_points_down: int=37
):
    def get_hammersley(i: int, num: int) -> tuple[float, float]:
        bits = ((i << 16) | (i >> 16)) & 0xFFFFFFFF
        bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
        bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
        bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
        bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
        rdi = bits * 2.3283064365386963e-10
        return float(i) / float(num), rdi

    def get_random_points_on_hemisphere(num_pts: int) -> np.ndarray:
        num_pts += 1
        points = np.zeros((num_pts, 3))
        for i in np.arange(num_pts):
            hx, hy = get_hammersley(i, num_pts)
            phi = 2 * np.pi * hx
            cos_theta = 1 - hy
            points[i, 0] = np.cos(phi) * np.sqrt(1 - cos_theta * cos_theta)
            points[i, 1] = np.sin(phi) * np.sqrt(1 - cos_theta * cos_theta)
            points[i, 2] = cos_theta
        return points[1:]

    def get_rotation(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        rotation = np.eye(4)
        g = v / np.linalg.norm(v)
        r = np.cross(g, u)
        r /= np.linalg.norm(r)
        t = np.cross(r, g)
        t /= np.linalg.norm(t)
        rotation[0, :3] = r
        rotation[1, :3] = t
        rotation[2, :3] = -g
        return rotation

    def get_translation(v: np.ndarray) -> np.ndarray:
        translation = np.eye(4)
        translation[:3, 3] = -v
        return translation

    def look_at(direction: np.ndarray, up: np.ndarray, eye: np.ndarray) -> np.ndarray:
        translation = get_translation(eye)
        rotation = get_rotation(direction, up)
        return rotation @ translation

    positions = get_random_points_on_hemisphere(num_points) * radius
    if num_points_down > 0:
        positions_down = -get_random_points_on_hemisphere(num_points_down) * radius
        positions = np.concat([positions, positions_down], axis=0)
    transformations = {
        'camera_angle_x': angle_x,
        'frames': []
    }
    for i, pos in enumerate(positions):
        transformations['frames'].append(
            {
                'rotation': 0.0,
                'transform_matrix': np.linalg.inv(look_at(-pos, np.array([0, 0, 1]), pos)).tolist()
            }
        )
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(transformations, f, ensure_ascii=False, indent=4)
