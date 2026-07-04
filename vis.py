import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, get_source_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image

import argparse

from os.path import join as pjoin

import importlib
import json
import math
import numpy as np
from arguments import get_default_args
from main_utils import get_gaussians

from utils.general_utils import mat2quat, quat_mult


def deform(target: GaussianModel, source: GaussianModel, trans_info: dict,
           time):
  d = torch.tensor(trans_info['axis']['d'],
                   dtype=target.get_xyz.dtype,
                   device=target.get_xyz.device)
  o = torch.tensor(trans_info['axis']['o'],
                   dtype=target.get_xyz.dtype,
                   device=target.get_xyz.device)
  if trans_info['type'] == 'translate':
    t = d / torch.norm(d) * trans_info['translate'] * time
    target.get_xyz[:] = source.get_xyz[:] + t
  elif trans_info['type'] == 'rotate':
    theta = math.radians(trans_info['rotate']) * time
    K = torch.tensor([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]],
                     dtype=target.get_xyz.dtype,
                     device=target.get_xyz.device)
    I = torch.eye(3, dtype=target.get_xyz.dtype, device=target.get_xyz.device)
    r = I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    t = o - r @ o
    target.get_xyz[:] = torch.einsum('ij,nj->ni', r, source.get_xyz[:]) + t
    target.get_rotation_raw[:] = quat_mult(mat2quat(r), source.get_rotation[:])
  pass


def render_view(num_movable: int,
                model_path: str,
                data_path: str,
                view_idx: int,
                iteration: int = 30,
                num_frames: int = 10):
  with torch.no_grad():
    dataset, pipes, opt = get_default_args()
    dataset.eval = True
    dataset.sh_degree = 0
    dataset.source_path = os.path.realpath(pjoin(data_path, 'start'))
    dataset.model_path = model_path
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    parts = [
        get_gaussians(model_path, from_chk=False, iters=iteration + k)
        for k in range(num_movable + 1)
    ]
    deformed_parts = [
        get_gaussians(model_path, from_chk=False, iters=iteration + k)
        for k in range(1, num_movable + 1)
    ]
    scene = Scene(dataset, parts[0], load_iteration=iteration, shuffle=False)

    with open(pjoin(model_path, 'trans_pred.json'), 'r') as json_file:
      trans = json.load(json_file)

    render_path = pjoin(model_path, f'train/view_{view_idx}')
    makedirs(render_path, exist_ok=True)
    print(f'rendering to {render_path}')
    view = scene.getTrainCameras()[view_idx]
    print(view.image_name)
    times = np.concatenate(
        (np.linspace(0., 1.,
                     num_frames // 2), np.linspace(1., 0., num_frames // 2)))
    for idx, time in enumerate(tqdm(times, desc="Rendering progress")):
      # print(f'Rendering frame {idx} at time {time:.2f}')
      gaussians = parts[0]
      for k in range(num_movable):
        deform(deformed_parts[k], parts[k + 1], trans[k], time)
        gaussians += deformed_parts[k]

      render_pkg = render(view, gaussians, pipes, background)
      rendering = render_pkg["render"]
      torchvision.utils.save_image(
          rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    try:
      imageio = importlib.import_module('imageio.v2')
    except ImportError:
      imageio = None

    if imageio is None:
      print('Skipping video export because imageio is not available.')
      return

    frame_files = sorted([
        os.path.join(render_path, name)
        for name in os.listdir(render_path)
        if name.endswith('.png')
    ])
    if not frame_files:
      print('No frames found for video export.')
      return

    makedirs('video', exist_ok=True)
    video_name = model_path.replace('/', '_').replace('.', 'v')
    video_path = os.path.join('video', f'{video_name}.mp4')
    print(f'Exporting video to {video_path}')

    # Build a short video preview from the rendered png frames
    with imageio.get_writer(video_path, fps=30) as writer:
      for frame_path in frame_files:
        writer.append_data(imageio.imread(frame_path))


if __name__ == '__main__':
  K = 4
  data = './datasets/artpro/teeburu34178'
  out = 'outputs_final/artpro/tbr4'
  index = 11

  parser = ArgumentParser()
  parser.add_argument("--num_movable", type=int, default=K)
  parser.add_argument("--model_path", type=str, default=out)
  parser.add_argument("--data_path", type=str, default=data)
  parser.add_argument("--view_idx", type=int, default=index)
  args = parser.parse_args()

  # render_view(K, out, data, index)
  render_view(args.num_movable,
              args.model_path,
              args.data_path,
              args.view_idx,
              num_frames=60)

  pass
"""
CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 4 --model_path ./outputs_final/artgs/table_31249 --data_path ./datasets/artgs/table_31249 --view_idx 0

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 3 --model_path ./outputs_final/artgs/table_25493 --data_path ./datasets/artgs/table_25493 --view_idx 0

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 6 --model_path ./outputs_final/artgs/storage_47648 --data_path ./datasets/artgs/storage_47648 --view_idx 0

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 3 --model_path ./outputs_final/artgs/storage_45503 --data_path ./datasets/artgs/storage_45503 --view_idx 28

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 3 --model_path ./outputs_final/artgs/oven_101908 --data_path ./datasets/artgs/oven_101908 --view_idx 0

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 2 --model_path ./outputs_final/artpro/window103238 --data_path ./datasets/artpro/window103238 --view_idx 25

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 5 --model_path ./outputs_final/artpro/table34610 --data_path ./datasets/artpro/table34610 --view_idx 25

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 4 --model_path ./outputs_final/artpro/table34178 --data_path ./datasets/artpro/table34178 --view_idx 25

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 3 --model_path ./outputs_final/artpro/table33116 --data_path ./datasets/artpro/table33116 --view_idx 91

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 4 --model_path ./outputs_final/artpro/table23372 --data_path ./datasets/artpro/table23372 --view_idx 25

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 10 --model_path ./outputs_final/artpro/storage47585 --data_path ./datasets/artpro/storage47585 --view_idx 39

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 4 --model_path ./outputs_final/artpro/storage45759 --data_path ./datasets/artpro/storage45759 --view_idx 25

CUDA_VISIBLE_DEVICES=1 python vis.py --num_movable 6 --model_path ./outputs_final/artpro/storage40417 --data_path ./datasets/artpro/storage40417 --view_idx 25

"""
