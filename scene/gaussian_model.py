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

import copy
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from utils.general_utils import knn
from arguments import get_default_args

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.prev_size = None
        self.neighbor_indices = None
        self.neighbor_weight = None
        self.neighbor_dist = None
        self.prev_xyz = None
        self.prev_rotation = None
        self.prev_rotation_inv = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        return self

    def restore_gpsr(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup_pgsr(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        # self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        # self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_raw(self):
        return self._rotation

    @get_rotation_raw.setter
    def get_rotation_raw(self, value):
        self._rotation = value
    
    @property
    def get_xyz(self):
        return self._xyz

    @get_xyz.setter
    def get_xyz(self, value):
        self._xyz = value
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_raw(self):
        return self._opacity

    @get_opacity_raw.setter
    def get_opacity_raw(self, value):
        self._opacity = value

    @property
    def get_scaling_raw(self):
        return self._scaling

    @get_scaling_raw.setter
    def get_scaling_raw(self, value):
        self._scaling = value

    @property
    def get_features_dc(self):
        return self._features_dc

    @get_features_dc.setter
    def get_features_dc(self, value):
        self._features_dc = value

    @property
    def get_features_rest(self):
        return self._features_rest

    @get_features_rest.setter
    def get_features_rest(self, value):
        self._features_rest = value

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_inverse_covariance(self, indices: torch.tensor):
        """
        Calculates the inverse of covariance matrix (with detached gaussian attributes)
        :return: inv cov
        """
        scaling = self.get_scaling.detach()
        rot_raw = self._rotation.detach()
        L = build_scaling_rotation(1 / (scaling + 1e-8), rot_raw)
        cov_inv = L @ L.transpose(1, 2)
        return cov_inv[indices]

    def eval_opacity_at(self, positions: torch.tensor, indices: torch.tensor):
        """
        Calculates the self's opacity at N positions (with detached gaussian attributes)
        :param positions: has shape of (N, 3)
        :param indices: has shape of (N, K)
        :return: tensor of (N,)
        """
        centers = self.get_xyz[indices].detach()
        opacities = self.get_opacity[:, 0][indices]             # (N, K)
        cov_inv = self.get_inverse_covariance(indices)  # (N, K, 3, 3)
        dist = positions.unsqueeze(1).detach() - centers         # (N, K, 3)
        quad =  torch.einsum('nki,nkij,nkj->nk', dist, cov_inv, dist).clamp(max=16, min=-16)
        opacities = opacities * torch.exp(-0.5 * quad)  # (N, K)
        return torch.sum(opacities, dim=1)

    def __getitem__(self, mask):
        gaussians = GaussianModel(self.max_sh_degree)
        gaussians._xyz = self._xyz[mask]
        gaussians._features_dc = self._features_dc[mask]
        gaussians._features_rest = self._features_rest[mask]
        gaussians._scaling = self._scaling[mask]
        gaussians._rotation = self._rotation[mask]
        gaussians._opacity = self._opacity[mask]
        return gaussians

    def __add__(self, other):
        gaussians = GaussianModel(self.max_sh_degree)
        gaussians._xyz = torch.cat((self._xyz, other._xyz), dim=0)
        gaussians._features_dc = torch.cat((self._features_dc, other._features_dc), dim=0)
        gaussians._features_rest = torch.cat((self._features_rest, other._features_rest), dim=0)
        gaussians._scaling = torch.cat((self._scaling, other._scaling), dim=0)
        gaussians._rotation = torch.cat((self._rotation, other._rotation), dim=0)
        gaussians._opacity = torch.cat((self._opacity, other._opacity), dim=0)
        return gaussians

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup_pgsr(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        # self.max_abs_split_points = training_args.max_abs_split_points
        # self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path, min_opacity=0.005, auxiliary_attr=None, prune=False):
        self.save_ply_helper(path)
        # removes gaussians whose opacities is too small, ensuring the accuracy in final point cloud
        if prune:
            gaussians = GaussianModel(0).load_ply(path)
            _, _, opt = get_default_args()
            gaussians.training_setup(opt)
            gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
            prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
            auxiliary_attr = gaussians.prune_points(prune_mask, auxiliary_attr=auxiliary_attr)
            gaussians.save_ply_helper(path)
        return auxiliary_attr

    def save_ply_helper(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        return self

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, auxiliary_attr=None):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        if auxiliary_attr is not None:
            if isinstance(auxiliary_attr, tuple):
                auxiliary_attr = tuple(attr[valid_points_mask] for attr in auxiliary_attr)
            else:
                auxiliary_attr = auxiliary_attr[valid_points_mask]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        return auxiliary_attr

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, auxiliary_attr=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if auxiliary_attr is not None:
            if isinstance(auxiliary_attr, tuple):
                new_attrs = tuple(attr[selected_pts_mask].repeat(N) for attr in auxiliary_attr)
                auxiliary_attr = tuple(
                    torch.cat((existing, new), dim=0) for existing, new in zip(auxiliary_attr, new_attrs))
            else:
                new_attr = auxiliary_attr[selected_pts_mask].repeat(N)
                auxiliary_attr = torch.cat((auxiliary_attr, new_attr), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        auxiliary_attr = self.prune_points(prune_filter, auxiliary_attr=auxiliary_attr)

        return auxiliary_attr

    def densify_and_clone(self, grads, grad_threshold, scene_extent, auxiliary_attr=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if auxiliary_attr is not None:
            if isinstance(auxiliary_attr, tuple):
                new_attrs = tuple(attr[selected_pts_mask] for attr in auxiliary_attr)
                auxiliary_attr = tuple(
                    torch.cat((existing, new), dim=0) for existing, new in zip(auxiliary_attr, new_attrs)
                )
            else:
                new_attr = auxiliary_attr[selected_pts_mask]
                auxiliary_attr = torch.cat((auxiliary_attr, new_attr), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return auxiliary_attr

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, auxiliary_attr=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        auxiliary_attr = self.densify_and_clone(grads, max_grad, extent, auxiliary_attr=auxiliary_attr)
        auxiliary_attr = self.densify_and_split(grads, max_grad, extent, auxiliary_attr=auxiliary_attr)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        auxiliary_attr = self.prune_points(prune_mask, auxiliary_attr=auxiliary_attr)

        torch.cuda.empty_cache()
        return auxiliary_attr

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def cancel_grads(self):
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        # self.optimizer = None
        return self

    def enable_grads(self):
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)
        return self

    def training_se3_setup(self, training_args):
        # self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        # self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def enable_se3_grads(self):
        self._xyz.requires_grad_(True)
        self._rotation.requires_grad_(True)

    def size(self):
        return len(self._xyz)

    def initialize_neighbors(self, num_knn=20, lambda_omega=2000, simple=False):
        neighbor_dist, neighbor_indices = knn(self._xyz.detach().cpu().numpy(), num_knn)
        neighbor_weight = 2 * np.exp(-lambda_omega * neighbor_dist)
        self.prev_size = self.size()
        self.neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
        self.neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
        if simple:
            return
        self.neighbor_dist = torch.tensor(neighbor_dist).cuda().float().contiguous()
        self.prev_xyz = self._xyz.detach()
        self.prev_rotation = self.rotation_activation(self._rotation).detach()
        self.prev_rotation_inv = self.prev_rotation
        self.prev_rotation_inv[:, 1:] *= -1

    def set_colors(self, rgb_tensor: torch.Tensor):
        with torch.no_grad():
            self._features_dc.copy_(RGB2SH(rgb_tensor).unsqueeze(1))
            self._features_rest.zero_()

    def hide_static(self, dist, threshold):
        with torch.no_grad():
            self._opacity[dist < threshold] = -1e514

    def duplicate(self, n_rep: int=2):
        self._xyz = torch.cat([self._xyz] * n_rep, dim=0)
        self._features_dc = torch.cat([self._features_dc] * n_rep, dim=0)
        self._features_rest = torch.cat([self._features_rest] * n_rep, dim=0)
        self._opacity = torch.cat([self._opacity] * n_rep, dim=0)
        self._scaling = torch.cat([self._scaling] * n_rep, dim=0)
        self._rotation = torch.cat([self._rotation] * n_rep, dim=0)

    def op_grad(self):
        return self._opacity.grad

    def save_vis(self, path: str, fused_color):
        mkdir_p(os.path.dirname(path))

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity_raw.detach().cpu().numpy()
        scale = self.get_scaling_raw.detach().cpu().numpy()
        rotation = self.get_rotation_raw.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
