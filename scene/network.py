#
# Created by lxl.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def compute_volume_tv(v: torch.tensor, p: int=2):
    if p == 2:
        h_tv = torch.mean((v[..., 1:, :, :] - v[..., :-1, :, :]) ** 2)
        w_tv = torch.mean((v[..., :, 1:, :] - v[..., :, :-1, :]) ** 2)
        l_tv = torch.mean((v[..., :, :, 1:] - v[..., :, :, :-1]) ** 2)
    else:
        h_tv = torch.mean(torch.abs(v[..., 1:, :, :] - v[..., :-1, :, :]))
        w_tv = torch.mean(torch.abs(v[..., :, 1:, :] - v[..., :, :-1, :]))
        l_tv = torch.mean(torch.abs(v[..., :, :, 1:] - v[..., :, :, :-1]))
    return h_tv + w_tv + l_tv

def compute_volume_sep(v: torch.tensor):
    return torch.mean(torch.abs(v))

def compute_volume_smooth(v: torch.tensor):
    h_diff = v[..., 1:, :, :] - v[..., :-1, :, :]
    w_diff = v[..., :, 1:, :] - v[..., :, :-1, :]
    l_diff = v[..., :, :, 1:] - v[..., :, :, :-1]
    h_smo = torch.mean(torch.abs(h_diff[..., 1:, :, :] - h_diff[..., :-1, :, :]))
    w_smo = torch.mean(torch.abs(w_diff[..., :, 1:, :] - w_diff[..., :, :-1, :]))
    l_smo = torch.mean(torch.abs(l_diff[..., :, :, 1:] - l_diff[..., :, :, :-1]))
    return h_smo + w_smo + l_smo

class FeatureVolume(nn.Module):
    def __init__(self, out_dim, res, bounds=1.6, num_dim=3):
        super().__init__()
        self._grid = torch.nn.Parameter(torch.randn([1, out_dim] + [res] * num_dim, dtype=torch.float32))
        self._grid2 = torch.nn.Parameter(torch.randn([1, out_dim] + [res * 2] * num_dim, dtype=torch.float32))
        self._aabb = nn.Parameter(
            torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]]), requires_grad=False
        )
        self.out_dim = out_dim

    @property
    def get_aabb(self):
        return self._aabb[0], self._aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self._aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb =", self._aabb)

    def forward(self, pts):
        pts = normalize_aabb(pts, self.get_aabb)
        feat = F.grid_sample(self._grid, pts[None, None, None, :, :], mode='bilinear',
                             align_corners=True)  # [1, C, 1, 1, N]
        # return feat[0, :, 0, 0, :].permute(1, 0)  # [N, C]
        feat2 = F.grid_sample(self._grid2, pts[None, None, None, :, :], mode='bilinear',
                             align_corners=True)  # [1, C, 1, 1, N]
        return torch.cat((feat[0, :, 0, 0, :].permute(1, 0), feat2[0, :, 0, 0, :].permute(1, 0)), dim=1)

    def eval_tv(self):
        return compute_volume_tv(self._grid) + compute_volume_tv(self._grid2)

    def eval_sep(self):
        return compute_volume_sep(self._grid) + compute_volume_sep(self._grid2)

    def eval_smo(self):
        return compute_volume_smooth(self._grid) + compute_volume_smooth(self._grid2)

class DeformNet(nn.Module):
    def __init__(
            self,
            voxel_dim: int=20,  # feature volume
            voxel_res: int=50,
            hidden_dim: int=64, # mlp
            out_dim: int=20,
            num_layers: int=3,
    ):
        super(DeformNet, self).__init__()
        self.fv = FeatureVolume(voxel_dim, voxel_res)
        # self.mlp = self.create_mlp(out_dim, voxel_dim, hidden_dim, num_layers)
        self.mlp = self.create_mlp(out_dim, voxel_dim * 2, hidden_dim, num_layers)
        self.pos_head = self.create_head(out_dim, 3)
        self.rot_head = self.create_head(out_dim, 4)

    @staticmethod
    def create_mlp(dim_out: int, dim_in: int=20, width: int=64, layers: int=3) -> nn.Sequential:
        if layers == 1:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        return nn.Sequential(
            nn.Linear(dim_in, width), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True)) for _ in range(layers - 2)],
            nn.Linear(width, dim_out),
        )

    @staticmethod
    def create_head(width: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_features))

    @property
    def get_aabb(self):
        return self.fv.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb =", xyz_max, xyz_min)
        self.fv.set_aabb(xyz_max, xyz_min)

    def forward(self, pos: torch.tensor):
        hidden = self.mlp(self.fv(pos))
        dt = self.pos_head(hidden)
        dr = self.rot_head(hidden)
        dr += torch.tensor([1, 0, 0, 0], device=pos.device)
        dr = dr / dr.norm(dim=1, keepdim=True)
        return dt, dr

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "fv" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "fv" in name:
                parameter_list.append(param)
        return parameter_list

    def eval_tv(self):
        return self.fv.eval_tv()

    def eval_sep(self):
        return self.fv.eval_sep()

    def eval_smo(self):
        return self.fv.eval_smo()

if __name__ == '__main__':
    dn = DeformNet()
    points = torch.randn(7, 3)

    print(points.max(dim=0)[0])
    print(points.min(dim=0)[0])
    print()
    dn.set_aabb(points.max(dim=0)[0].tolist(), points.min(dim=0)[0].tolist())

    print(dn(points))
    print([p.shape for p in dn.get_mlp_parameters()])
    print([p.shape for p in dn.get_grid_parameters()])

    for name, param in dn.named_parameters():
        print(name)

    pass
