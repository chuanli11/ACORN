import numpy as np
import copy
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree as spKDTree
from data_structs import QuadTree, OctTree

from photon_library import PhotonLibrary


class VolumeDataset():
    def __init__(self, filename):
        if not filename:
            return

        step = 5
        plib = PhotonLibrary(filename)
        self.data = plib.numpy()
        self.data = self.data[0:-1:step, 0:-1:step, 0:-1:step, :]
        self.data_shape = self.data.shape[0:-1]

        # get the min and max
        buffer = np.reshape(np.sum(self.data, -1), (self.data.shape[0], -1)) 
        buffer = (buffer - np.amin(buffer)) / (np.amax(buffer) - np.amin(buffer)) - 0.5
        data_min = np.min(buffer, axis=1)
        data_max = np.max(buffer, axis=1)
        data_mean = np.mean(abs(buffer), axis=1)
        data_std = np.std(abs(buffer), axis=1)

        self.data = np.reshape(self.data, (-1, self.data.shape[-1]))
        self.data = np.sum(self.data, -1)
        self.data = ((self.data - np.amin(self.data)) / (np.amax(self.data) - np.amin(self.data)) - 0.5).astype(np.float32)

        x = np.linspace(0, self.data_shape[0] - 1, self.data_shape[0])
        y = np.linspace(0, self.data_shape[1] - 1, self.data_shape[1])
        z = np.linspace(0, self.data_shape[2] - 1, self.data_shape[2])

        coordx, coordy, coordz = np.meshgrid(x, y, z)

        self.dim_x = len(x)
        self.dim_y = len(y)
        self.dim_z = len(z)

        self.coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))

        self.coord = (self.coord / self.data_shape[0:3] - 0.5) * 2.0
        self.kd_tree_sp = spKDTree(self.coord)


class Block3DWrapperMultiscaleAdaptive(torch.utils.data.Dataset):
    def __init__(self, dataset, octant_size=16, sidelength=None, random_coords=False,
                 max_octants=600, jitter=True, num_workers=0, length=1000, scale_init=3):

        self.length = length
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength, sidelength)
        self.sidelength = sidelength

        self.volume = torch.from_numpy(dataset.data)

        # initialize quad tree
        self.octtree = OctTree(sidelength, octant_size, mesh_kd_tree=dataset.kd_tree_sp)        

        self.num_scales = self.octtree.max_octtree_level - self.octtree.min_octtree_level + 1

        # set patches at coarsest level to be active
        octants = self.octtree.get_octants_at_level(scale_init)
        for p in octants:
            p.activate()

        # handle parallelization
        self.num_workers = num_workers

        # make a copy of the tree for each worker
        self.octtrees = []
        print('Dataset: preparing dataloaders...')
        for idx in tqdm(range(num_workers)):
            self.octtrees.append(copy.deepcopy(self.octtree))
        self.last_active_octants = self.octtree.get_active_octants()

        self.octant_size = octant_size
        self.dataset = dataset
        self.pointcloud = None
        self.jitter = jitter
        self.eval = False

        self.max_octants = max_octants

        self.iter = 0

        self.dim_x = dataset.dim_x
        self.dim_y = dataset.dim_y
        self.dim_z = dataset.dim_z

    def toggle_eval(self):
        if not self.eval:
            self.jitter_bak = self.jitter
            self.jitter = False
            self.eval = True
        else:
            self.jitter = self.jitter_bak
            self.eval = False

    def synchronize(self):
        self.last_active_octants = self.octtree.get_active_octants()
        if self.num_workers == 0:
            return
        else:
            for idx in range(self.num_workers):
                self.octtrees[idx].synchronize(self.octtree)

    def __len__(self):
        return self.length

    def get_frozen_octants(self, oversample):
        octtree = self.octtree

        # get fine coords, get frozen patches is only called at eval
        fine_rel_coords, fine_abs_coords, vals,\
            coord_patch_idx = octtree.get_frozen_samples(oversample)

        return fine_abs_coords, vals

    def get_eval_samples(self, oversample):
        octtree = self.octtree

        # get fine coords
        fine_rel_coords, fine_abs_coords, coord_octant_idx, _ = octtree.get_stratified_samples(self.jitter, eval=True, oversample=oversample)

        # get block coords
        octants = octtree.get_active_octants()
        coords = torch.stack([p.block_coord for p in octants], dim=0)
        scales = torch.stack([torch.tensor(p.scale) for p in octants], dim=0)[:, None]
        scales = 2*scales / (self.num_scales-1) - 1
        coords = torch.cat((coords, scales), dim=-1)

        coords = coords[coord_octant_idx]

        # query for occupancy
        sz_b, sz_p, _ = fine_abs_coords.shape

        in_dict = {'coords': coords,
                   'fine_abs_coords': fine_abs_coords,
                   'fine_rel_coords': fine_rel_coords,
                   'coord_octant_idx': torch.tensor(coord_octant_idx, dtype=torch.int)}
        return in_dict

    def interpolate_bilinear(self, volume, fine_abs_coords, psize):        
        n_blocks = fine_abs_coords.shape[0]
        fine_abs_coords = fine_abs_coords.reshape(n_blocks, psize, psize, psize, 3)
        x = fine_abs_coords[..., :1]
        y = fine_abs_coords[..., 1:2]
        z = fine_abs_coords[..., 2:]
        coords = torch.cat([z, y, x], dim=-1)

        out = []
        for block in coords:
            tmp = torch.nn.functional.grid_sample(volume[None, ...], block[None, ...],
                                                  mode='bilinear',
                                                  padding_mode='reflection',
                                                  align_corners=False)
            out.append(tmp)
        out = torch.cat(out, dim=0)
        out = out.permute(0, 2, 3, 4, 1)
        return out.reshape(n_blocks, psize**3, 1)

    def __getitem__(self, idx):
        assert(not self.eval)

        octtree = self.octtree
        if not self.eval and self.num_workers > 0:
            worker_idx = torch.utils.data.get_worker_info().id
            octtree = self.octtrees[worker_idx]

        # get fine coords
        fine_rel_coords, fine_abs_coords, coord_octant_idx, coord_global_idx = octtree.get_stratified_samples(self.jitter, eval=self.eval)

        # get block coords
        octants = octtree.get_active_octants()
        coords = torch.stack([p.block_coord for p in octants], dim=0)
        scales = torch.stack([torch.tensor(p.scale) for p in octants], dim=0)[:, None]
        scales = 2*scales / (self.num_scales-1) - 1
        coords = torch.cat((coords, scales), dim=-1)

        if self.eval:
            coords = coords[coord_octant_idx]

        # Need to inplement 3D bilinear interpolation
        gt = self.interpolate_bilinear(self.volume.reshape(1, self.dim_x, self.dim_y, self.dim_z), fine_abs_coords, self.octant_size)

        in_dict = {'coords': coords,
                   'fine_abs_coords': fine_abs_coords,
                   'fine_rel_coords': fine_rel_coords}
        gt_dict = {'gt': gt}

        return in_dict, gt_dict

    def update_octant_err(self, err_per_octant, step):
        assert err_per_octant.shape[0] == len(self.last_active_octants), \
            f"Trying to update the error in active patches but list of patches and error tensor" \
            f" sizes are mismatched: {err_per_octant.shape[0]} vs {len(self.last_active_octants)}" \
            f"step: {step}"

        for i, p in enumerate(self.last_active_octants):
            # Log the history of error
            p.update_error(err_per_octant[i], step)

        self.per_octant_error = err_per_octant

    def update_tiling(self):
        return self.octtree.solve_optim(self.max_octants)

