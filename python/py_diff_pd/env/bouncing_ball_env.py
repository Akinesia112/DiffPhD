import os
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import get_contact_vertex, filter_hex
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.common.project_path import root_path


class BouncingBallEnv(EnvBase):
    """Bouncing ball env. 'het': True splits elements into stiff/mid/soft
    z-height bands (see _build_element_groups); otherwise the ball is a
    single homogeneous material.
    """

    default_band_multipliers = ndarray([3.0, 0.1, 1.0])

    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = options.get('youngs_modulus', 1e6)
        poissons_ratio = options.get('poissons_ratio', 0.45)
        state_force_parameters = options.get('state_force_parameters', ndarray([0.0, 0.0, -9.81]))
        density = 1e3

        # Shape of the bouncing body.
        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'lock.bin'
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        mesh.Scale(0.2)
        self.__mesh = mesh
        element_num = mesh.NumOfElements()

        self.het = bool(options.get('het', False))
        if self.het:
            self._build_element_groups(mesh, element_num)
            per_element_E = self._build_element_modulus(mesh, element_num, youngs_modulus, options)
        else:
            self.stiff_ele, self.mid_ele, self.soft_ele = [], [], []
            per_element_E = np.full(element_num, youngs_modulus, dtype=float)
        self.__element_modulus_array = per_element_E

        tmp_bin_file_name = str(Path(folder) / '.tmp.bin')
        mesh.SaveToFile(tmp_bin_file_name)
        deformable = HexDeformable()
        avg_E = float(np.mean(per_element_E))
        deformable.Initialize(tmp_bin_file_name, density, 'none', avg_E, poissons_ratio)
        os.remove(tmp_bin_file_name)

        per_element_mu = per_element_E / (2 * (1 + poissons_ratio))
        per_element_la = per_element_E * poissons_ratio / (
            (1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        if self.het:
            self._add_grouped_corotated_volume_energies(deformable, per_element_mu, per_element_la)
        else:
            deformable.AddPdEnergy('corotated', [float(2.0 * per_element_mu[0])], [])
            deformable.AddPdEnergy('volume', [float(per_element_la[0])], [])

        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

        # Collisions.
        friction_node_idx = get_contact_vertex(mesh)
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Initial state.
        dofs = deformable.dofs()
        print('Bouncing ball element: {:d}, DoFs: {:d}.'.format(mesh.NumOfElements(), dofs))
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = avg_E
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = True

        self._spp = options.get('spp', 4)
        self._camera_pos = (0.2, -1, .25)
        self._camera_lookat = (0.2, 0, 0.1)
        self._color = (0.3, 0.9, 0.3)
        self._scale = 0.5

    def mesh(self):
        return self.__mesh

    def get_element_modulus_array(self):
        return self.__element_modulus_array

    def _build_element_groups(self, mesh, element_num):
        self.stiff_ele = []
        self.mid_ele = []
        self.soft_ele = []

        vertices = ndarray(mesh.py_vertices()).reshape((-1, 3))
        min_z = np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])
        z_range = max_z - min_z

        for i in range(element_num):
            ele = ndarray(mesh.py_element(i)).astype(int)
            center_z = np.mean(vertices[ele, 2])
            relative_z = (center_z - min_z) / z_range if z_range > 0 else 0.5

            if relative_z < 1.0 / 3.0:
                self.stiff_ele.append(i)
            elif relative_z < 2.0 / 3.0:
                self.mid_ele.append(i)
            else:
                self.soft_ele.append(i)

    def element_modulus_from_band_multipliers(self, base_youngs_modulus, band_multipliers):
        band_multipliers = ndarray(band_multipliers).ravel()
        assert band_multipliers.size == 3, \
            'band_multipliers must contain stiff, mid, and soft values'
        assert np.all(np.isfinite(band_multipliers)), \
            'band_multipliers contains non-finite values'
        assert np.all(band_multipliers > 0), \
            'band_multipliers must be positive'

        per_element_E = np.full(self.__mesh.NumOfElements(), base_youngs_modulus, dtype=float)
        per_element_E[self.stiff_ele] = base_youngs_modulus * band_multipliers[0]
        per_element_E[self.mid_ele] = base_youngs_modulus * band_multipliers[1]
        per_element_E[self.soft_ele] = base_youngs_modulus * band_multipliers[2]
        return per_element_E

    def element_modulus_from_log_band_multipliers(self, base_youngs_modulus, log_band_multipliers):
        return self.element_modulus_from_band_multipliers(
            base_youngs_modulus, np.exp(ndarray(log_band_multipliers).ravel()))

    def _build_element_modulus(self, mesh, element_num, default_E, options):
        if 'log_band_multipliers' in options:
            return self.element_modulus_from_log_band_multipliers(
                default_E, options['log_band_multipliers'])
        elif 'band_multipliers' in options:
            return self.element_modulus_from_band_multipliers(
                default_E, options['band_multipliers'])
        return self.element_modulus_from_band_multipliers(default_E, self.default_band_multipliers)

    def _add_grouped_corotated_volume_energies(self, deformable, per_element_mu, per_element_la):
        modulus = self.__element_modulus_array
        for E in np.unique(modulus):
            indices = np.flatnonzero(modulus == E).astype(int).tolist()
            sample_idx = indices[0]
            deformable.AddPdEnergy('corotated',
                [float(2.0 * per_element_mu[sample_idx])], indices)
            deformable.AddPdEnergy('volume',
                [float(per_element_la[sample_idx])], indices)

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        if self.het:
            return np.zeros((self._deformable.NumOfPdElementEnergies(), 2))
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def band_multiplier_gradients(self, dl_dmat_w, log_band_multipliers):
        """Compute dl/d(log_mult_k) for k in {0,1,2} from backward dl_dmat_w.

        PD energies were registered by iterating np.unique(modulus) ascending,
        so the ordering here must match that registration order.
        """
        band_mults = np.exp(ndarray(log_band_multipliers).ravel())
        nu = self._poissons_ratio
        base_E = self._youngs_modulus
        band_E = base_E * band_mults          # [E_stiff, E_mid, E_soft]
        _, inverse = np.unique(np.round(band_E, decimals=8), return_inverse=True)
        dl_dlog_mult = np.zeros(3)
        for band_k in range(3):
            i = inverse[band_k]
            dl_corot = dl_dmat_w[2 * i]
            dl_vol   = dl_dmat_w[2 * i + 1]
            dl_dEk = dl_corot / (1.0 + nu) + dl_vol * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            dl_dlog_mult[band_k] = dl_dEk * base_E * band_mults[band_k]
        return dl_dlog_mult

    def is_dirichlet_dof(self, dof):
        return False

    def _display_mesh(self, mesh_file, file_name):
        if not (self.stiff_ele and self.mid_ele and self.soft_ele):
            super()._display_mesh(mesh_file, file_name)
            return

        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': self._camera_pos,
            'camera_lookat': self._camera_lookat,
            'resolution': self._resolution
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        transforms = [('s', self._scale)]
        stiff_mesh = filter_hex(mesh, self.stiff_ele)
        mid_mesh = filter_hex(mesh, self.mid_ele)
        soft_mesh = filter_hex(mesh, self.soft_ele)
        renderer.add_hex_mesh(stiff_mesh, render_voxel_edge=True,
            color='ff4d4d', transforms=transforms)
        renderer.add_hex_mesh(mid_mesh, render_voxel_edge=True,
            color='9932cc', transforms=transforms)
        renderer.add_hex_mesh(soft_mesh, render_voxel_edge=True,
            color='0096c7', transforms=transforms)

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 3)])
        renderer.render()

    def _stepwise_loss_and_grad(self, q, v, i):
        mesh_file = self._folder / 'groundtruth' / '{:04d}.bin'.format(i)
        if not mesh_file.exists(): return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = HexMesh3d()
        mesh.Initialize(str(mesh_file))
        q_ref = ndarray(mesh.py_vertices())
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)
