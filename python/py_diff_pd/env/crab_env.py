from pathlib import Path

import numpy as np
import os

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import print_info, create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, read_tetgen_file
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable, StdRealVector
from py_diff_pd.common.renderer import PbrtRenderer


class CrabEnv(EnvBase):
    """
    Crab environment with per-element heterogeneous Young's modulus loaded from
    an optional crab_mu.npy stiffness file (falls back to a uniform Neohookean
    material when the file is absent).
    """

    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options.get('youngs_modulus', 1e6)
        poissons_ratio = options.get('poissons_ratio', 0.45)
        state_force_parameters = options.get('state_force_parameters', ndarray([0.0, 0.0, -9.81]))

        self.__visualize_boundary_nodes = bool(options.get('visualize_boundary_nodes', False))
        self.__boundary_node_radius = float(options.get('boundary_node_radius', 0.0015))

        density = 1e3

        # Load crab mesh.
        ele_file_name = Path(root_path) / 'asset' / 'mesh' / 'crab.ele'
        node_file_name = Path(root_path) / 'asset' / 'mesh' / 'crab.node'
        verts, eles = read_tetgen_file(node_file_name, ele_file_name)

        # Optional per-element stiffness file.
        stiff_file_name = Path(root_path) / 'asset' / 'mesh' / 'crab_mu.npy'
        element_modulus_array = None
        if stiff_file_name.exists():
            print_info(f"Loading stiffness values from {stiff_file_name}...")
            element_modulus_array = np.load(stiff_file_name).ravel()
            element_modulus_array[element_modulus_array > 1e10] = 1e7
        else:
            print_info(f"Stiffness file {stiff_file_name} not found. Using uniform stiffness.")

        # To make the mesh consistent with our coordinate system.
        R = ndarray([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        verts = verts @ R.T
        min_z = np.min(verts, axis=0)[2]
        verts[:, 2] -= min_z
        verts /= 1000

        tmp_bin_file_name = str(Path(folder) / '.tmp_crab.bin')
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        self.__mesh = mesh

        self.het = element_modulus_array is not None
        if self.het:
            element_modulus_vector = StdRealVector(list(element_modulus_array.astype(np.float64)))
            deformable = TetDeformable()
            deformable.InitializeHeterogeneous(tmp_bin_file_name, density, 'neohookean',
                element_modulus_vector, poissons_ratio)
        else:
            deformable = TetDeformable()
            deformable.Initialize(tmp_bin_file_name, density, 'neohookean', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)
        self.__element_modulus_array = element_modulus_array

        # Boundary conditions: fix the lowest z nodes; track min/max-x and max-z nodes.
        vert_num = mesh.NumOfVertices()
        all_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        max_corner = np.max(all_verts, axis=0)
        min_corner = np.min(all_verts, axis=0)
        center = (max_corner + min_corner) / 2
        min_z, max_z = min_corner[2], max_corner[2]
        min_x, max_x = min_corner[0], max_corner[0]
        dirichlet_dofs = []
        self.__min_x_nodes = []
        self.__max_x_nodes = []
        self.__max_z_nodes = []
        for i in range(vert_num):
            vx, vy, vz = all_verts[i]
            if vx - min_x < 1e-3:
                self.__min_x_nodes.append(i)
            if max_x - vx < 1e-3:
                self.__max_x_nodes.append(i)
            if vz - min_z < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs += [3 * i, 3 * i + 1, 3 * i + 2]
            if max_z - vz < 1e-3:
                self.__max_z_nodes.append(i)
        self.__dirichlet_dofs = dirichlet_dofs

        deformable.AddStateForce('gravity', state_force_parameters)

        # Initial state: rotate about z, scaled linearly by height.
        q0 = np.copy(all_verts)
        theta = float(options.get('init_rotate_angle', 0))
        for i in range(vert_num):
            vi = all_verts[i]
            th = (vi[2] - min_z) / (max_z - min_z) * theta
            c, s = np.cos(th), np.sin(th)
            R = ndarray([[c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])
            q0[i] = R @ (vi - center) + center

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)

        self.__spp = options.get('spp', 4)

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # This (0, 2) shape is due to the usage of Neohookean materials.
        return np.zeros((0, 2))

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dofs

    def get_element_modulus_array(self):
        if self.het:
            return self.__element_modulus_array
        return np.full(self.__mesh.NumOfElements(), self._youngs_modulus)

    def min_x_nodes(self):
        return self.__min_x_nodes

    def max_x_nodes(self):
        return self.__max_x_nodes

    def max_z_nodes(self):
        return self.__max_z_nodes

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)

    def _create_filtered_mesh_by_elements(self, q, element_indices, temp_file):
        vertex_num = self.__mesh.NumOfVertices()
        vertices = q.reshape((vertex_num, 3))

        filtered_elements = []
        for elem_idx in element_indices:
            ele_verts = ndarray(self.__mesh.py_element(int(elem_idx)))
            filtered_elements.append(ele_verts.astype(int))

        if len(filtered_elements) == 0:
            return False

        filtered_elements = ndarray(filtered_elements).astype(int)
        generate_tet_mesh(vertices, filtered_elements, temp_file)
        return True

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.12, -0.8, 0.34),
            'camera_lookat': (0, 0, .15),
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        vert_num = mesh.NumOfVertices()
        all_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        q = all_verts.ravel()

        if not self.het:
            renderer.add_tri_mesh(mesh, color='0096c7',
                transforms=[('s', 2)], render_tet_edge=True)
        else:
            element_stiffness = self.__element_modulus_array
            min_stiffness = float(np.min(element_stiffness))
            max_stiffness = float(np.max(element_stiffness))
            stiff_elements = [i for i in range(len(element_stiffness)) if element_stiffness[i] == max_stiffness]
            soft_elements = [i for i in range(len(element_stiffness)) if element_stiffness[i] == min_stiffness]
            stiff_mesh_file = f'{mesh_file}_stiff.mesh'
            soft_mesh_file = f'{mesh_file}_soft.mesh'
            self._create_filtered_mesh_by_elements(q, stiff_elements, stiff_mesh_file)
            self._create_filtered_mesh_by_elements(q, soft_elements, soft_mesh_file)
            stiff_mesh = TetMesh3d()
            stiff_mesh.Initialize(stiff_mesh_file)
            soft_mesh = TetMesh3d()
            soft_mesh.Initialize(soft_mesh_file)
            renderer.add_tri_mesh(stiff_mesh, color='ff4d4d',
                transforms=[('s', 2)], render_tet_edge=True)
            renderer.add_tri_mesh(soft_mesh, color='0096c7',
                transforms=[('s', 2)], render_tet_edge=True)
            os.remove(stiff_mesh_file)
            os.remove(soft_mesh_file)

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        if self.__visualize_boundary_nodes:
            for idx in self.__max_z_nodes:
                renderer.add_shape_mesh(
                    {
                        'name': 'sphere',
                        'center': all_verts[int(idx)],
                        'radius': self.__boundary_node_radius,
                    },
                    transforms=[('s', 2)],
                    color='4dff4d')

        renderer.render()
