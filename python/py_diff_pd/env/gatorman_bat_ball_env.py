import os
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tetrahedralize
from py_diff_pd.core.py_diff_pd_core import TetDeformable, TetMesh3d
from py_diff_pd.env.env_base import EnvBase


def _extract_boundary_faces(elements, vertices):
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_dict = {}
    for tet in ndarray(elements).astype(int):
        for face in faces:
            vidx = (int(tet[face[0]]), int(tet[face[1]]), int(tet[face[2]]))
            key = tuple(sorted(vidx))
            if key in face_dict:
                del face_dict[key]
            else:
                face_dict[key] = vidx

    vertices = ndarray(vertices)
    centroid = vertices.mean(axis=0)
    oriented = []
    for vidx in face_dict.values():
        a, b, c = vertices[vidx[0]], vertices[vidx[1]], vertices[vidx[2]]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        if np.dot(normal, face_center - centroid) < 0:
            oriented.append((vidx[0], vidx[2], vidx[1]))
        else:
            oriented.append(vidx)
    return ndarray(oriented).astype(int)


def _rotation_matrix_from_degrees(rotation_degrees):
    rx, ry, rz = np.deg2rad(rotation_degrees)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotate_x = ndarray([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ])
    rotate_y = ndarray([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ])
    rotate_z = ndarray([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ])
    return rotate_z @ rotate_y @ rotate_x


class GatormanBatBallEnv(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e5
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        state_force_parameters = ndarray(
            options['state_force_parameters']
        ) if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        density = 1e3

        self.__include_ball = bool(options.get('include_ball', False))
        self.__base_node_tol = float(options.get('base_node_tol', 1e-3))
        self.__fix_head_anchor = bool(options.get('fix_head_anchor', True))
        self.__visualize_material_stiffness = bool(options.get('visualize_material_stiffness', False))
        self.__visualize_force_nodes = bool(options.get('visualize_force_nodes', False))
        self.__visualize_head_anchor = bool(options.get('visualize_head_anchor', self.__fix_head_anchor))
        self.__force_marker_radius = float(options.get('force_marker_radius', 0.0012))
        self.__head_anchor_marker_radius = float(options.get('head_anchor_marker_radius', 0.0025))
        self.__force_marker_stride = int(options.get('force_marker_stride', 20))
        self.__head_anchor_rank = int(options.get('head_anchor_rank', 10))
        self.__ball_radius = float(options.get('ball_radius', 0.008))
        self.__ball_center = ndarray(options.get('ball_center', [-0.075, -0.025, 0.045]))
        self.__ball_mesh_file = str(options.get('ball_mesh_file', 'sphere.obj'))
        self.__ball_rotation_degrees = ndarray(options.get('ball_rotation_degrees', [0.0, 0.0, 0.0]))
        self.__ball_youngs_modulus = float(options.get('ball_youngs_modulus', 1e6))
        self.__contact_model = str(options.get(
            'contact_model', 'mesh_contact' if self.__include_ball else 'none'
        ))
        self.__contact_radius = float(options.get('contact_radius', 0.5 * self.__ball_radius))
        self.__mesh_contact_kn = float(options.get('mesh_contact_kn', 6e3))
        self.__mesh_contact_kf = float(options.get('mesh_contact_kf', 0.0))
        self.__mesh_contact_mu = float(options.get('mesh_contact_mu', 0.2))
        self.__use_ball_support = bool(options.get('use_ball_support', False))
        self.__ball_support_size = ndarray(options.get('ball_support_size', [0.003, 0.003, 0.003]))
        self.__ball_support_top_z = float(options.get(
            'ball_support_top_z',
            self.__ball_center[2] - self.__ball_radius - 0.5 * self.__contact_radius
        ))
        self.__ball_support_youngs_modulus = float(options.get('ball_support_youngs_modulus', 1e6))
        self.__ball_support_contact_radius = float(options.get(
            'ball_support_contact_radius', self.__contact_radius
        ))
        self.__ball_support_contact_kn = float(options.get(
            'ball_support_contact_kn', self.__mesh_contact_kn
        ))
        self.__ball_support_contact_kf = float(options.get(
            'ball_support_contact_kf', self.__mesh_contact_kf
        ))
        self.__ball_support_contact_mu = float(options.get(
            'ball_support_contact_mu', self.__mesh_contact_mu
        ))
        self.__use_ball_ground = bool(options.get('use_ball_ground', self.__include_ball))
        self.__ball_ground_z = float(options.get('ball_ground_z', 0.0))
        self.__loss_type = str(options.get('loss_type', 'random_linear'))
        self.__pd_energy_mode = str(options.get('pd_energy_mode', 'per_element'))
        self.__pd_energy_groups = {}
        self.__sword_pd_energy_idx = None
        if self.__base_node_tol <= 0:
            raise ValueError('base_node_tol must be positive.')
        if self.__force_marker_radius <= 0:
            raise ValueError('force_marker_radius must be positive.')
        if self.__head_anchor_marker_radius <= 0:
            raise ValueError('head_anchor_marker_radius must be positive.')
        if self.__force_marker_stride <= 0:
            raise ValueError('force_marker_stride must be positive.')
        if self.__head_anchor_rank <= 0:
            raise ValueError('head_anchor_rank must be positive.')
        if self.__ball_radius <= 0:
            raise ValueError('ball_radius must be positive.')
        if self.__ball_center.shape != (3,):
            raise ValueError('ball_center must have shape (3,).')
        if self.__ball_rotation_degrees.shape != (3,):
            raise ValueError('ball_rotation_degrees must have shape (3,).')
        if self.__ball_youngs_modulus <= 0:
            raise ValueError('ball_youngs_modulus must be positive.')
        if self.__contact_model not in ('none', 'mesh_contact', 'mesh_boundary'):
            raise ValueError('contact_model must be "none", "mesh_contact", or "mesh_boundary".')
        if self.__contact_model == 'mesh_boundary' and self.__use_ball_ground:
            raise ValueError('mesh_boundary contact uses the single frictional boundary slot; set use_ball_ground=False.')
        if self.__contact_model == 'mesh_boundary' and self.__use_ball_support:
            raise ValueError('mesh_boundary contact is only wired for sword-ball; set use_ball_support=False.')
        if self.__contact_radius <= 0:
            raise ValueError('contact_radius must be positive.')
        if self.__mesh_contact_kn <= 0:
            raise ValueError('mesh_contact_kn must be positive.')
        if self.__mesh_contact_kf < 0:
            raise ValueError('mesh_contact_kf must be nonnegative.')
        if self.__mesh_contact_mu < 0:
            raise ValueError('mesh_contact_mu must be nonnegative.')
        if self.__ball_support_size.shape != (3,):
            raise ValueError('ball_support_size must have shape (3,).')
        if np.any(self.__ball_support_size <= 0):
            raise ValueError('ball_support_size values must be positive.')
        if self.__ball_support_youngs_modulus <= 0:
            raise ValueError('ball_support_youngs_modulus must be positive.')
        if self.__ball_support_contact_radius <= 0:
            raise ValueError('ball_support_contact_radius must be positive.')
        if self.__ball_support_contact_kn <= 0:
            raise ValueError('ball_support_contact_kn must be positive.')
        if self.__ball_support_contact_kf < 0:
            raise ValueError('ball_support_contact_kf must be nonnegative.')
        if self.__ball_support_contact_mu < 0:
            raise ValueError('ball_support_contact_mu must be nonnegative.')
        if self.__loss_type not in ('random_linear', 'ball_xy_distance'):
            raise ValueError('loss_type must be "random_linear" or "ball_xy_distance".')
        if self.__loss_type == 'ball_xy_distance' and not self.__include_ball:
            raise ValueError('loss_type="ball_xy_distance" requires include_ball=True.')
        if self.__pd_energy_mode not in (
                'per_element', 'region_grouped', 'region_grouped_corotated_volume'):
            raise ValueError(
                'pd_energy_mode must be "per_element", "region_grouped", '
                'or "region_grouped_corotated_volume".')

        verts, eles = self.__load_gatorman_mesh()
        element_modulus = self.__load_element_modulus(eles.shape[0], youngs_modulus)
        self.__gatorman_vertices = np.copy(verts)
        self.__gatorman_elements = np.copy(eles)
        self.__gatorman_vertex_indices = np.arange(verts.shape[0]).astype(int)
        self.__gatorman_element_indices = np.arange(eles.shape[0]).astype(int)
        self.__detect_regions(verts, eles, element_modulus, options)
        self.__detect_contact_surfaces(verts, eles)
        element_modulus = self.__apply_body_stiffness_scale(element_modulus, options)
        element_modulus = self.__apply_sword_stiffness(element_modulus, options)

        all_verts = verts
        all_eles = eles
        all_element_modulus = element_modulus
        if self.__include_ball:
            ball_verts, ball_eles = self.__load_ball_mesh()
            vertex_offset = verts.shape[0]
            element_offset = eles.shape[0]
            all_verts = np.vstack([verts, ball_verts])
            all_eles = np.vstack([eles, ball_eles + vertex_offset])
            ball_modulus = np.full(ball_eles.shape[0], self.__ball_youngs_modulus)
            all_element_modulus = np.concatenate([element_modulus, ball_modulus])
            self.__ball_vertex_indices = np.arange(
                vertex_offset, vertex_offset + ball_verts.shape[0]
            ).astype(int)
            self.__ball_element_indices = np.arange(
                element_offset, element_offset + ball_eles.shape[0]
            ).astype(int)
            self.__ball_vertices0 = np.copy(ball_verts)
            self.__ball_elements0 = np.copy(ball_eles + vertex_offset)
            self.__ball_surface_faces = _extract_boundary_faces(ball_eles, ball_verts) + vertex_offset
            self.__ball_surface_vertices = np.unique(self.__ball_surface_faces.ravel()).astype(int)
            if self.__use_ball_support:
                support_verts, support_eles = self.__create_ball_support_mesh()
                support_vertex_offset = all_verts.shape[0]
                support_element_offset = all_eles.shape[0]
                all_verts = np.vstack([all_verts, support_verts])
                all_eles = np.vstack([all_eles, support_eles + support_vertex_offset])
                support_modulus = np.full(
                    support_eles.shape[0], self.__ball_support_youngs_modulus
                )
                all_element_modulus = np.concatenate([all_element_modulus, support_modulus])
                self.__ball_support_vertex_indices = np.arange(
                    support_vertex_offset,
                    support_vertex_offset + support_verts.shape[0]
                ).astype(int)
                self.__ball_support_element_indices = np.arange(
                    support_element_offset,
                    support_element_offset + support_eles.shape[0]
                ).astype(int)
                self.__ball_support_surface_faces = (
                    _extract_boundary_faces(support_eles, support_verts) + support_vertex_offset
                )
                self.__ball_support_surface_vertices = np.unique(
                    self.__ball_support_surface_faces.ravel()
                ).astype(int)
            else:
                self.__ball_support_vertex_indices = np.zeros(0, dtype=int)
                self.__ball_support_element_indices = np.zeros(0, dtype=int)
                self.__ball_support_surface_faces = np.zeros((0, 3), dtype=int)
                self.__ball_support_surface_vertices = np.zeros(0, dtype=int)
        else:
            self.__ball_vertex_indices = np.zeros(0, dtype=int)
            self.__ball_element_indices = np.zeros(0, dtype=int)
            self.__ball_vertices0 = np.zeros((0, 3))
            self.__ball_elements0 = np.zeros((0, 4), dtype=int)
            self.__ball_surface_faces = np.zeros((0, 3), dtype=int)
            self.__ball_surface_vertices = np.zeros(0, dtype=int)
            self.__ball_support_vertex_indices = np.zeros(0, dtype=int)
            self.__ball_support_element_indices = np.zeros(0, dtype=int)
            self.__ball_support_surface_faces = np.zeros((0, 3), dtype=int)
            self.__ball_support_surface_vertices = np.zeros(0, dtype=int)

        tmp_bin_file_name = Path(folder) / '.tmp_gatorman_bat_ball.bin'
        try:
            generate_tet_mesh(all_verts, all_eles, str(tmp_bin_file_name))
            mesh = TetMesh3d()
            mesh.Initialize(str(tmp_bin_file_name))

            deformable = TetDeformable()
            deformable.Initialize(str(tmp_bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
            if self.__pd_energy_mode == 'per_element':
                nh_params = self.__build_pd_neohookean_params(all_element_modulus, poissons_ratio)
                deformable.AddPdEnergy('pd_neohookean', nh_params, [])
            elif self.__pd_energy_mode == 'region_grouped':
                self.__add_region_grouped_pd_neohookean(
                    deformable, all_element_modulus, poissons_ratio
                )
            else:
                self.__add_region_grouped_corotated_volume(
                    deformable, all_element_modulus, poissons_ratio
                )
        finally:
            if tmp_bin_file_name.exists():
                os.remove(tmp_bin_file_name)

        self.__mesh = mesh
        self.__element_modulus_array = np.copy(all_element_modulus)

        mesh_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(mesh.NumOfVertices())])
        gatorman_mesh_verts = mesh_verts[self.__gatorman_vertex_indices]
        min_corner = np.min(gatorman_mesh_verts, axis=0)
        max_corner = np.max(gatorman_mesh_verts, axis=0)
        center = 0.5 * (min_corner + max_corner)
        body_nodes = np.setdiff1d(self.__gatorman_vertex_indices, self.__sword_vertices)
        top_body_nodes = body_nodes[np.argsort(mesh_verts[body_nodes, 2])[::-1]]
        head_anchor_idx = min(self.__head_anchor_rank, top_body_nodes.size) - 1
        self.__head_anchor_node = int(top_body_nodes[head_anchor_idx])
        self.__head_anchor_position = np.copy(mesh_verts[self.__head_anchor_node])

        self.__max_x_nodes = []
        self.__min_x_nodes = []
        self.__min_y_nodes = []
        dirichlet_dofs = set()
        for i in self.__gatorman_vertex_indices:
            i = int(i)
            vx, vy, vz = mesh.py_vertex(i)
            if vy - min_corner[1] < 5e-3:
                self.__min_y_nodes.append(i)
            if max_corner[0] - vx < 5e-3:
                self.__max_x_nodes.append(i)
            if vx - min_corner[0] < 5e-3:
                self.__min_x_nodes.append(i)
            if vz - min_corner[2] < self.__base_node_tol:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])
        if self.__fix_head_anchor:
            i = self.__head_anchor_node
            vx, vy, vz = self.__head_anchor_position
            deformable.SetDirichletBoundaryCondition(3 * i, vx)
            deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
            dirichlet_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])
        for i in self.__ball_support_vertex_indices:
            i = int(i)
            vx, vy, vz = mesh.py_vertex(i)
            deformable.SetDirichletBoundaryCondition(3 * i, vx)
            deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
            dirichlet_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])

        print(f'gatorman bat ball bounds: min={min_corner}, max={max_corner}, center={center}')
        print(
            f'Number of max x nodes: {len(self.__max_x_nodes)}, '
            f'min x nodes: {len(self.__min_x_nodes)}, min y nodes: {len(self.__min_y_nodes)}'
        )
        print(
            f'Region counts: base_nodes={self.__base_nodes.size}, '
            f'drive_nodes={self.__drive_nodes.size}, '
            f'sword_root_nodes={self.__sword_root_nodes.size}, '
            f'sword_elements={self.__sword_elements.size}, '
            f'ball_vertices={self.__ball_vertex_indices.size}, '
            f'ball_elements={self.__ball_element_indices.size}, '
            f'ball_support_vertices={self.__ball_support_vertex_indices.size}, '
            f'ball_support_elements={self.__ball_support_element_indices.size}, '
            f'ball_ground_nodes={self.__ball_surface_vertices.size if self.__use_ball_ground else 0}'
        )
        if self.__include_ball:
            print(
                f'Ball setup: center={self.__ball_center}, '
                f'mesh={self.__ball_mesh_file}, '
                f'rotation_degrees={self.__ball_rotation_degrees}, '
                f'radius={self.__ball_radius:.3e}, '
                f'youngs_modulus={self.__ball_youngs_modulus:.3e}'
            )
            if self.__use_ball_support:
                print(
                    f'Ball support: size={self.__ball_support_size}, '
                    f'top_z={self.__ball_support_top_z:.3e}, '
                    f'contact_radius={self.__ball_support_contact_radius:.3e}, '
                    f'contact_kn={self.__ball_support_contact_kn:.3e}, '
                    f'contact_mu={self.__ball_support_contact_mu:.3e}'
                )
        self.__print_material_stats()
        print(
            f'Head anchor node: {self.__head_anchor_node}, '
            f'position={self.__head_anchor_position}, fixed={self.__fix_head_anchor}, '
            f'rank={self.__head_anchor_rank}'
        )
        print(f'Number of dirichlet dofs: {len(dirichlet_dofs)}, total dofs: {3 * mesh.NumOfVertices()}')

        deformable.AddStateForce('gravity', state_force_parameters)
        if self.__include_ball and self.__contact_model == 'mesh_contact':
            self.__add_mesh_contact(deformable)
        if self.__include_ball and self.__contact_model == 'mesh_boundary':
            self.__add_mesh_boundary_contact(deformable, options)
        if self.__include_ball and self.__use_ball_ground:
            self.__add_ball_ground(deformable)

        dofs = deformable.dofs()
        self._deformable = deformable
        self._q0 = mesh_verts.ravel()
        self._v0 = ndarray(np.zeros(dofs)).ravel()
        self._f_ext = ndarray(np.zeros(dofs)).ravel()
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self._spp = options['spp'] if 'spp' in options else 4

        self.__dirichlet_dofs = sorted(dirichlet_dofs)
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__initial_ball_center_xy = None
        if self.__include_ball:
            self.__initial_ball_center_xy = self.ball_center(self._q0)[:2]

    def __load_gatorman_mesh(self):
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'gator_dec.obj'
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False)

        rotate_x = ndarray([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        verts = verts @ rotate_x.T
        verts[:, 2] -= np.min(verts[:, 2])
        verts /= 500
        return verts, eles

    def __load_ball_mesh(self):
        obj_file_name = Path(self.__ball_mesh_file)
        if not obj_file_name.is_absolute():
            obj_file_name = Path(root_path) / 'asset' / 'mesh' / obj_file_name
        if not obj_file_name.exists():
            raise FileNotFoundError(f'Ball mesh file not found: {obj_file_name}')
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False)
        verts = verts * (self.__ball_radius / 0.03)

        center_idx = int(np.argmin(np.sum(verts ** 2, axis=1)))
        verts[center_idx] = 0.0
        rotation = _rotation_matrix_from_degrees(self.__ball_rotation_degrees)
        verts = verts @ rotation.T
        verts += self.__ball_center
        return verts, eles.astype(int)

    def __create_ball_support_mesh(self):
        sx, sy, sz = self.__ball_support_size
        cx, cy = self.__ball_center[:2]
        z1 = self.__ball_support_top_z
        z0 = z1 - sz
        x0, x1 = cx - 0.5 * sx, cx + 0.5 * sx
        y0, y1 = cy - 0.5 * sy, cy + 0.5 * sy
        verts = ndarray([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])
        eles = ndarray([
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]).astype(int)
        return verts, eles

    def __load_element_modulus(self, element_num, youngs_modulus):
        stiffness_file_name = Path(root_path) / 'asset' / 'mesh' / 'gator_dec_mu.npy'
        if stiffness_file_name.exists():
            print(f'Loading stiffness values from {stiffness_file_name}...')
            element_modulus = np.load(stiffness_file_name).ravel()
            if element_modulus.shape[0] != element_num:
                raise ValueError('Stiffness file length does not match number of elements.')
            element_modulus[element_modulus > 1e10] = 1e6
            print_info(f'Using non-uniform stiffness from {stiffness_file_name}.')
            return element_modulus.astype(np.float64)

        print_info(f'Stiffness file {stiffness_file_name} not found. Using uniform stiffness.')
        return np.full(element_num, youngs_modulus, dtype=np.float64)

    def __apply_body_stiffness_scale(self, element_modulus, options):
        self.__body_stiffness_scale = float(options.get('body_stiffness_scale', 1.0))
        if self.__body_stiffness_scale <= 0:
            raise ValueError('body_stiffness_scale must be positive.')
        if np.isclose(self.__body_stiffness_scale, 1.0):
            return np.copy(element_modulus).astype(np.float64)

        element_modulus = np.copy(element_modulus).astype(np.float64)
        body_elements = np.setdiff1d(
            np.arange(element_modulus.size), self.__sword_elements, assume_unique=False
        )
        element_modulus[body_elements] *= self.__body_stiffness_scale
        return element_modulus

    def __apply_sword_stiffness(self, element_modulus, options):
        element_modulus = np.copy(element_modulus).astype(np.float64)
        if 'sword_stiffness' in options:
            sword_stiffness = float(options['sword_stiffness'])
            if sword_stiffness <= 0:
                raise ValueError('sword_stiffness must be positive.')
            element_modulus[self.__sword_elements] = sword_stiffness
            self.__sword_stiffness = sword_stiffness
        else:
            self.__sword_stiffness = float(np.mean(element_modulus[self.__sword_elements]))
        return element_modulus

    def __print_material_stats(self):
        element_modulus = self.__element_modulus_array
        sword_modulus = element_modulus[self.__sword_elements]
        print(
            'Material stiffness: '
            f'all[min={np.min(element_modulus):.3e}, max={np.max(element_modulus):.3e}, '
            f'mean={np.mean(element_modulus):.3e}], '
            f'sword[min={np.min(sword_modulus):.3e}, max={np.max(sword_modulus):.3e}, '
            f'mean={np.mean(sword_modulus):.3e}, elements={self.__sword_elements.size}]'
        )

    def __build_pd_neohookean_params(self, element_modulus, poissons_ratio):
        per_ele_mu = element_modulus / (2 * (1 + poissons_ratio))
        per_ele_la = element_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        nh_params = []
        for mu, la in zip(per_ele_mu, per_ele_la):
            nh_params.append(float(mu))
            nh_params.append(float(la))
        return nh_params

    def __add_region_grouped_pd_neohookean(self, deformable, element_modulus, poissons_ratio):
        self.__pd_energy_groups = {}
        self.__sword_pd_energy_idx = None
        energy_idx = 0

        def add_group(name, indices):
            nonlocal energy_idx
            indices = ndarray(indices).astype(int)
            if indices.size == 0:
                return
            group_E = float(np.mean(element_modulus[indices]))
            mu = group_E / (2 * (1 + poissons_ratio))
            la = group_E * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            deformable.AddPdEnergy('pd_neohookean', [float(mu), float(la)], indices.tolist())
            self.__pd_energy_groups[name] = {
                'energy_idx': int(energy_idx),
                'element_count': int(indices.size),
                'youngs_modulus': group_E,
            }
            if name == 'sword':
                self.__sword_pd_energy_idx = int(energy_idx)
            energy_idx += 1

        body_elements = np.setdiff1d(
            self.__gatorman_element_indices, self.__sword_elements, assume_unique=False
        )
        add_group('body', body_elements)
        add_group('sword', self.__sword_elements)
        add_group('ball', self.__ball_element_indices)
        add_group('ball_support', self.__ball_support_element_indices)
        print(f'PD energy mode: region_grouped, groups={self.__pd_energy_groups}')

    def __add_region_grouped_corotated_volume(self, deformable, element_modulus, poissons_ratio):
        self.__pd_energy_groups = {}
        self.__sword_pd_energy_idx = None
        energy_idx = 0

        def add_group(name, indices):
            nonlocal energy_idx
            indices = ndarray(indices).astype(int)
            if indices.size == 0:
                return
            group_E = float(np.mean(element_modulus[indices]))
            mu = group_E / (2 * (1 + poissons_ratio))
            la = group_E * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            deformable.AddPdEnergy('corotated', [float(2 * mu)], indices.tolist())
            corotated_idx = int(energy_idx)
            energy_idx += 1
            deformable.AddPdEnergy('volume', [float(la)], indices.tolist())
            volume_idx = int(energy_idx)
            energy_idx += 1
            self.__pd_energy_groups[name] = {
                'energy_idx': int(corotated_idx),
                'corotated_idx': int(corotated_idx),
                'volume_idx': int(volume_idx),
                'element_count': int(indices.size),
                'youngs_modulus': group_E,
            }
            if name == 'sword':
                self.__sword_pd_energy_idx = int(corotated_idx)

        body_elements = np.setdiff1d(
            self.__gatorman_element_indices, self.__sword_elements, assume_unique=False
        )
        add_group('body', body_elements)
        add_group('sword', self.__sword_elements)
        add_group('ball', self.__ball_element_indices)
        add_group('ball_support', self.__ball_support_element_indices)
        print(f'PD energy mode: region_grouped_corotated_volume, groups={self.__pd_energy_groups}')

    def __detect_regions(self, verts, eles, element_modulus, options):
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        size = max_corner - min_corner

        base_tol = self.__base_node_tol
        drive_min_z_ratio = float(options.get('drive_min_z_ratio', 0.4))
        if not 0.0 <= drive_min_z_ratio <= 1.0:
            raise ValueError('drive_min_z_ratio must be in [0, 1].')

        self.__base_nodes = np.flatnonzero(verts[:, 2] - min_corner[2] < base_tol).astype(int)
        self.__sword_elements = self.__detect_sword_elements(verts, eles, element_modulus).astype(int)
        self.__sword_vertices = np.unique(eles[self.__sword_elements].ravel()).astype(int)

        sword_points = verts[self.__sword_vertices]
        sword_min = np.min(sword_points, axis=0)
        sword_max = np.max(sword_points, axis=0)
        sword_x_cutoff = sword_min[0] + 0.8 * (sword_max[0] - sword_min[0])
        self.__sword_root_nodes = self.__sword_vertices[
            sword_points[:, 0] >= sword_x_cutoff
        ].astype(int)

        drive_z_cutoff = min_corner[2] + drive_min_z_ratio * size[2]
        upper_body_nodes = np.flatnonzero(
            (verts[:, 2] > drive_z_cutoff)
            & (max_corner[2] - verts[:, 2] >= base_tol)
        ).astype(int)
        self.__drive_nodes = np.setdiff1d(
            upper_body_nodes, self.__sword_vertices, assume_unique=False
        ).astype(int)

        self.__validate_region_counts(verts.shape[0], eles.shape[0])

    def __detect_contact_surfaces(self, verts, eles):
        gatorman_surface_faces = _extract_boundary_faces(eles, verts)
        sword_vertex_set = set(self.__sword_vertices.tolist())
        self.__sword_surface_faces = ndarray([
            face for face in gatorman_surface_faces
            if all(int(v) in sword_vertex_set for v in face)
        ]).astype(int)
        self.__sword_surface_vertices = np.unique(self.__sword_surface_faces.ravel()).astype(int)
        if self.__sword_surface_faces.size == 0:
            raise ValueError('Sword surface face detection produced an empty set.')

    def __add_mesh_contact(self, deformable):
        self.__add_mesh_contact_pair(
            deformable,
            'sword-ball contact',
            self.__sword_surface_faces,
            self.__ball_surface_faces,
            self.__contact_radius,
            self.__mesh_contact_kn,
            self.__mesh_contact_kf,
            self.__mesh_contact_mu,
        )
        if self.__use_ball_support:
            self.__add_mesh_contact_pair(
                deformable,
                'ball-support contact',
                self.__ball_surface_faces,
                self.__ball_support_surface_faces,
                self.__ball_support_contact_radius,
                self.__ball_support_contact_kn,
                self.__ball_support_contact_kf,
                self.__ball_support_contact_mu,
            )

    def __add_mesh_contact_pair(self, deformable, name, faces_a, faces_b, radius, kn, kf, mu):
        mesh_params = [
            radius,
            kn,
            kn,
            kf,
            mu,
            float(faces_a.shape[0]),
            float(faces_b.shape[0]),
        ]
        mesh_params.extend(ndarray(faces_a).ravel().astype(float).tolist())
        mesh_params.extend(ndarray(faces_b).ravel().astype(float).tolist())
        deformable.AddStateForce('mesh_contact', mesh_params)
        print(
            f'Mesh contact ({name}): '
            f'radius={radius:.3e}, kn={kn:.3e}, '
            f'kf={kf:.3e}, mu={mu:.3e}, '
            f'faces_a={faces_a.shape[0]}, faces_b={faces_b.shape[0]}'
        )

    def __add_mesh_boundary_contact(self, deformable, options):
        faces_left = self.__sword_surface_faces
        faces_right = self.__ball_surface_faces
        split_index = int(self.__ball_vertex_indices[0])
        boundary_params = [
            self.__contact_radius,
            float(split_index),
            float(faces_left.shape[0]),
            float(faces_right.shape[0]),
        ]
        boundary_params.extend(ndarray(faces_left).ravel().astype(float).tolist())
        boundary_params.extend(ndarray(faces_right).ravel().astype(float).tolist())

        candidate_mode = str(options.get('mesh_boundary_candidate_mode', 'both'))
        if candidate_mode == 'ball':
            boundary_indices = self.__ball_surface_vertices
        elif candidate_mode == 'sword':
            boundary_indices = self.__sword_surface_vertices
        elif candidate_mode == 'both':
            boundary_indices = np.unique(np.concatenate([
                self.__sword_surface_vertices,
                self.__ball_surface_vertices,
            ])).astype(int)
        else:
            raise ValueError('mesh_boundary_candidate_mode must be "ball", "sword", or "both".')

        candidate_stride = int(options.get('mesh_boundary_candidate_stride', 1))
        if candidate_stride <= 0:
            raise ValueError('mesh_boundary_candidate_stride must be positive.')
        boundary_indices = ndarray(boundary_indices).astype(int)
        if candidate_stride > 1:
            boundary_indices = boundary_indices[::candidate_stride]
        if boundary_indices.size == 0:
            raise ValueError('mesh_boundary contact produced no candidate vertices.')

        deformable.SetFrictionalBoundary('mesh', boundary_params, boundary_indices.tolist())
        print(
            'Mesh boundary contact (sword-ball): '
            f'radius={self.__contact_radius:.3e}, '
            f'split_index={split_index}, '
            f'candidate_mode={candidate_mode}, '
            f'candidate_stride={candidate_stride}, '
            f'candidates={boundary_indices.size}, '
            f'sword_faces={faces_left.shape[0]}, ball_faces={faces_right.shape[0]}'
        )

    def __add_ball_ground(self, deformable):
        deformable.SetFrictionalBoundary(
            'planar',
            [0.0, 0.0, 1.0, -self.__ball_ground_z],
            self.__ball_surface_vertices.astype(int).tolist()
        )
        print(
            'Ball ground boundary: '
            f'z={self.__ball_ground_z:.3e}, nodes={self.__ball_surface_vertices.size}'
        )

    def __detect_sword_elements(self, verts, eles, element_modulus):
        min_stiffness = float(np.min(element_modulus))
        max_stiffness = float(np.max(element_modulus))
        if max_stiffness > min_stiffness * 1.01:
            return np.flatnonzero(np.isclose(element_modulus, max_stiffness))

        centers = np.mean(verts[eles], axis=1)
        sword_mask = (
            ((centers[:, 1] < -0.0136) & (centers[:, 0] < -0.0085))
            | (
                (centers[:, 1] >= -0.0136)
                & (centers[:, 1] < 0.0265)
                & (centers[:, 0] < -0.022)
                & (centers[:, 2] > 0.0285)
            )
        )
        return np.flatnonzero(sword_mask)

    def __validate_region_counts(self, vertex_num, element_num):
        checks = [
            ('base_nodes', self.__base_nodes.size, vertex_num),
            ('drive_nodes', self.__drive_nodes.size, vertex_num),
            ('sword_root_nodes', self.__sword_root_nodes.size, vertex_num),
            ('sword_elements', self.__sword_elements.size, element_num),
        ]
        for name, count, total in checks:
            if count == 0:
                raise ValueError(f'{name} detection produced an empty set.')
            if count >= total:
                raise ValueError(f'{name} detection covered the whole mesh.')

    def material_stiffness_differential(self, young_modulus, poissons_ratio):
        return np.zeros((self._deformable.NumOfPdElementEnergies(), 0))

    def sword_stiffness_gradient(self, dl_dmat_w):
        if self.__pd_energy_mode not in ('region_grouped', 'region_grouped_corotated_volume'):
            raise ValueError('sword_stiffness_gradient requires a grouped pd_energy_mode.')
        if self.__sword_pd_energy_idx is None:
            raise ValueError('No sword PD energy group was registered.')
        dl_dmat_w = ndarray(dl_dmat_w).ravel()
        sword_group = self.__pd_energy_groups.get('sword', {})
        if self.__pd_energy_mode == 'region_grouped_corotated_volume':
            corotated_idx = int(sword_group['corotated_idx'])
            volume_idx = int(sword_group['volume_idx'])
            if max(corotated_idx, volume_idx) >= dl_dmat_w.size:
                raise ValueError('dl_dmat_w does not contain the sword PD energy indices.')
            nu = self._poissons_ratio
            return float(
                dl_dmat_w[corotated_idx] / (1 + nu)
                + dl_dmat_w[volume_idx] * nu / ((1 + nu) * (1 - 2 * nu))
            )
        if self.__sword_pd_energy_idx >= dl_dmat_w.size:
            raise ValueError('dl_dmat_w does not contain the sword PD energy index.')
        return float(dl_dmat_w[self.__sword_pd_energy_idx] / (1 + self._poissons_ratio))

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dofs

    def mesh(self):
        return self.__mesh

    def include_ball(self):
        return self.__include_ball

    def gatorman_vertices(self):
        return np.copy(self.__gatorman_vertices)

    def gatorman_elements(self):
        return np.copy(self.__gatorman_elements)

    def get_element_modulus_array(self):
        return np.copy(self.__element_modulus_array)

    def sword_stiffness(self):
        return self.__sword_stiffness

    def loss_type(self):
        return self.__loss_type

    def pd_energy_mode(self):
        return self.__pd_energy_mode

    def pd_energy_groups(self):
        return dict(self.__pd_energy_groups)

    def min_x_nodes(self):
        return self.__min_x_nodes

    def max_x_nodes(self):
        return self.__max_x_nodes

    def min_y_nodes(self):
        return self.__min_y_nodes

    def base_nodes(self):
        return np.copy(self.__base_nodes)

    def head_anchor_node(self):
        return self.__head_anchor_node

    def head_anchor_position(self):
        return np.copy(self.__head_anchor_position)

    def head_anchor_rank(self):
        return self.__head_anchor_rank

    def drive_nodes(self):
        return np.copy(self.__drive_nodes)

    def sword_root_nodes(self):
        return np.copy(self.__sword_root_nodes)

    def sword_elements(self):
        return np.copy(self.__sword_elements)

    def sword_vertices(self):
        return np.copy(self.__sword_vertices)

    def region_summary(self):
        return {
            'base_nodes': int(self.__base_nodes.size),
            'head_anchor_node': int(self.__head_anchor_node),
            'head_anchor_rank': int(self.__head_anchor_rank),
            'drive_nodes': int(self.__drive_nodes.size),
            'sword_root_nodes': int(self.__sword_root_nodes.size),
            'sword_elements': int(self.__sword_elements.size),
            'sword_vertices': int(self.__sword_vertices.size),
            'sword_surface_faces': int(self.__sword_surface_faces.shape[0]),
            'ball_vertices': int(self.__ball_vertex_indices.size),
            'ball_elements': int(self.__ball_element_indices.size),
            'ball_surface_faces': int(self.__ball_surface_faces.shape[0]),
            'ball_support_vertices': int(self.__ball_support_vertex_indices.size),
            'ball_support_elements': int(self.__ball_support_element_indices.size),
            'ball_support_surface_faces': int(self.__ball_support_surface_faces.shape[0]),
            'ball_ground_nodes': int(
                self.__ball_surface_vertices.size if self.__use_ball_ground else 0
            ),
            'contact_model': self.__contact_model,
            'pd_energy_mode': self.__pd_energy_mode,
        }

    def ball_vertices(self):
        return np.copy(self.__ball_vertex_indices)

    def ball_elements(self):
        return np.copy(self.__ball_element_indices)

    def ball_radius(self):
        return self.__ball_radius

    def ball_mesh_file(self):
        return self.__ball_mesh_file

    def ball_rotation_degrees(self):
        return np.copy(self.__ball_rotation_degrees)

    def use_ball_support(self):
        return self.__use_ball_support

    def ball_support_size(self):
        return np.copy(self.__ball_support_size)

    def ball_support_elements(self):
        return np.copy(self.__ball_support_element_indices)

    def use_ball_ground(self):
        return self.__use_ball_ground

    def ball_ground_z(self):
        return self.__ball_ground_z

    def contact_model(self):
        return self.__contact_model

    def contact_radius(self):
        return self.__contact_radius

    def sword_surface_faces(self):
        return np.copy(self.__sword_surface_faces)

    def sword_surface_vertices(self):
        return np.copy(self.__sword_surface_vertices)

    def ball_surface_faces(self):
        return np.copy(self.__ball_surface_faces)

    def ball_surface_vertices(self):
        return np.copy(self.__ball_surface_vertices)

    def initial_ball_center(self):
        return np.copy(self.__ball_center)

    def ball_center(self, q=None):
        if not self.__include_ball:
            raise ValueError('ball_center is only available when include_ball=True.')
        q = self.default_init_position() if q is None else ndarray(q)
        vertices = q.reshape((-1, 3))[self.__ball_vertex_indices]
        return np.mean(vertices, axis=0)

    def ball_lowest_point(self, q=None):
        if not self.__include_ball:
            raise ValueError('ball_lowest_point is only available when include_ball=True.')
        q = self.default_init_position() if q is None else ndarray(q)
        vertices = q.reshape((-1, 3))[self.__ball_vertex_indices]
        return vertices[int(np.argmin(vertices[:, 2]))]

    def _create_filtered_mesh_by_elements(self, q, element_indices, temp_file):
        vertices = ndarray(q).reshape((self.__mesh.NumOfVertices(), 3))
        elements = []
        for elem_idx in element_indices:
            elements.append(ndarray(self.__mesh.py_element(int(elem_idx))).astype(int))
        if len(elements) == 0:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
        generate_tet_mesh(vertices, ndarray(elements).astype(int), temp_file)
        return True

    def __add_node_markers(self, renderer, vertices, node_indices, color, radius):
        for idx in ndarray(node_indices).astype(int)[::self.__force_marker_stride]:
            renderer.add_shape_mesh(
                {'name': 'sphere', 'center': vertices[int(idx)], 'radius': radius},
                transforms=[('s', 2)],
                color=color,
            )

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': (-0.3, -0.8, 0.2),
            'camera_lookat': (0.4, 0, .15),
            'fov': 50,
            'resolution': (1280, 720),
        }
        renderer = PbrtRenderer(options)
        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)

        q = ndarray([ndarray(mesh.py_vertex(i)) for i in range(mesh.NumOfVertices())]).ravel()
        if self.__visualize_material_stiffness:
            body_elements = np.setdiff1d(
                self.__gatorman_element_indices, self.__sword_elements, assume_unique=False
            )
            material_parts = [
                ('body', body_elements, '0096c7'),
                ('sword', self.__sword_elements, 'ff4d4d'),
            ]
            for part_name, element_indices, color in material_parts:
                part_mesh_file = f'{mesh_file}_{part_name}.mesh'
                if self._create_filtered_mesh_by_elements(q, element_indices, part_mesh_file):
                    part_mesh = TetMesh3d()
                    part_mesh.Initialize(part_mesh_file)
                    renderer.add_tri_mesh(part_mesh, color=color, transforms=[('s', 2)], render_tet_edge=False)
                    os.remove(part_mesh_file)
        else:
            renderer.add_tri_mesh(mesh, color='0096c7', transforms=[('s', 2)], render_tet_edge=False)

        renderer.add_tri_mesh(
            Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7',
            transforms=[('s', 8), ('r', (-0.7, 0, 0, 1))],
        )

        if self.__include_ball:
            ball_mesh_file = f'{mesh_file}_ball.mesh'
            if self._create_filtered_mesh_by_elements(q, self.__ball_element_indices, ball_mesh_file):
                ball_mesh = TetMesh3d()
                ball_mesh.Initialize(ball_mesh_file)
                renderer.add_tri_mesh(
                    ball_mesh, color='ffb000', transforms=[('s', 2)], render_tet_edge=False
                )
                os.remove(ball_mesh_file)
            support_mesh_file = f'{mesh_file}_ball_support.mesh'
            if self._create_filtered_mesh_by_elements(
                q, self.__ball_support_element_indices, support_mesh_file
            ):
                support_mesh = TetMesh3d()
                support_mesh.Initialize(support_mesh_file)
                renderer.add_tri_mesh(
                    support_mesh, color='555555', transforms=[('s', 2)], render_tet_edge=False
                )
                os.remove(support_mesh_file)

        if self.__visualize_force_nodes:
            vertices = q.reshape((-1, 3))
            self.__add_node_markers(
                renderer, vertices, self.__drive_nodes, 'ffff00', self.__force_marker_radius
            )
        if self.__visualize_head_anchor:
            vertices = q.reshape((-1, 3))
            renderer.add_shape_mesh(
                {
                    'name': 'sphere',
                    'center': vertices[self.__head_anchor_node],
                    'radius': self.__head_anchor_marker_radius,
                },
                transforms=[('s', 2)],
                color='00ff00',
            )
        renderer.render()

    def _loss_and_grad(self, q, v):
        if self.__loss_type == 'ball_xy_distance':
            ball_vertices = self.__ball_vertex_indices
            final_xy = self.ball_center(q)[:2]
            dxy = final_xy - self.__initial_ball_center_xy
            loss = -0.5 * float(dxy.dot(dxy))
            grad_q = np.zeros_like(q)
            grad_v = np.zeros_like(v)
            grad_q_vertices = grad_q.reshape((-1, 3))
            grad_q_vertices[ball_vertices, :2] = -dxy / float(ball_vertices.size)
            return loss, grad_q, grad_v

        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)
