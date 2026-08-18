"""
ArmadilloEnv
============
Armadillo environment with per-element heterogeneous material support.

Per-element stiffness is split into three levels by element centroid
z-height (bottom stiff / middle / top soft), encoded directly into the
PD energies so it applies to both CPU and GPU solvers.
"""
import os
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, read_tetgen_file
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer


class ArmadilloEnv(EnvBase):
    """
    Armadillo environment with per-element heterogeneous Young's modulus.

    The per-element stiffness is passed to AddPdEnergy as an array of length
    element_num, which encodes it into the A matrix for both CPU and GPU solvers.
    """

    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options.get('youngs_modulus', 5e5)
        poissons_ratio = options.get('poissons_ratio', 0.45)
        state_force_parameters = options.get('state_force_parameters', [0.0, 0.0, -9.81])
        density = 1e3

        # ── Load mesh ──────────────────────────────────────────────────────────
        ele_file_name  = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.ele'
        node_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.node'
        verts, eles = read_tetgen_file(node_file_name, ele_file_name)

        # Rotate: +x by 90° then z by 180°, shift min_z=0, scale by 1/1000
        R = ndarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        verts = verts @ R.T
        R = ndarray([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        verts = verts @ R.T
        verts[:, 2] -= np.min(verts, axis=0)[2]
        verts /= 1000

        tmp_bin = '.tmp_hetero.bin'
        generate_tet_mesh(verts, eles, tmp_bin)
        mesh = TetMesh3d()
        mesh.Initialize(tmp_bin)
        self.__mesh = mesh

        element_num = mesh.NumOfElements()
        vert_num    = mesh.NumOfVertices()
        all_verts   = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        max_corner  = np.max(all_verts, axis=0)
        min_corner  = np.min(all_verts, axis=0)

        # ── Per-element Young's modulus array (3-level z-split) ────────────────
        self.soft_ele = []
        self.mid_ele = []
        self.stiff_ele = []
        per_element_E = self._build_element_modulus(mesh, element_num)
        self.__element_modulus_array = per_element_E
        self.het = per_element_E.max() > per_element_E.min() * 1.01

        nu = poissons_ratio
        per_element_mu = per_element_E / (2 * (1 + nu))
        per_element_la = per_element_E * nu / ((1 + nu) * (1 - 2 * nu))

        avg_E = float(np.mean(per_element_E))

        # Print summary
        print(f"[ArmadilloEnv] element_num={element_num}")
        print(f"  E  min={per_element_E.min():.2e}  max={per_element_E.max():.2e}  "
              f"mean={per_element_E.mean():.2e}  range={per_element_E.max()/per_element_E.min():.1f}x")

        # ── Deformable: 'none' material + per-element PD energies ─────────────
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin, density, 'none', avg_E, nu)
        os.remove(tmp_bin)

        # material option: 'pd_corotated' (default, corotated + volume) or
        # 'pd_neohookean' (single NH PD energy with per-element mu/lambda).
        self.__material = options.get('material', 'pd_corotated')
        if self.__material == 'pd_neohookean':
            # Per-element [mu_1, la_1, mu_2, la_2, ..., mu_n, la_n] layout.
            nh_params = []
            for i in range(element_num):
                nh_params.append(float(per_element_mu[i]))
                nh_params.append(float(per_element_la[i]))
            deformable.AddPdEnergy('pd_neohookean', nh_params, [])
            print(f"  PD energy: pd_neohookean (per-element mu/lambda)")
        else:
            # per-element corotated: params of length element_num → element_stiffness(i)
            deformable.AddPdEnergy('corotated', list(2.0 * per_element_mu), [])
            deformable.AddPdEnergy('volume',    list(per_element_la),        [])

        # ── Boundary conditions ────────────────────────────────────────────────
        min_x = min_corner[0]
        max_x = max_corner[0]
        min_z = min_corner[2]
        max_z = max_corner[2]
        dirichlet_dofs = []
        self.__min_x_nodes = []
        self.__max_x_nodes = []

        for i in range(vert_num):
            vx, vy, vz = all_verts[i]
            if vx - min_x < 1e-3:
                self.__min_x_nodes.append(i)
            if max_x - vx < 1e-3:
                self.__max_x_nodes.append(i)
            if vz - min_z < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i,     vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs += [3 * i, 3 * i + 1, 3 * i + 2]
            if max_z - vz < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs += [3 * i + 2]

        self.__dirichlet_dofs = dirichlet_dofs
        deformable.AddStateForce('gravity', state_force_parameters)

        # ── Initial state ──────────────────────────────────────────────────────
        center = (max_corner + min_corner) / 2
        q0 = np.copy(all_verts)
        theta = float(options.get('init_rotate_angle', 0))
        for i in range(vert_num):
            vi = all_verts[i]
            th = (vi[2] - min_z) / (max_z - min_z) * theta
            c, s = np.cos(th), np.sin(th)
            R = ndarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            q0[i] = R @ (vi - center) + center

        dofs     = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = q0.ravel()
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # ── Store members ──────────────────────────────────────────────────────
        self._deformable             = deformable
        self._q0                     = q0
        self._v0                     = v0
        self._f_ext                  = f_ext
        self._youngs_modulus         = avg_E
        self._poissons_ratio         = nu
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss          = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__spp = options.get('spp', 4)

    # ── Per-element modulus builder ────────────────────────────────────────────

    def _build_element_modulus(self, mesh, element_num):
        """Return a numpy array of per-element Young's modulus, 3-level z-split."""
        per_element_E = np.zeros(element_num, dtype=float)
        c1 = c2 = c3 = 0
        for i in range(element_num):
            ele_verts = ndarray(mesh.py_element(i))
            center = np.zeros(3)
            for vi in ele_verts:
                center += ndarray(mesh.py_vertex(int(vi)))
            center /= len(ele_verts)
            z = center[2]
            if z < 0.05:
                per_element_E[i] = 1e6
                c1 += 1
                self.stiff_ele.append(i)
            elif z < 0.10:
                per_element_E[i] = 5e5
                c2 += 1
                self.mid_ele.append(i)
            else:
                per_element_E[i] = 2.5e5
                c3 += 1
                self.soft_ele.append(i)
        print(f"  Element z-split: z<0.05:{c1} (E=1e6), 0.05≤z<0.10:{c2} (E=5e5), z≥0.10:{c3} (E=2.5e5)")
        return per_element_E

    # ── Public interface ───────────────────────────────────────────────────────

    def min_x_nodes(self):
        return self.__min_x_nodes

    def max_x_nodes(self):
        return self.__max_x_nodes

    def get_element_modulus_array(self):
        return self.__element_modulus_array

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # Jacobian shape must match NumOfPdElementEnergies().
        num_pd = self._deformable.NumOfPdElementEnergies()
        return np.zeros((num_pd, 2))

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dofs

    def _loss_and_grad(self, q, v):
        # Random linear loss normalised by sqrt(N) so initial |dl/dq|, |dl/dv|
        # stay O(1); prevents adjoint blow-up under 30-frame rollouts.
        scale = 1.0 / np.sqrt(float(len(q)))
        gq = self.__loss_q_grad * scale
        gv = self.__loss_v_grad * scale
        loss = q.dot(gq) + v.dot(gv)
        return loss, np.copy(gq), np.copy(gv)

    def _create_filtered_mesh_by_elements(self, q, element_indices, temp_file):
        """
        Create a temporary mesh file containing only the specified elements.
        q: The deformed vertex positions (flattened array)
        element_indices: Iterable of element indices
        temp_file: Output filename for the filtered mesh
        """
        from py_diff_pd.common.tet_mesh import generate_tet_mesh

        # Reshape q to get vertex positions
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
            stiff_eles = self.stiff_ele
            mid_eles = self.mid_ele
            soft_eles = self.soft_ele
            stiff_mesh_file = f'{mesh_file}_stiff.mesh'
            mid_mesh_file = f'{mesh_file}_mid.mesh'
            soft_mesh_file = f'{mesh_file}_soft.mesh'
            self._create_filtered_mesh_by_elements(q, stiff_eles, stiff_mesh_file)
            self._create_filtered_mesh_by_elements(q, mid_eles, mid_mesh_file)
            self._create_filtered_mesh_by_elements(q, soft_eles, soft_mesh_file)
            stiff_mesh = TetMesh3d()
            mid_mesh = TetMesh3d()
            soft_mesh = TetMesh3d()
            stiff_mesh.Initialize(stiff_mesh_file)
            mid_mesh.Initialize(mid_mesh_file)
            soft_mesh.Initialize(soft_mesh_file)
            renderer.add_tri_mesh(stiff_mesh, color='ff4d4d',
                transforms=[('s', 2)], render_tet_edge=True)
            renderer.add_tri_mesh(mid_mesh, color='9932cc',
                transforms=[('s', 2)], render_tet_edge=True)
            renderer.add_tri_mesh(soft_mesh, color='0096c7',
                transforms=[('s', 2)], render_tet_edge=True)
            os.remove(stiff_mesh_file)
            os.remove(mid_mesh_file)
            os.remove(soft_mesh_file)

        renderer.add_tri_mesh(
            Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])
        renderer.render()
