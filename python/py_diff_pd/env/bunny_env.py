from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tetrahedralize, fix_tet_faces
from py_diff_pd.common.hex_mesh import get_contact_vertex as get_hex_contact_vertex
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class BunnyEnv(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.49
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        mesh_type = options['mesh_type'] if 'mesh_type' in options else 'hex'
        assert mesh_type in ['hex', 'tet']
        het_bool = options['het'] if 'het' in options else False

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        bunny_size = 0.1    
        tmp_bin_file_name = '.tmp.bin'
        if mesh_type == 'hex':
            bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight.bin'
            mesh = HexMesh3d()
            mesh.Initialize(str(bin_file_name))
            deformable = HexDeformable()
        elif mesh_type == 'tet':
            obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight_simplified2.obj'
            verts, eles = tetrahedralize(obj_file_name)
            generate_tet_mesh(verts, eles, tmp_bin_file_name)
            mesh = TetMesh3d()
            mesh.Initialize(str(tmp_bin_file_name))
            deformable = TetDeformable()
        else:
            raise NotImplementedError
        # Rescale the mesh.
        mesh.Scale(bunny_size)
        mesh.SaveToFile(tmp_bin_file_name)
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # Elasticity.
        if not het_bool:
            deformable.AddPdEnergy('corotated', [2 * mu,], [])
            deformable.AddPdEnergy('volume', [la,], [])
        else:
            #  Heterogeneous stiffness: bottom half is softer, top half is stiffer,
            #  split symmetrically around the base Young's modulus by contrast_factor:
            #    E_stiff = E * sqrt(cf), E_soft = E / sqrt(cf)  (so E_stiff / E_soft == cf).
            cf = float(options.get('contrast_factor', 100.0))
            stiff_scale = float(np.sqrt(cf))
            soft_scale  = float(1.0 / np.sqrt(cf))
            self._stiff_scale = stiff_scale
            self._soft_scale  = soft_scale
            element_num = mesh.NumOfElements()
            stiff_ele = []
            soft_ele = []
            mid = np.mean(ndarray(mesh.py_vertices()).reshape((-1, 3)), axis=0)
            for i in range(element_num):
                # split element according to the y coordinate of its center.
                ele = ndarray(mesh.py_element(i))
                v0 = ndarray(mesh.py_vertex(int(ele[0])))
                v1 = ndarray(mesh.py_vertex(int(ele[1])))
                v2 = ndarray(mesh.py_vertex(int(ele[2])))
                v3 = ndarray(mesh.py_vertex(int(ele[3])))
                center = (v0 + v1 + v2 + v3) / 4
                if center[1] < mid[1]:
                    soft_ele.append(i)
                else:
                    stiff_ele.append(i)
            E_stiff = youngs_modulus * stiff_scale
            E_soft  = youngs_modulus * soft_scale
            print(f"  Bunny het (contrast_factor={cf}):"
                  f" stiff={len(stiff_ele)} (E={E_stiff:.2e}), "
                  f"soft={len(soft_ele)} (E={E_soft:.2e}),"
                  f" spread={cf:.2e}x")
            deformable.AddPdEnergy('corotated', [stiff_scale * 2 * mu,], stiff_ele)
            deformable.AddPdEnergy('volume',    [stiff_scale * la,],     stiff_ele)
            deformable.AddPdEnergy('corotated', [soft_scale  * 2 * mu,], soft_ele)
            deformable.AddPdEnergy('volume',    [soft_scale  * la,],     soft_ele)
        
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)
        # Collisions.
        if mesh_type == 'hex':
            friction_node_idx = get_hex_contact_vertex(mesh)
        elif mesh_type == 'tet':
            friction_node_idx = get_tet_contact_vertex(mesh, threshold=np.pi * 1.2)
        else:
            raise NotImplementedError

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Initial states.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self._target_com = ndarray(options['target_com']) if 'target_com' in options else ndarray([0.15, 0.15, 0.15])
        self._bunny_size = bunny_size
        self._mesh_type = mesh_type

        self.__spp = options['spp'] if 'spp' in options else 4
        self._het = het_bool
        
        self.soft_ele = soft_ele if het_bool else None
        self.stiff_ele = stiff_ele if het_bool else None

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        if self._het:
            # Energy order in the heterogeneous setup:
            # [ stiff corotated, stiff volume, soft corotated, soft volume ]
            # Scales match AddPdEnergy calls in __init__ (contrast-driven).
            s_stiff = getattr(self, '_stiff_scale', 1.0)
            s_soft  = getattr(self, '_soft_scale',  0.01)
            jac_total = np.zeros((4, 2))
            jac_total[0] = 2 * s_stiff * jac[1]
            jac_total[1] =     s_stiff * jac[0]
            jac_total[2] = 2 * s_soft  * jac[1]
            jac_total[3] =     s_soft  * jac[0]
        else:
            jac_total = np.zeros((2, 2))
            jac_total[0] = 2 * jac[1]
            jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.15, -1.75, 0.6),
            'camera_lookat': (0, .15, .4)
        }
        renderer = PbrtRenderer(options)

        if self._mesh_type == 'hex':
            mesh = HexMesh3d()
            mesh.Initialize(mesh_file)
            vertices = []
            for i in range(mesh.NumOfVertices()):
                vertices.append(mesh.py_vertex(i))
            vertices = ndarray(vertices)

            face_dict = {}
            face_idx = [
                (0, 1, 3, 2),
                (4, 6, 7, 5),
                (0, 4, 5, 1),
                (2, 3, 7, 6),
                (1, 5, 7, 3),
                (0, 2, 6, 4)
            ]
            for e in range(mesh.NumOfElements()):
                fi = ndarray(mesh.py_element(e))
                for f in face_idx:
                    vidx = [int(fi[fij]) for fij in f]
                    vidx_key = tuple(sorted(vidx))
                    if vidx_key in face_dict:
                        del face_dict[vidx_key]
                    else:
                        face_dict[vidx_key] = (vidx, e)
            faces = [val[0] for val in face_dict.values()]
            face_owner = [val[1] for val in face_dict.values()]
            fij = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif self._mesh_type == 'tet':
            mesh = TetMesh3d()
            mesh.Initialize(mesh_file)
            vertices = []
            for i in range(mesh.NumOfVertices()):
                vertices.append(mesh.py_vertex(i))
            vertices = ndarray(vertices)

            face_dict = {}
            for e in range(mesh.NumOfElements()):
                fi = list(mesh.py_element(e))
                element_vert = []
                for vi in fi:
                    element_vert.append(mesh.py_vertex(vi))
                element_vert = ndarray(element_vert)
                face_idx = fix_tet_faces(element_vert)
                for f in face_idx:
                    vidx = [int(fi[fij]) for fij in f]
                    vidx_key = tuple(sorted(vidx))
                    if vidx_key in face_dict:
                        del face_dict[vidx_key]
                    else:
                        face_dict[vidx_key] = (vidx, e)
            faces = [val[0] for val in face_dict.values()]
            face_owner = [val[1] for val in face_dict.values()]
            fij = [(0, 1), (1, 2), (2, 0)]
        else:
            raise NotImplementedError

        scale = 3
        default_color = (0.7, .5, 0.7)
        soft_color = (0.2, 0.55, 0.95)
        stiff_color = (0.95, 0.35, 0.25)
        use_heterogeneous_color = self._het and self.soft_ele is not None and self.stiff_ele is not None
        soft_ele_set = set(self.soft_ele) if use_heterogeneous_color else None
        stiff_ele_set = set(self.stiff_ele) if use_heterogeneous_color else None
        # Draw wireframe of the bunny.
        for face_id, f in enumerate(faces):
            edge_color = default_color
            if use_heterogeneous_color:
                owner = face_owner[face_id]
                if owner in soft_ele_set:
                    edge_color = soft_color
                elif owner in stiff_ele_set:
                    edge_color = stiff_color
            for i, j in fij:
                vi = vertices[f[i]]
                vj = vertices[f[j]]
                # Draw line vi to vj.
                renderer.add_shape_mesh({
                        'name': 'curve',
                        'point': ndarray([vi, (2 * vi + vj) / 3, (vi + 2 * vj) / 3, vj]),
                        'width': 0.001
                    },
                    color=edge_color,
                    transforms=[
                        ('s', scale)
                    ])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        # Add target CoM and mesh CoM.
        renderer.add_shape_mesh({ 'name': 'sphere', 'center': self._target_com, 'radius': 0.0075 },
            transforms=[('s', scale)], color=(0.1, 0.1, 0.9))

        com = np.mean(ndarray(mesh.py_vertices()).reshape((-1, 3)), axis=0)
        renderer.add_shape_mesh({ 'name': 'sphere', 'center': com, 'radius': 0.0075 },
            transforms=[('s', scale) ], color=(0.9, 0.1, 0.1))

        renderer.render()

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        # Compute loss.
        com_diff = com - self._target_com
        loss = 0.5 * com_diff.dot(com_diff) / (self._bunny_size ** 2)
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        for i in range(3):
            grad_q[i::3] = com_diff[i] / vertex_num / (self._bunny_size ** 2)
        grad_v = np.zeros(v.size) / (self._bunny_size ** 2)
        return loss, grad_q, grad_v
