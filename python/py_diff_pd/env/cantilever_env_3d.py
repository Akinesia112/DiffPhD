import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.common.renderer import PbrtRenderer


class CantileverEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh. We use 8 for cantilever_3d.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement'] if 'refinement' in options else 2
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([5.])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        het_stiff_bool = options['heterogeneous_stiffness'] if 'heterogeneous_stiffness' in options else False

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        cell_nums = (4 * refinement, refinement, refinement)
        origin = ndarray([0, 0, 0])
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        dx = 0.08 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for j in range(node_nums[1]):
            for k in range(node_nums[2]):
                node_idx = j * node_nums[2] + k
                vx, vy, vz = mesh.py_vertex(node_idx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

        upper_elements = [] # harder
        lower_elements = [] # softer
        if not het_stiff_bool:
            # Homogeneous stiffness.
            # Elasticity.
            deformable.AddPdEnergy('corotated', [2 * mu,], [])
            deformable.AddPdEnergy('volume', [la,], [])
        else:
            # upper half stiffer, lower half softer.
            # Elasticity.
            def ele_idx(i, j, k):
                return i * cell_nums[1] * cell_nums[2] + j * cell_nums[2] + k
            for i in range(cell_nums[0]):
                for j in range(cell_nums[1]):
                    for k in range(cell_nums[2]):
                        if i > cell_nums[0] // 3 and i <= cell_nums[0] // 3 * 2:
                            upper_elements.append(ele_idx(i, j, k))
                        else:
                            lower_elements.append(ele_idx(i, j, k))
            print(f"[CantileverEnv3d] Heterogeneous stiffness setup:")
            print(f"  Total elements: {mesh.NumOfElements()}")
            print(f"  Upper half (i > {cell_nums[0] // 3} and i <= {cell_nums[0] // 3 * 2}): {len(upper_elements)} elements with 10x stiffness")
            print(f"  Lower half (i < {cell_nums[0] // 3} or i > {cell_nums[0] // 3 * 2}): {len(lower_elements)} elements with 1x stiffness")
            print(f"  Stiffness ratio (upper/lower): 10x")

            deformable.AddPdEnergy('corotated', [20 * mu,], upper_elements)
            deformable.AddPdEnergy('volume', [10 * la,], upper_elements)

            deformable.AddPdEnergy('corotated', [2 * mu,], lower_elements)
            deformable.AddPdEnergy('volume', [la,], lower_elements)

        def to_index(i, j, k):
            return i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
        # Collisions.
        collision_indices = [to_index(cell_nums[0], 0, 0), to_index(cell_nums[0], cell_nums[1], 0)]
        deformable.AddPdEnergy('planar_collision', [1e2, 0.0, 0.0, 1.0, 0.1], collision_indices)
        # Actuation.
        act_indices = []
        for i in range(cell_nums[0]):
            j = 0
            k = 0
            act_indices.append(i * cell_nums[1] * cell_nums[2] + j * cell_nums[2] + k)
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [1.0, 0.0, 0.0], act_indices)

        # Initial state.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        max_x_nodes = []
        min_x_nodes = []
        for i in range(node_nums[1]):
            for j in range(node_nums[2]):
                idx = cell_nums[0] * node_nums[1] * node_nums[2] + i * node_nums[2] + j
                max_x_nodes.append(idx)
                idx_min = i * node_nums[1] * node_nums[2] + j * node_nums[2]
                min_x_nodes.append(idx_min)
        v0 = np.zeros(dofs)
        f_ext = ndarray(np.zeros(dofs)).ravel()

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self._het_stiff_bool = het_stiff_bool
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__node_nums = node_nums
        self.__upper_elements = upper_elements
        self.__lower_elements = lower_elements
        self.__max_x_nodes = max_x_nodes
        self.__min_x_nodes = min_x_nodes

        self.__spp = options['spp'] if 'spp' in options else 4
    def max_x_nodes(self):
        return self.__max_x_nodes
    def min_x_nodes(self):
        return self.__min_x_nodes

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        if not self._het_stiff_bool:
            # Homogeneous: 2 energies (corotated, volume)
            # Weights: [2*mu, la]
            jac_total = np.zeros((2, 2))
            jac_total[0] = 2 * jac[1]  # d(2*mu)/d[E, nu]
            jac_total[1] = jac[0]      # d(la)/d[E, nu]
            return jac_total
        else:
            # Heterogeneous: 4 energies (upper corotated, upper volume, lower corotated, lower volume)
            # Weights: [20*mu, 10*la, 2*mu, la]
            jac_total = np.zeros((4, 2))
            jac_total[0] = 20 * jac[1]  # d(20*mu)/d[E, nu]
            jac_total[1] = 10 * jac[0]  # d(10*la)/d[E, nu]
            jac_total[2] = 2 * jac[1]   # d(2*mu)/d[E, nu]
            jac_total[3] = jac[0]       # d(la)/d[E, nu]
            return jac_total

    def is_dirichlet_dof(self, dof):
        i = dof // (self.__node_nums[1] * self.__node_nums[2])
        return i == 0

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'sample': self.__spp,
            'camera_pos': (2, -4, 2.4),
            'camera_lookat': (0.5, 0.5, 0.7),
            'resolution': (400, 400),
            'light_map': 'white.png'
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        
        if not self._het_stiff_bool:
            # Homogeneous: render with grid texture.
            render_hex_mesh(mesh, file_name=file_name,
                resolution=(400, 400), sample=self.__spp, transforms=[
                    ('t', (-0.16, 0.16, 0.05)),
                    ('s', 6)
                ],
                camera_pos=(2, -2.2, 1.4),
                render_voxel_edge=True,
                mesh_color=[.4, .4, .4])
        else:
            # Heterogeneous: render with different colors for upper and lower halves.
            # split upper and lower bin files.
            from py_diff_pd.common.hex_mesh import filter_hex
            upper_hex_mesh = filter_hex(mesh, self.__upper_elements)
            lower_hex_mesh = filter_hex(mesh, self.__lower_elements)
            renderer.add_hex_mesh(upper_hex_mesh, transforms=[
                    ('t', (-0.16, 0.16, 0.05)),
                    ('s', 6)
                ], render_voxel_edge=True, color=[.8, .3, .3])
            renderer.add_hex_mesh(lower_hex_mesh, transforms=[
                    ('t', (-0.16, 0.16, 0.05)),
                    ('s', 6)
                ], render_voxel_edge=True, color=[.3, .3, .8])
            renderer.render(light_rgb=(2.5, 2.5, 2.5))


    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)