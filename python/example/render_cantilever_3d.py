import sys
sys.path.append('../')

from pathlib import Path
import numpy as np

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.common.hex_mesh import hex2obj, hex2obj_with_textures
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('cantilever_3d')
    env = CantileverEnv3d(seed, folder, { 'refinement': 5, 'spp': 64, 'heterogeneous_stiffness': True })    # 'state_force_parameters': [0, 0, 0],
    deformable = env.deformable()

    # method = 'pd_eigen'
    # opt = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
    #         'use_bfgs': 1, 'bfgs_history_size': 10 }
    method = 'pd_eigen_alg_phd'
    opt = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 8,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5}

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    # a0 = np.random.uniform(size=act_dofs)
    # f0 = np.random.normal(scale=0.1, size=dofs) * 1e-3
    dt = 1e-2
    frame_num = 25
    a0 = np.zeros(act_dofs)
    f0 = [np.zeros(dofs).reshape((-1, 3)) for _ in range(frame_num)]
    # Add a force at max x elements
    for t in range(frame_num):
        f_right = np.array([1, 0, 0]) * t / frame_num
        for i in env.max_x_nodes():
            f0[t][i] = f_right * 80
    f0 = [fi.ravel() for fi in f0]
    
    
    env.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)],
        f0, require_grad=False, vis_folder='groundtruth2')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj_with_textures(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i))

    generate_mesh('groundtruth2', 'groundtruth_mesh')