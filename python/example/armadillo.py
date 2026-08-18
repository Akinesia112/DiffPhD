import sys
sys.path.append('../')

import numpy as np
from pathlib import Path
import pickle

from py_diff_pd.common.common import ndarray, print_info, PrettyTabular
from py_diff_pd.env.armadillo_env import ArmadilloEnv
from py_diff_pd.common.display import export_mp4


def test_armadillo(verbose):
    """Example using the C++ heterogeneous Young's modulus implementation for Armadillo."""
    seed = 42
    folder = Path('armadillo')

    base_options = {
        'youngs_modulus': 5e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, 0],
        'spp': 4,
    }
    # pd_eigen_alg_phd only converges with the pd_neohookean formulation; the
    # other two methods run on the default corotated+volume env so their
    # settings stay untouched by this option.
    env_default = ArmadilloEnv(seed, folder, base_options)
    env_alg_phd = ArmadilloEnv(seed, folder, {**base_options, 'material': 'pd_neohookean'})
    envs = {
        'pd_eigen': env_default,
        'pd_eigen_cuda_mas_pcg': env_default,
        'pd_eigen_alg_phd': env_alg_phd,
    }

    deformable = env_default.deformable()

    # Print heterogeneous material statistics
    modulus_array = env_default.get_element_modulus_array()
    print(f"\n{'='*70}")
    print(f"Material Heterogeneity Analysis")
    print(f"{'='*70}")
    print(f"\nElement-level Young's Modulus:")
    print(f"  Total elements: {len(modulus_array)}")
    print(f"  Min: {modulus_array.min():.2e} Pa")
    print(f"  Max: {modulus_array.max():.2e} Pa")
    print(f"  Mean: {modulus_array.mean():.2e} Pa")
    print(f"  Std: {modulus_array.std():.2e} Pa")
    print(f"  Range: {modulus_array.max() / modulus_array.min():.1f}x")
    print(f"{'='*70}\n")

    # Setup simulation parameters
    thread_cts = [8]
    methods = ('pd_eigen', 'pd_eigen_cuda_mas_pcg', 'pd_eigen_alg_phd')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'use_mas': 1},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5},)

    # Compute initial state
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env_default.default_init_position()
    v0 = env_default.default_init_velocity()
    a0 = np.zeros(act_dofs)
    dt = 3e-2
    frame_num = 30

    # Create forcing pattern: twisting motion
    f0 = [np.zeros(dofs).reshape((-1, 3)) for _ in range(frame_num)]
    for t in range(frame_num):
        f_min_x = ndarray([
            np.cos(-np.pi / 4 + t / frame_num * np.pi / 2),
            np.sin(-np.pi / 4 + t / frame_num * np.pi / 2),
            0
        ]) * t / frame_num
        for i in env_default.min_x_nodes():
            f0[t][i] = f_min_x
        for i in env_default.max_x_nodes():
            f0[t][i] = -f_min_x
    f0 = [fi.ravel() for fi in f0]

    # Visualization.
    if verbose:
        for method, opt in zip(methods, opts):
            envs[method].simulate(dt, frame_num, method, opt, q0, v0,
                [a0 for _ in range(frame_num)], f0,
                require_grad=False, vis_folder=method)
            export_mp4(folder / method, '{}.mp4'.format(str(folder / method)), fps=12)

    # Benchmark performance across different tolerances
    print('Performance Benchmark with C++ Heterogeneous Materials')
    print('='*70)
    print(f'DoFs: {dofs:d}, Frames: {frame_num:d}, dt: {dt:.3e}')

    rel_tols = [1e-1, 1e-2, 1e-3, 1e-4]
    benchmark_data = {}
    
    for rel_tol in rel_tols:
        print_info(f'rel_tol: {rel_tol:.3e}')
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'forward and backward (s)': '{:3.3f}',
            'forward only (s)': '{:3.3f}',
            'backward only (s)': '{:3.3f}',
            'loss': '{:3.3f}',
            '|grad|': '{:3.3f}'
        })
        print_info(tabular.head_string())
        
        for method, opt in zip(methods, opts):
            opt['rel_tol'] = rel_tol
            for thread_ct in thread_cts:
                opt['thread_ct'] = thread_ct
                meth_thread_num = f'{method}_{thread_ct}threads'

                loss, grad, info = envs[method].simulate(
                    dt, frame_num, method,
                    opt, q0, v0, [a0 for _ in range(frame_num)], f0,
                    require_grad=True, vis_folder=None
                )
                
                grad_q, grad_v, grad_a, grad_f = grad
                grad_combined = np.zeros(dofs + dofs + act_dofs + dofs)
                grad_combined[:dofs] = grad_q
                grad_combined[dofs:2*dofs] = grad_v
                grad_combined[2*dofs:2*dofs+act_dofs] = np.sum(ndarray(grad_a), axis=0)
                grad_combined[2*dofs+act_dofs:] = np.sum(ndarray(grad_f), axis=0)
                
                forward_time = info['forward_time']
                backward_time = info['backward_time']
                grad_norm = np.linalg.norm(grad_combined)
                
                print(tabular.row_string({
                    'method': meth_thread_num,
                    'forward and backward (s)': forward_time + backward_time,
                    'forward only (s)': forward_time,
                    'backward only (s)': backward_time,
                    'loss': loss,
                    '|grad|': grad_norm
                }))
                
                benchmark_data[f'{meth_thread_num}_rel_tol_{rel_tol}'] = {
                    'forward_time': forward_time,
                    'backward_time': backward_time,
                    'loss': loss,
                    'grad_norm': grad_norm
                }
    
    # Save benchmark results
    pickle.dump(benchmark_data, open(folder / 'benchmark_cpp_hetero.pkl', 'wb'))
    
    print(f"\n{'='*70}")
    print("Benchmark complete! Results saved.")
    print(f"{'='*70}\n")
    
    return benchmark_data


if __name__ == '__main__':
    verbose = True
    results = test_armadillo(verbose)