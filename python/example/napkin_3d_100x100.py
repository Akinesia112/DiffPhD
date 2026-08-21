import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, PrettyTabular
from py_diff_pd.env.napkin_env_3d import NapkinEnv3d

if __name__ == '__main__':
    verbose = True
    benchmark = True
    seed = 42
    parent_folder = Path('napkin_3d_100x100')
    create_folder(parent_folder, exist_ok=True)
    dt = 2e-3
    frame_num = 125
    thread_ct = 8

    methods = ('pd_eigen_alg_phd', 'pd_eigen_pcg', 'pd_eigen_mas_pcg')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
                'use_bfgs': 1, 'use_acc': 1, 'use_sparse': 0, 'project_newton': 0,'use_abs': 0},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 1, 'thread_ct': thread_ct,
                 'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'project_newton': 0 },
            { 'max_pd_iter': 5000,'max_cg_iter': 50000,'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
                 'use_bfgs': 1, 'use_acc': 1,'use_mas': 1, 'bfgs_history_size': 10, 'project_newton': 0 })

    for ratio in [0.4, 0.8, 1.0, 1.6]:
        folder = parent_folder / 'ratio_{:3f}'.format(ratio)
        env = NapkinEnv3d(seed, folder, {
            'contact_ratio': ratio,
            'cell_nums': (100, 100, 1),
            'spp': 1,
            'het': 1,
        })
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.default_init_position()
        v0 = env.default_init_velocity()
        a0 = np.zeros(act_dofs)
        f0 = np.zeros(dofs)

        a0 = [a0 for _ in range(frame_num)]
        f0 = [f0 for _ in range(frame_num)]

        if verbose:
            for method, opt in zip(methods, opts):
                loss, info = env.simulate(dt, frame_num, method, opt, np.copy(q0), np.copy(v0), a0, f0, require_grad=False,
                    vis_folder=method)
                print('{} forward: {:3.3f}s'.format(method, info['forward_time']))
                pickle.dump(info, open(folder / '{}.data'.format(method), 'wb'))
        
        if benchmark:
            # Benchmark time.
            print('Reporting time cost. DoFs: {:d}, Contact Ratio: {:.3f}, frames: {:d}, dt: {:3.3e}'.format(dofs,
                ratio, frame_num, dt))
            rel_tols = [1e-4]
            forward_backward_times = {}
            forward_times = {}
            backward_times = {}
            losses = {}
            grads = {}
            for method in methods:
                    meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                    forward_backward_times[meth_thread_num] = []
                    forward_times[meth_thread_num] = []
                    backward_times[meth_thread_num] = []
                    losses[meth_thread_num] = []
                    grads[meth_thread_num] = []

            for rel_tol in rel_tols:
                print_info('rel_tol: {:3.3e}'.format(rel_tol))
                tabular = PrettyTabular({
                    'method': '{:^30s}',
                    'forward and backward (s)': '{:3.3f}',
                    'forward only (s)': '{:3.3f}',
                    'backward only (s)': '{:3.3f}',
                    'loss': '{:3.3f}',
                    '|grad|': '{:3.3f}',
                    'S mem (MiB)': '{:7.2f}',
                    'GPU VRAM (MiB)': '{:7.2f}'
                })
                print_info(tabular.head_string())

                for method, opt in zip(methods, opts):
                    opt['rel_tol'] = rel_tol
                    opt['thread_ct'] = thread_ct
                    meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                    loss, grad, info = env.simulate(dt, frame_num, method,
                        opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
                    grad_q, grad_v, grad_a, grad_f = grad
                    grad = np.zeros(q0.size + v0.size + a0[0].size + f0[0].size)
                    grad[:dofs] = grad_q
                    grad[dofs:2 * dofs] = grad_v
                    grad[2 * dofs:2 * dofs + act_dofs] = np.sum(ndarray(grad_a), axis=0)
                    grad[2 * dofs + act_dofs:] = np.sum(ndarray(grad_f), axis=0)
                    l, g, forward_time, backward_time = loss, grad, info['forward_time'], info['backward_time']
                    print(tabular.row_string({
                        'method': meth_thread_num,
                        'forward and backward (s)': forward_time + backward_time,
                        'forward only (s)': forward_time,
                        'backward only (s)': backward_time,
                        'loss': l,
                        '|grad|': np.linalg.norm(g),
                        'S mem (MiB)': info['s_memory_bytes'] / (1024 * 1024),
                        'GPU VRAM (MiB)': info['gpu_vram_used_bytes'] / (1024 * 1024) }))
                    forward_backward_times[meth_thread_num].append(forward_time + backward_time)
                    forward_times[meth_thread_num].append(forward_time)
                    backward_times[meth_thread_num].append(backward_time)
                    losses[meth_thread_num].append(l)
                    grads[meth_thread_num].append(g)
                pickle.dump((rel_tols, forward_times, backward_times, losses, grads), open(folder / 'table.bin', 'wb'))
