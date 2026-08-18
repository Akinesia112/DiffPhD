import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, print_info
from py_diff_pd.env.plant_env_3d import PlantEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('plant_3d')
    youngs_modulus = 1e6
    poissons_ratio = 0.4
    env = PlantEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    methods = ('pd_eigen', 'pd_eigen_cuda_mas_pcg', 'pd_eigen_alg_phd')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'use_mas': 1},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5},)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    dt = 1e-2

    q_default = env.default_init_position()
    v_zero = np.zeros(dofs)

    # Phase 1: 3 push frames.
    vertex_num = int(dofs // 3)
    f_push = np.zeros((vertex_num, 3))
    f_push[:, 0] = -1
    f_push = f_push.ravel()
    a0_3   = [np.zeros(act_dofs) for _ in range(3)]
    f0_3   = [f_push] * 3

    # Phase 2: 100 free-oscillation frames.
    free_frames = 100
    a0_100 = [np.zeros(act_dofs) for _ in range(free_frames)]
    f0_100 = [np.zeros(dofs) for _ in range(free_frames)]

    # ── Generate groundtruth ──────────────────────────────────────────────────
    print_info('Groundtruth (E={:.3e}, nu={:.4f})...'.format(youngs_modulus, poissons_ratio))
    _, info_push = env.simulate(dt, 3, methods[0], opts[0],
                                q_default, v_zero, a0_3, f0_3,
                                require_grad=False, vis_folder=None)
    q0_gt = info_push['q'][-1]

    env.simulate(dt, free_frames, methods[0], opts[0],
                 q0_gt, v_zero, a0_100, f0_100,
                 require_grad=False, vis_folder='groundtruth')

    # ── Optimization ──────────────────────────────────────────────────────────
    x_lb = ndarray([np.log(1e5), np.log(0.2)])
    x_ub = ndarray([np.log(5e6), np.log(0.45)])
    x_init = np.random.uniform(x_lb, x_ub)
    print_info('Initial guess: E={:.3e}, nu={:.4f}'.format(
        np.exp(x_init[0]), np.exp(x_init[1])))
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    
    # ── Visualize specific x ──────────────────────────────────────────────────
    # x_init = x_ub
    # env_init = PlantEnv3d(seed, folder, {
    #     'youngs_modulus': np.exp(x_init[0]),
    #     'poissons_ratio': np.exp(x_init[1])})
    # _, info1 = env_init.simulate(dt, 3, methods[0], opts[0],
    #                             q_default, v_zero, a0_3, f0_3,
    #                             require_grad=False, vis_folder=None)
    # q0_init = info1['q'][-1]
    # env_init.simulate(dt, free_frames, methods[0], opts[0],
    #                   q0_init, v_zero, a0_100, f0_100,
    #                   require_grad=False, vis_folder='specific')
    
    data = {}
    for method, opt in zip(methods, opts):
        data[method] = []
        print_info('Starting optimization with {}...'.format(method))
        def loss_and_grad(x):
            E = np.exp(x[0])
            nu = np.exp(x[1])
            env_cur = PlantEnv3d(seed, folder, {'youngs_modulus': E, 'poissons_ratio': nu})

            # Phase 1 forward only: get deformed initial position for this E.
            _, info1 = env_cur.simulate(dt, 3, method, opt,
                                        q_default, v_zero, a0_3, f0_3,
                                        require_grad=False, vis_folder=None)
            q0_E = info1['q'][-1]

            # Phase 2: free oscillation from (q0_E, v=0), compute loss + grad.
            loss, _, info2 = env_cur.simulate(dt, free_frames, method, opt,
                                              q0_E, v_zero, a0_100, f0_100,
                                              require_grad=True, vis_folder=None)
            raw_grad = info2['material_parameter_gradients']
            grad = raw_grad * np.exp(x)  # Chain rule for log parameterization.
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = np.clip(grad, -1e4, 1e4)
            print("raw_grad -> dL/dE: {:.3e}, dL/dnu: {:.3e}".format(raw_grad[0], raw_grad[1]))
            print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, '
                  'forward: {:6.3f}s, backward: {:6.3f}s'.format(
                  loss, np.linalg.norm(grad), E, nu,
                  info2['forward_time'], info2['backward_time']))
            data[method].append({
                'loss': loss, 'grad': np.copy(grad),
                'E': E, 'nu': nu,
                'forward_time': info2['forward_time'],
                'backward_time': info2['backward_time'],
            })
            return loss, grad

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={'ftol': 1e-2, 'maxiter': 10})
        t1 = time.time()
        print(result.success)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f}s'.format(method, t1 - t0))
        print_info('Final:  E={:.3e}, nu={:.4f}'.format(
            np.exp(x_final[0]), np.exp(x_final[1])))
        print_info('Target: E={:.3e}, nu={:.4f}'.format(youngs_modulus, poissons_ratio))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize final result.
        env_final = PlantEnv3d(seed, folder, {
            'youngs_modulus': np.exp(x_final[0]),
            'poissons_ratio': np.exp(x_final[1])})
        _, info_push_final = env_final.simulate(dt, 3, method, opt,
                                                q_default, v_zero, a0_3, f0_3,
                                                require_grad=False, vis_folder=None)
        q0_final = info_push_final['q'][-1]
        env_final.simulate(dt, free_frames, method, opt,
                           q0_final, v_zero, a0_100, f0_100,
                           require_grad=False, vis_folder=method)