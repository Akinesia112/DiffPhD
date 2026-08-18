import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray
from py_diff_pd.common.common import print_info
from py_diff_pd.env.bouncing_ball_env import BouncingBallEnv

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    youngs_modulus = 2e6
    poissons_ratio = 0.4
    folder = Path('bouncing_ball_3d_het')

    env = BouncingBallEnv(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'het': True,    # change the hetero settings in env 
    })
    deformable = env.deformable()

    modulus_array = env.get_element_modulus_array()
    print_info('Heterogeneous material summary:')
    print('  element count: {:d}'.format(len(modulus_array)))
    print('  E min={:.2e}, max={:.2e}, mean={:.2e}, std={:.2e}, range={:.1f}x'.format(
        modulus_array.min(), modulus_array.max(), modulus_array.mean(),
        modulus_array.std(), modulus_array.max() / modulus_array.min()))
    print('  unique E values: {}'.format(
        ['{:.2e}'.format(v) for v in np.unique(np.round(modulus_array, decimals=6))]))

    # Optimization parameters.
    thread_ct = 8
    contact_opt = { 'friction_mu': 0.5, 'friction_kf': 2e2, 'restitution': 0.0,
        'backward_grad_clip_norm': None }
    methods = ('pd_eigen', 'pd_eigen_cuda_mas_pcg', 'pd_eigen_alg_phd')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'use_mas': 1},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5, **contact_opt },)

    dt = 4e-3
    frame_num = 100

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_gt = ndarray([0.0, 0.0, 0.07])
    v_gt = ndarray([2.0, 1.7, -1.5])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    loss_gt, _ = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0,
        require_grad=False, vis_folder='groundtruth')
    print(f'Groundtruth loss (this run vs itself, should be 0): {loss_gt:.3f}')

    # Optimization.
    # Decision variables: log stiff/mid/soft band multipliers.
    # Poisson ratio and the base Young's modulus stay fixed.
    opt_frame_num = 100
    a0_opt = [np.zeros(act_dofs) for _ in range(opt_frame_num)]
    f0_opt = [np.zeros(dofs) for _ in range(opt_frame_num)]
    x_lb = np.log(ndarray([0.1, 0.03, 0.1]))
    x_ub = np.log(ndarray([4.0, 1.0, 2.0]))
    x_init = np.log(ndarray([2.0, 0.3, 1.0]))
    target_x = np.log(env.default_band_multipliers)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    data = {}
    for method, opt in zip(reversed(methods), reversed(opts)):
        data[method] = []
        def loss_and_grad(x):
            multipliers = np.exp(x)
            E_stiff, E_mid, E_soft = youngs_modulus * multipliers
            print("simulating with E_stiff: {:.3e}, E_mid: {:.3e}, E_soft: {:.3e}...".format(
                E_stiff, E_mid, E_soft))
            env_opt = BouncingBallEnv(seed, folder, {
                'youngs_modulus': youngs_modulus,
                'poissons_ratio': poissons_ratio,
                'het': True,
                'log_band_multipliers': x,
            })
            loss, _, info = env_opt.simulate(dt, opt_frame_num, method, opt, q0,
                v0, a0_opt, f0_opt, require_grad=True, vis_folder=None)
            grad = env_opt.band_multiplier_gradients(info['dl_dmat_w'], x)
            if not np.all(np.isfinite(grad)):
                print('  WARNING: non-finite gradient detected, replacing with zeros')
                grad = np.zeros_like(x)
            dist = np.linalg.norm(x - target_x)
            print('loss: {:8.3f}, |grad|: {:8.3f}, E_stiff: {:8.3e}, E_mid: {:8.3e}, E_soft: {:8.3e}, '
                'forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), E_stiff, E_mid, E_soft,
                info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['multipliers'] = np.copy(multipliers)
            single_data['x'] = np.copy(x)
            single_data['distance_to_target'] = dist
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, grad

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={ 'ftol': 1e-12, 'gtol': 1e-6, 'maxiter': 10 })
        t1 = time.time()
        print(result.success)
        print(result.message)
        print('nit:', result.nit, 'nfev:', result.nfev)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(
            method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(
            thread_ct), 'wb'))

        # Visualize results.
        env_opt = BouncingBallEnv(seed, folder, {
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'het': True,
            'log_band_multipliers': x_final,
        })
        env_opt.simulate(dt, opt_frame_num, method, opt, q0, v0, a0_opt, f0_opt,
            require_grad=False, vis_folder=method)
