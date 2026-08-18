import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, print_info
from py_diff_pd.env.bouncing_ball_env import BouncingBallEnv

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('bouncing_ball_3d')
    youngs_modulus = 2e6
    poissons_ratio = 0.4
    env = BouncingBallEnv(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    contact_opt = { 'friction_mu': 0.5, 'restitution': 1.0, 'skip_contact_woodbury': 0,
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
    v_gt = ndarray([2.3, 1.0, -2.0])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion. The optimization loop below reads these
    # frames back from the 'groundtruth' folder via _stepwise_loss_and_grad,
    # so this must run before any loss/gradient computation is meaningful.
    loss_gt, _ = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0,
        require_grad=False, vis_folder='groundtruth')
    print(f'Groundtruth loss (this run vs itself, should be 0): {loss_gt:.3f}')

    # Optimization.
    # Decision variables: log(E), log(nu).
    x_lb = ndarray([np.log(1e6), np.log(0.2)])
    x_ub = ndarray([np.log(1e7), np.log(0.45)])
    # For this example, we intentionally pick an initial guess far from the ground truth so that the initial motion
    # is distinguishable enough from the ground truth motion.
    x_init = ndarray([np.log(5e6), np.log(0.35)])
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    data = {}
    for method, opt in zip(reversed(methods), reversed(opts)):
        data[method] = []
        def loss_and_grad(x):
            E = np.exp(x[0])
            nu = np.exp(x[1])
            print("simulating with E: {:3e}, nu: {:3f}...".format(E, nu))
            env_opt = BouncingBallEnv(seed, folder, { 'youngs_modulus': E,
                'poissons_ratio': nu })
            loss, _, info = env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
            raw_grad = info['material_parameter_gradients']
            print('  raw dL/dlog(E,nu): [{:.3e}, {:.3e}]  norm={:.3e}'.format(
                raw_grad[0], raw_grad[1], np.linalg.norm(raw_grad)))
            grad = raw_grad * np.exp(x)
            if not np.all(np.isfinite(grad)):
                print('  WARNING: non-finite gradient detected, replacing with zeros')
                grad = np.zeros_like(grad)
            print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['E'] = E
            single_data['nu'] = nu
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, grad
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={ 'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 30, 'maxls': 20 })
        t1 = time.time()
        print(result.success)
        print(result.message)
        print('nit:', result.nit, 'nfev:', result.nfev)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        E = np.exp(x_final[0])
        nu = np.exp(x_final[1])
        env_opt = BouncingBallEnv(seed, folder, { 'youngs_modulus': E, 'poissons_ratio': nu })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method)
