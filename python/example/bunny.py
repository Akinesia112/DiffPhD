import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.bunny_env import BunnyEnv

def apply_transform(q, R, t):
    q = ndarray(q).reshape((-1, 3))
    com = np.mean(q, axis=0)
    return ((q - com) @ R.T + t).ravel()

if __name__ == '__main__':
    seed = 211
    folder = Path('bunny')
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    target_com = ndarray([0.15, 0.15, 0.15])
    env = BunnyEnv(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target_com': target_com,
        'mesh_type': 'tet',
        'spp': 64,
        'het': True,
        'contrast_factor': 100.0,
        })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    contact_opt = { 'friction_mu': 0.1, 'restitution': 0.0, 'skip_contact_woodbury': 0 }
    methods = ('pd_eigen', 'pd_eigen_cuda_mas_pcg', 'pd_eigen_alg_phd')
    opts = ({ 'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10},
            { 'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'use_mas': 1},
            { 'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5, **contact_opt },)

    dt = 1e-3
    frame_num = 100

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_initial_states(x):
        init_rpy = x[:3]
        init_R = rpy_to_rotation(init_rpy)
        init_com_q = x[3:6]
        init_com_v = x[6:9]
        init_q = apply_transform(q0, init_R, init_com_q)
        init_v = (v0.reshape((-1, 3)) + init_com_v).ravel()
        return np.copy(init_q), np.copy(init_v)
    def variable_to_initial_states_gradient(x, grad_init_q, grad_init_v):
        grad = np.zeros(x.size)
        # init_rpy:
        offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
        # init_q = (offset @ R.T + init_com_q).ravel()
        rpy = x[:3]
        dR_dr, dR_dp, dR_dy = rpy_to_rotation_gradient(rpy)
        grad[0] = (offset @ dR_dr.T).ravel().dot(grad_init_q)
        grad[1] = (offset @ dR_dp.T).ravel().dot(grad_init_q)
        grad[2] = (offset @ dR_dy.T).ravel().dot(grad_init_q)
        # init_com_q:
        grad[3:6] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # init_com_v:
        grad[6:9] = np.sum(grad_init_v.reshape((-1, 3)), axis=0)
        return grad

    # Optimization.
    # Variables to be optimized:
    # init_rpy (3D), init_com_q (3D), init_com_v (3D).
    x_lb = ndarray([-np.pi / 3, -np.pi / 3, -np.pi / 3, -0.03, -0.03, 0.19, -2.0, -2.0, -8.0])
    x_ub = ndarray([np.pi / 3, np.pi / 3, np.pi / 3,  0.03,  0.03, 0.21,  2.0,  4.0, -4.0])
    rng = np.random.default_rng(seed)
    x_init = rng.uniform(x_lb, x_ub)
    # Visualize initial guess.
    init_q, init_v = variable_to_initial_states(x_init)
    loss, _ = env.simulate(dt, frame_num, methods[0], opts[0], init_q, init_v, a0, f0, require_grad=False, vis_folder="init_try")
    print('loss: ', loss)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    data = {}
    for method, opt in zip(methods, opts):
        data[method] = []

        def loss_and_grad(x):
            init_q, init_v = variable_to_initial_states(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=True, vis_folder=None)
            # Assemble the gradients.
            grad_init_q = grad[0]
            grad_init_v = grad[1]
            grad_x = variable_to_initial_states_gradient(x, grad_init_q, grad_init_v)
            final_com = np.mean(info['q'][-1].reshape((-1, 3)), axis=0)
            print('loss: {:10.6e}, |grad|: {:10.6e}, final_com: [{:8.4f}, {:8.4f}, {:8.4f}], forward: {:6.3f}s, backward: {:6.3f}s'.format(
                loss, np.linalg.norm(grad_x), final_com[0], final_com[1], final_com[2], info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad_x)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            single_data['final_com'] = np.copy(final_com)
            data[method].append(single_data)
            return loss, np.copy(grad_x)

        # Sanity check the gradients; tune opt's rel_tol if this fails to converge.
        print("x init: ", x_init)
        check_gradients(loss_and_grad, x_init, eps=1e-6)

        t0 = time.time()
        # Try L-BFGS-B with relaxed line search first.
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, 
            options={ 
                'ftol': 1e-7,
                'gtol': 1e-5,
                'maxls': 100,  # Allow more line search attempts.
                'maxiter': 500  # Cap iterations to avoid excessive contact recomputation.
            })
        t1 = time.time()
        print_info('L-BFGS-B result:')
        print(result.message)
        print('status:', result.status)
        
        # If L-BFGS-B fails with line search error, try simple gradient descent.
        if result.status == 2:  # ABNORMAL_TERMINATION_IN_LNSRCH
            print_error('L-BFGS-B line search failed; switching to gradient descent.')
            print_info('Running gradient descent with adaptive step size...')
            
            x_gd = np.copy(x_init)
            gd_step = 0.01
            gd_history = []
            for gd_iter in range(50):
                loss_k, grad_k = loss_and_grad(x_gd)
                grad_norm = np.linalg.norm(grad_k)
                
                if grad_norm < 1e-4:
                    print_ok('Gradient descent converged at iter {}: |grad|={:8.3e}'.format(gd_iter, grad_norm))
                    break
                
                # Try adaptive step size.
                x_trial = np.clip(x_gd - gd_step * grad_k / (grad_norm + 1e-10), x_lb, x_ub)
                loss_trial, _ = loss_and_grad(x_trial)
                
                if loss_trial < loss_k:
                    # Accept step and possibly increase step size.
                    x_gd = x_trial
                    gd_step = min(gd_step * 1.05, 0.05)
                    improvement = loss_k - loss_trial
                else:
                    # Reject and decrease step size.
                    gd_step *= 0.5
                    if gd_step < 1e-6:
                        print_info('Gradient descent step size too small; stopping.')
                        break
                
                print_info('GD iter {:3d}: loss={:8.3f}, |grad|={:8.3e}, step={:8.6f}'.format(
                    gd_iter, loss_trial, grad_norm, gd_step))
                gd_history.append(loss_trial)
            
            result.x = x_gd
            result.success = (grad_norm < 1e-4)
            data[method + '_gd_history'] = gd_history
        if result.success:
            x_final = result.x
        else:
            print_error('Optimizer did not converge; use best-so-far state for simulation output.')
            if len(data[method]) > 0:
                best_idx = int(np.argmin([d['loss'] for d in data[method]]))
                x_final = np.copy(data[method][best_idx]['x'])
                print_info('Using best-so-far loss: {:8.3f}'.format(data[method][best_idx]['loss']))
            else:
                x_final = np.copy(x_init)
                print_info('No optimization samples available; fallback to x_init.')
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads2.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        final_q, final_v = variable_to_initial_states(x_final)
        env.simulate(dt, frame_num, method, opt, final_q, final_v, a0, f0, require_grad=False, vis_folder='final_render')