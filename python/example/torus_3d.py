import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, print_info
from py_diff_pd.env.torus_env_3d import TorusEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('torus_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.4
    act_stiffness = 2e5
    act_group_num = 8
    env = TorusEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'act_stiffness': act_stiffness,
        'act_group_num': act_group_num
    })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    methods = ('pd_eigen', 'pd_eigen_cuda_mas_pcg', 'pd_eigen_alg_phd')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'use_acc': 1, 'bfgs_history_size': 10, 'use_mas': 1},
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 0, 'use_acc': 1, 'project_newton': 1, 'use_abs': 1, 'aa_window': 5,
        'friction_mu': 0.3, 'restitution': 1.0, 'skip_contact_woodbury': 0},)

    dt = 4e-3
    frame_num = 200
    control_frame_num = 20
    assert frame_num % control_frame_num == 0

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = env.default_init_velocity()
    v0 = (v0.reshape((-1, 3)) + ndarray([0.25, 0.0, 0.0])).ravel()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Compute actuation.
    control_frame = int(frame_num // control_frame_num)
    x_lb = np.zeros(act_group_num * control_frame)
    x_ub = np.ones(act_group_num * control_frame) * 2
    x_init = np.random.uniform(low=x_lb, high=x_ub)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    act_groups = env.act_groups()
    def variable_to_act(x):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        acts = []
        for x_aug_frame in x_aug:
            frame_act = np.zeros(act_dofs)
            for i, group in enumerate(act_groups):
                for j in group:
                    frame_act[j] = x_aug_frame[i]
            acts.append(frame_act)
        acts = ndarray(acts)
        return acts

    def variable_to_gradient(x, dl_dact):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        grad_x_aug = np.zeros((frame_num, act_group_num))
        for k in range(frame_num):
            x_aug_frame = x_aug[k]
            grad_act = dl_dact[k]
            for i, group in enumerate(act_groups):
                for j in group:
                    grad_x_aug[k, i] += grad_act[j]

        # Backpropagate from grad_x_aug to grad.
        grad = np.zeros(x.shape)
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                grad[c] += (1 - t) * grad_x_aug[c * control_frame_num + i]
                grad[c_next] += t * grad_x_aug[c * control_frame_num + i]

        return grad.ravel()

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 4 # 16
    random_loss = []
    # Since this example is easily trapped in local minima, we pick the best random guesses as the initial state.
    x_init = np.random.uniform(low=x_lb, high=x_ub)
    best_loss = np.inf
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        act = variable_to_act(x_rand)
        loss, _ = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, act, f0, require_grad=False, vis_folder=None)
        print('loss: {:3f}'.format(loss))
        random_loss.append(loss)
        if loss < best_loss:
            best_loss = loss
            x_init = x_rand
    loss_range = ndarray([0, -0.062594])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    # Initial state after selecting the best from random guesses.
    a0 = variable_to_act(x_init)
    # Sanity check — uncomment to verify analytic == numeric.
    # from py_diff_pd.common.grad_check import check_gradients
    # random_weight = np.random.normal(size=ndarray(a0).shape)
    # def loss_and_grad(x):
    #     act = variable_to_act(x)
    #     loss = np.sum(ndarray(act) * random_weight)
    #     grad = variable_to_gradient(x, random_weight)
    #     return loss, grad
    # check_gradients(loss_and_grad, x_init, verbose=True)

    # Optimization.
    data = { 'loss_range': loss_range }
    for method, opt in zip(methods, opts):
        data[method] = []
        data[method + '_iter'] = []
        eval_cache = {}
        def key_of(x):
            return np.ascontiguousarray(np.round(ndarray(x).ravel(), 12), dtype=np.float64).tobytes()

        def loss_and_grad(x):
            act = variable_to_act(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, act, f0, require_grad=True, vis_folder=None)
            dl_act = grad[2]
            grad = variable_to_gradient(x, dl_act)
            grad_abs = np.abs(grad)
            rec = {
                'loss': float(loss),
                'grad_norm': float(np.linalg.norm(grad)),
                'grad_abs_min': float(np.min(grad_abs)),
                'grad_abs_max': float(np.max(grad_abs)),
                'forward_time': info['forward_time'],
                'backward_time': info['backward_time'],
                'x': np.copy(x),
                'grad': np.copy(grad),
            }
            eval_cache[key_of(x)] = rec

            print('loss: {:8.3f}, |grad|: {:8.3f}, grad_min/max: {:.3e}/{:.3e}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), np.min(np.abs(grad)), np.max(np.abs(grad)), info['forward_time'], info['backward_time']))

            data[method].append(rec)
            return loss, np.copy(grad)

        def on_iter(xk):
            rec = eval_cache.get(key_of(xk))
            if rec is None:
                # The callback may receive xk with tiny numeric differences.
                rec = {
                    'loss': np.nan,
                    'grad_norm': np.nan,
                    'grad_abs_min': np.nan,
                    'grad_abs_max': np.nan,
                    'x': np.copy(xk),
                }
            data[method + '_iter'].append(rec)
            print('[iter] #{:02d} accepted loss: {:8.3f}, |grad|: {:8.3f}, grad_min/max: {:.3e}/{:.3e}'.format(
                len(data[method + '_iter']),
                rec['loss'], rec['grad_norm'], rec['grad_abs_min'], rec['grad_abs_max']))

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, callback=on_iter,
            options={ 'ftol': 1e-3, 'maxiter': 30, 'gtol': 1e-3 })
        t1 = time.time()
        print(result.success)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        a_final = variable_to_act(x_final)
        env.simulate(dt, frame_num, method, opt, q0, v0, a_final, f0, require_grad=False, vis_folder=method)
