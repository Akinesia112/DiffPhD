import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np

from py_diff_pd.common.common import ndarray, print_info
from py_diff_pd.env.gatorman_bat_ball_env import GatormanBatBallEnv


FORWARD_METHOD = 'pd_eigen_alg_phd'


def default_config(frame_num=180):
    return {
        'seed': 42,
        'folder': Path('gatorman_ball'),
        'youngs_modulus': 1e5,
        'poissons_ratio': 0.45,
        'sword_stiffness': 6e6,
        'ball_youngs_modulus': 1e5,
        'ball_radius': 0.014,
        'ball_mesh_file': 'sphere_ico3.obj',
        'ball_rotation_degrees': ndarray([0.0, 0.0, 0.0]),
        'ball_center': ndarray([0.02, -0.08, 0.085]),
        'ball_initial_velocity': ndarray([-0.1, 0.0, 1.075]),
        'contact_model': 'mesh_boundary',
        'contact_radius': 1e-3,
        'use_ball_support': False,
        'use_ball_ground': False,
        'body_stiffness_scale': 1.0,
        'dt': 2e-3,
        'frame_num': int(frame_num),
        'release_frame': int(frame_num) // 2,
        'swing_force_scale': 0.025,
        'render_frame_skip': 2,
        'head_anchor_rank': 17,
        'thread_ct': 8,
        'method': 'pd_eigen_alg_phd',
    }


def solver_options(thread_ct, method):
    opt = {
        'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9,
        'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
    }
    if method == 'pd_eigen_alg_phd':
        opt.update({
            'use_bfgs': 0, 'use_acc': 1, 'use_sparse': 0,
            'project_newton': 1, 'use_abs': 1, 'aa_window': 5,
            'friction_mu': 0.1, 'friction_kf': 1.0, 'restitution': 0.0,
        })
    elif method == 'pd_eigen':
        opt.update({
            'use_bfgs': 1, 'bfgs_history_size': 10,
            'project_newton': 1, 'use_abs': 1,
        })
    else:
        raise ValueError('Unsupported method: {}'.format(method))
    return opt


def with_method(config, method):
    config = dict(config)
    config['method'] = method
    return config


def solver_label(config):
    return config.get('method', 'pd_eigen_alg_phd')


def build_swing_forces(env, q0, frame_num, release_frame, swing_force_scale):
    q0_vertices = ndarray(q0).reshape((-1, 3))
    drive_nodes = env.drive_nodes()
    base_center = np.mean(q0_vertices[env.base_nodes()], axis=0)
    head_anchor = q0_vertices[env.head_anchor_node()]
    pivot = np.copy(base_center)
    pivot[:2] = head_anchor[:2]
    axis = ndarray([0.0, 0.0, 1.0])
    dofs = env.deformable().dofs()
    f_ext = [np.zeros((dofs // 3, 3), dtype=np.float64) for _ in range(frame_num)]

    for t in range(frame_num):
        if t < release_frame:
            phase = (t + 1) / float(release_frame)
            envelope = np.sin(np.pi * phase)
            direction = 1.0
        else:
            envelope = 1.0
            direction = -1.0
        r = q0_vertices[drive_nodes] - pivot
        tangent = direction * -np.cross(axis, r)
        f_ext[t][drive_nodes] = swing_force_scale * envelope * tangent

    return [f.ravel() for f in f_ext]


def apply_ball_initial_velocity(env, v0, ball_initial_velocity):
    v = ndarray(v0).reshape((-1, 3)).copy()
    if env.include_ball():
        v[env.ball_vertices()] = ball_initial_velocity
    return v.ravel()


def ball_xy_objective(env, q0, q_final):
    initial_xy = env.ball_center(q0)[:2]
    final_xy = env.ball_center(q_final)[:2]
    displacement = final_xy - initial_xy
    return {
        'initial_xy': initial_xy,
        'final_xy': final_xy,
        'displacement': displacement,
        'xy_distance': float(np.linalg.norm(displacement)),
    }


def make_env(config, sword_stiffness=None, loss_type='random_linear',
        pd_energy_mode='per_element'):
    np.random.seed(config['seed'])
    options = {
        'youngs_modulus': config['youngs_modulus'],
        'poissons_ratio': config['poissons_ratio'],
        'sword_stiffness': config['sword_stiffness'] if sword_stiffness is None else sword_stiffness,
        'body_stiffness_scale': config['body_stiffness_scale'],
        'ball_youngs_modulus': config['ball_youngs_modulus'],
        'ball_radius': config['ball_radius'],
        # Keep the current tuned forward path: the denser mesh is available but
        # not enabled here until the forward constants are intentionally changed.
        # 'ball_mesh_file': config['ball_mesh_file'],
        'ball_rotation_degrees': config['ball_rotation_degrees'],
        'ball_center': config['ball_center'],
        'use_ball_support': config['use_ball_support'],
        'use_ball_ground': config['use_ball_ground'],
        'contact_model': config['contact_model'],
        'contact_radius': config['contact_radius'],
        'mesh_boundary_candidate_mode': 'ball',
        'mesh_boundary_candidate_stride': 1,
        'head_anchor_rank': config['head_anchor_rank'],
        'state_force_parameters': [0.0, 0.0, -9.81],
        'include_ball': True,
        'loss_type': loss_type,
        'pd_energy_mode': pd_energy_mode,
        'spp': 4,
        'visualize_material_stiffness': True,
        'visualize_force_nodes': True,
        'visualize_head_anchor': True,
        'force_marker_stride': 25,
        'force_marker_radius': 0.0011,
        'head_anchor_marker_radius': 0.0025,
    }
    return GatormanBatBallEnv(config['seed'], config['folder'], options)


def build_rollout(env, config, frame_num=None):
    frame_num = config['frame_num'] if frame_num is None else int(frame_num)
    release_frame = min(config['release_frame'], frame_num)
    deformable = env.deformable()
    q0 = env.default_init_position()
    v0 = apply_ball_initial_velocity(
        env, env.default_init_velocity(), config['ball_initial_velocity']
    )
    a0 = [np.zeros(deformable.act_dofs()) for _ in range(frame_num)]
    f0 = build_swing_forces(
        env, q0, frame_num, release_frame, config['swing_force_scale']
    )
    return q0, v0, a0, f0


def print_setup(env, config):
    deformable = env.deformable()
    print_info('Gatorman bat-ball complementary setup:')
    print('  contact_model: {}'.format(env.contact_model()))
    print('  contact_radius: {:.3e}'.format(env.contact_radius()))
    print('  use_ball_support: {}'.format(env.use_ball_support()))
    print('  use_ball_ground: {}'.format(env.use_ball_ground()))
    print('  ball_mesh_file: {}'.format(env.ball_mesh_file()))
    print('  ball_initial_velocity: {}'.format(config['ball_initial_velocity']))
    print('  method: {}'.format(solver_label(config)))
    print('  loss_type: {}'.format(env.loss_type()))
    print('  pd_energy_mode: {}'.format(env.pd_energy_mode()))
    print('  pd_energy_groups: {}'.format(env.pd_energy_groups()))
    print('  DoFs: {:d}'.format(deformable.dofs()))
    print('  regions: {}'.format(env.region_summary()))


def test_gatorman_ball(verbose=True, frame_num=180,
        vis_folder=None, method=None):
    """Forward-only sword-ball contact test for Gatorman."""
    method = FORWARD_METHOD if method is None else method
    vis_folder = method if vis_folder is None else vis_folder
    config = with_method(default_config(frame_num), method)
    env = make_env(config)
    print_setup(env, config)

    if not verbose:
        return None

    q0, v0, a0, f0 = build_rollout(env, config)
    method = solver_label(config)
    opt = solver_options(config['thread_ct'], method)
    print('Running complementary sword-ball forward simulation...')
    print(method, opt)
    t0 = time.time()
    loss, info = env.simulate(
        config['dt'], config['frame_num'], method, opt, q0, v0, a0, f0,
        require_grad=False, vis_folder=vis_folder,
        render_frame_skip=config['render_frame_skip']
    )
    elapsed = time.time() - t0
    objective = ball_xy_objective(env, q0, info['q'][-1])

    print('=== Gatorman Bat-Ball Complementary Summary ===')
    print('frames: {:d}, dt: {:.3e}, release_frame: {:d}'.format(
        config['frame_num'], config['dt'], config['release_frame']))
    print('loss: {:.6e}'.format(loss))
    print('forward time: {:.3f}s'.format(info['forward_time']))
    print('elapsed: {:.3f}s'.format(elapsed))
    print('initial ball xy: {}'.format(objective['initial_xy']))
    print('final ball xy: {}'.format(objective['final_xy']))
    print('ball xy displacement: {}'.format(objective['displacement']))
    print('ball xy distance: {:.6e}'.format(objective['xy_distance']))
    if vis_folder is not None:
        print('Visualization output saved to folder: {}'.format(config['folder'] / vis_folder))
    return loss, info


if __name__ == '__main__':
    test_gatorman_ball(verbose=True, method=FORWARD_METHOD)
