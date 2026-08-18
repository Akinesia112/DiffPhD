"""End-to-end installation check.

Drops a heterogeneous napkin onto a spherical obstacle and runs the same scene
through two solvers:

  pd_eigen_alg_phd         GPU sparse-inverse global solve + NCP contact
  pd_eigen_pcg_original baseline Projective Dynamics (reference)

It exercises the GPU path, the contact solver, heterogeneous materials, the pbrt
renderer and MP4 export, then reports whether the napkin actually bends on
contact instead of falling through or exploding.

Usage (from python/example, with the conda env active):

    python napkin_3d_test.py             # both solvers, with rendering
    python napkin_3d_test.py --no-vis    # simulation only, much faster
"""
import sys
sys.path.append('../')

import argparse
import pickle
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info, print_ok, print_error
from py_diff_pd.env.napkin_env_3d import NapkinEnv3d

# Scene / solver configuration -------------------------------------------------
SEED = 42
DT = 2e-3
FRAME_NUM = 125
THREAD_CT = 8
CONTACT_RATIO = 0.4
CELL_NUMS = (10, 10, 1)

METHODS = ('pd_eigen_alg_phd', 'pd_eigen_pcg_original')
OPTS = {
    'pd_eigen_alg_phd': {
        'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
        'verbose': 0, 'thread_ct': THREAD_CT, 'use_bfgs': 1, 'use_acc': 1,
        'use_sparse': 0, 'project_newton': 0, 'use_abs': 0,
    },
    'pd_eigen_pcg_original': {
        'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
        'verbose': 0, 'thread_ct': THREAD_CT, 'use_bfgs': 1, 'bfgs_history_size': 10,
        'project_newton': 0, 'use_abs': 0,
    },
}

# A correct run falls flat (z_std ~ 0.02) and then bends once it lands.
FLAT_Z_STD = 0.0200
BEND_Z_STD_MIN = 0.025      # must exceed this by the time it has landed
SCENE_SCALE_MAX = 5.0       # |q| far above this means the contact solve blew up


def summarize(q_flat):
    """q_flat: (frames, dofs) -> per-frame vertical mean / spread."""
    q = np.array(q_flat).reshape(len(q_flat), -1, 3)
    return q, q[:, :, 2]


def classify(q, z):
    max_abs_q = np.abs(q).max()
    if not np.isfinite(max_abs_q) or max_abs_q > SCENE_SCALE_MAX:
        return 'EXPLODED', 'contact solve blew up (max|q| = {:.3e})'.format(max_abs_q)
    landed = z[FRAME_NUM - 35:].std(axis=1).max()
    if landed < BEND_Z_STD_MIN:
        return 'NO CONTACT', ('napkin never bends (max z_std after landing = {:.5f}, '
                              'flat is {:.4f})'.format(landed, FLAT_Z_STD))
    return 'OK', 'lands and bends (max z_std after landing = {:.5f})'.format(landed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-vis', action='store_true',
                        help='skip pbrt rendering and MP4 export')
    args = parser.parse_args()

    parent = Path('napkin_3d_test')
    create_folder(parent, exist_ok=True)
    folder = parent / 'ratio_{:3f}'.format(CONTACT_RATIO)

    results = {}
    for method in METHODS:
        print_info('\n=== {} ==='.format(method))
        env = NapkinEnv3d(SEED, folder, {
            'contact_ratio': CONTACT_RATIO,
            'cell_nums': CELL_NUMS,
            'spp': 1,
            'het': 1,
        })
        deformable = env.deformable()
        q0 = env.default_init_position()
        v0 = env.default_init_velocity()
        a0 = [np.zeros(deformable.act_dofs())] * FRAME_NUM
        f0 = [np.zeros(deformable.dofs())] * FRAME_NUM

        _, info = env.simulate(DT, FRAME_NUM, method, dict(OPTS[method]),
                               np.copy(q0), np.copy(v0), a0, f0,
                               require_grad=False,
                               vis_folder=None if args.no_vis else method)
        pickle.dump(info, open(folder / '{}.data'.format(method), 'wb'))

        q, z = summarize(info['q'])
        verdict, detail = classify(q, z)
        results[method] = (verdict, detail, info['forward_time'], z)

        print('  forward: {:.3f}s'.format(info['forward_time']))
        print('  frame   z_mean     z_std')
        for f in (31, 32, 50, 90, FRAME_NUM):
            print('   {:3d}   {:8.5f}  {:8.5f}'.format(f, z[f].mean(), z[f].std()))

    print_info('\n=== summary ===')
    all_ok = True
    for method, (verdict, detail, t, _) in results.items():
        line = '{:24s} {:10s} {:.3f}s  {}'.format(method, verdict, t, detail)
        if verdict == 'OK':
            print_ok('  ' + line)
        else:
            print_error('  ' + line)
            all_ok = False

    # The two solvers discretise the same problem, so their trajectories should
    # agree to well under the scene scale.
    if all(v[0] == 'OK' for v in results.values()):
        za = results[METHODS[0]][3]
        zb = results[METHODS[1]][3]
        drift = np.abs(za - zb).max()
        print('  max |z_alg_phd - z_baseline| = {:.4e}'.format(drift))

    if not args.no_vis:
        print('\n  frames + MP4 under {}/'.format(folder))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
