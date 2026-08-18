#!/usr/bin/env python3
"""Optimize slab_center (sx, sy) — material/contact/damping fixed.

GT world transform is precomputed ONCE with REF_SLAB_CENTER, then held
fixed while we vary the sim's slab_center. This decouples sim dice
position from GT placement so chamfer has a real signal in (sx, sy).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dice_material_optimize import (  # noqa: E402
    ATLAS_FRAME, METHOD, OPT_FWD, asym_poke_keyframes, build_sim,
    load_gt_surfaces, simulate_and_loss,
)
from py_diff_pd.core.py_diff_pd_core import StdRealVector  # noqa: E402
from ur5_poke_demo import compute_fk  # noqa: E402


REF_SLAB_CENTER = (0.70, 0.089, -0.170)
SLAB_SIZE       = (0.30, 0.30, 0.06)
SLAB_Z          = -0.170

E_FIXED   = 5e5
NU_FIXED  = 0.40
POKE_AMP  = 0.18

FRAME_START = ATLAS_FRAME
FRAME_END   = 53
DT          = 1.0 / 30.0
FD_EPS      = 1e-3   # 1 mm finite-diff step (slab is 30 cm)

OUT_DIR  = Path(__file__).resolve().parent / 'dice_xy_opt'
LOG_PATH = OUT_DIR / 'optimize.log'


def _log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open('a') as f:
        f.write(msg + '\n')


def _silent_forward_full(sx: float, sy: float, gt_by_frame, keyframes):
    """Returns (sim, fwd_dict, fwd_time_seconds). Keeps sim + fwd state for
    the backward pass."""
    sc = (float(sx), float(sy), SLAB_Z)
    sim = build_sim(E_dice=E_FIXED, nu_dice=NU_FIXED, slab_center=sc)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    t0 = time.time()
    try:
        fwd = simulate_and_loss(
            sim, keyframes, gt_by_frame, FRAME_START, DT, METHOD, OPT_FWD,
            lambda_recon=1.0, lambda_geom=1.0)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    return sim, fwd, time.time() - t0


def _silent_forward(sx, sy, gt_by_frame, keyframes):
    sim, fwd, t = _silent_forward_full(sx, sy, gt_by_frame, keyframes)
    return float(fwd['loss']), t


def _backward_dq0(sim, fwd, n_substeps, n_total) -> tuple:
    """Adjoint chain over all sub-steps, returns (dl_dq0, bwd_time).
    Mirrors backward_material_grad exactly, but tracks dl_dq_next only and
    returns it at step 0 — that IS dl/d(q0)."""
    deformable = sim['deformable']
    set_all_dirichlet = sim['set_all_dirichlet']
    dofs       = deformable.dofs()
    act_dofs   = deformable.act_dofs()
    mat_w_dofs = deformable.NumOfPdElementEnergies()
    act_w_dofs = deformable.NumOfPdMuscleEnergies()
    state_p_dofs = deformable.NumOfStateForceParameters()
    grad_clip  = 1e3

    def _clip(g):
        g = np.nan_to_num(g, nan=0.0,
                          posinf=grad_clip, neginf=-grad_clip)
        n = np.linalg.norm(g)
        if n > grad_clip:
            g = g * (grad_clip / n)
        return g

    dt_sub = DT / n_substeps
    kf_per_substep = fwd['kf_per_substep']

    dl_dq_next = _clip(fwd['dl_dq'][n_total].copy())
    dl_dv_next = np.zeros(dofs)

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2); os.dup2(devnull, 2)
    t0 = time.time()
    try:
        for i in reversed(range(n_total)):
            fk = compute_fk(kf_per_substep[i])
            pos_ee, R_ee = fk['ee_link']
            set_all_dirichlet(R_ee, pos_ee)

            dl_dq   = StdRealVector(dofs)
            dl_dv_c = StdRealVector(dofs)
            dl_da   = StdRealVector(act_dofs)
            dl_df   = StdRealVector(dofs)
            dl_dmw  = StdRealVector(mat_w_dofs)
            dl_daw  = StdRealVector(act_w_dofs)
            dl_dsp  = StdRealVector(state_p_dofs)

            act_i = np.zeros(act_dofs)
            f_ext_i = fwd['f_ext'][i]
            deformable.PyBackward(
                METHOD, fwd['q'][i], fwd['v'][i], act_i, f_ext_i, dt_sub,
                fwd['q'][i + 1], fwd['v'][i + 1], fwd['ci'][i + 1],
                dl_dq_next, dl_dv_next, OPT_FWD,
                dl_dq, dl_dv_c, dl_da, dl_df, dl_dmw, dl_daw, dl_dsp)

            dl_dq_next = _clip(np.asarray(dl_dq) + fwd['dl_dq'][i])
            dl_dv_next = _clip(np.asarray(dl_dv_c))
    finally:
        os.dup2(saved, 2); os.close(saved); os.close(devnull)

    return dl_dq_next, time.time() - t0


def loss_and_grad_fd(x, gt_by_frame, keyframes):
    """Forward-difference gradient over (sx, sy)."""
    sx, sy = x
    loss,    t_fwd = _silent_forward(sx, sy, gt_by_frame, keyframes)
    loss_dx, t_dx  = _silent_forward(sx + FD_EPS, sy, gt_by_frame, keyframes)
    loss_dy, t_dy  = _silent_forward(sx, sy + FD_EPS, gt_by_frame, keyframes)
    g = np.array([(loss_dx - loss) / FD_EPS,
                  (loss_dy - loss) / FD_EPS])
    return loss, g, {
        'mode':    'fd',
        'fwd':     t_fwd,
        'grad_dx': t_dx,
        'grad_dy': t_dy,
        'grad':    t_dx + t_dy,
        'total':   t_fwd + t_dx + t_dy,
    }


def loss_and_grad_bwd(x, gt_by_frame, keyframes, n_substeps=30):
    """Analytical gradient via PyBackward chain.
    Chain rule: q0_slab = cv_slab_local + slab_center, so
    dl_dsx = sum over slab verts i of dl_dq0[3i + 0], likewise dl_dsy."""
    sx, sy = x
    sim, fwd, t_fwd = _silent_forward_full(sx, sy, gt_by_frame, keyframes)
    n_total = len(keyframes) * n_substeps
    dl_dq0, t_bwd = _backward_dq0(sim, fwd, n_substeps, n_total)

    Ns = sim['Ns']  # number of slab verts (slab verts come first in q)
    dl_dq0_xyz = dl_dq0.reshape(-1, 3)
    dl_dsx = float(dl_dq0_xyz[:Ns, 0].sum())
    dl_dsy = float(dl_dq0_xyz[:Ns, 1].sum())
    g = np.array([dl_dsx, dl_dsy])
    return float(fwd['loss']), g, {
        'mode':  'bwd',
        'fwd':   t_fwd,
        'grad':  t_bwd,
        'total': t_fwd + t_bwd,
    }


def loss_and_grad(x, gt_by_frame, keyframes, mode='fd'):
    if mode == 'bwd':
        return loss_and_grad_bwd(x, gt_by_frame, keyframes)
    return loss_and_grad_fd(x, gt_by_frame, keyframes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maxiter', type=int, default=12)
    p.add_argument('--sx-init', type=float, default=0.73)
    p.add_argument('--sy-init', type=float, default=0.115)
    p.add_argument('--step',    type=float, default=0.012,
                   help='initial GD step in metres (max move per iter)')
    p.add_argument('--shrink',  type=float, default=0.5,
                   help='backtracking shrink factor on rejected step')
    p.add_argument('--sx-lb',   type=float, default=0.62)
    p.add_argument('--sx-ub',   type=float, default=0.80)
    p.add_argument('--sy-lb',   type=float, default=0.02)
    p.add_argument('--sy-ub',   type=float, default=0.18)
    p.add_argument('--grad-mode', choices=['fd', 'bwd'], default='fd',
                   help="gradient method: 'fd' (2 forward evals) or 'bwd' "
                        "(PyBackward chain rule on dl_dq0)")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    _log('# slab_center (sx, sy) optimization — GD + backtracking')
    _log(f'# fixed:  E={E_FIXED:.3e}  nu={NU_FIXED:.4f}  '
         f'poke_amp={POKE_AMP:.3f}  ref={REF_SLAB_CENTER}')
    _log(f'# init=({args.sx_init:.4f}, {args.sy_init:.4f})  '
         f'step={args.step:.4f}  shrink={args.shrink}')
    _log(f'# bounds  x∈[{args.sx_lb}, {args.sx_ub}]  '
         f'y∈[{args.sy_lb}, {args.sy_ub}]')

    keyframes = asym_poke_keyframes(
        n_per_cycle=FRAME_END - FRAME_START, poke_amp=POKE_AMP)
    gt_by_frame = load_gt_surfaces(
        FRAME_START, FRAME_END, slab_size=SLAB_SIZE,
        slab_center=REF_SLAB_CENTER)
    _log(f'Loaded {len(gt_by_frame)} GT meshes  '
         f'(frames {FRAME_START}..{FRAME_END})')

    def clip(x):
        return np.array([
            min(max(x[0], args.sx_lb), args.sx_ub),
            min(max(x[1], args.sy_lb), args.sy_ub)])

    x = np.array([args.sx_init, args.sy_init], dtype=np.float64)
    history = []
    t0 = time.time()

    _log(f'# grad_mode={args.grad_mode}')

    loss, g, t = loss_and_grad(x, gt_by_frame, keyframes, mode=args.grad_mode)
    _log(f'\n--- iter   0 ---')
    _log(f'  sx={x[0]:.4f}  sy={x[1]:.4f}  loss={loss:.6f}  '
         f'|grad|={np.linalg.norm(g):.3e}  '
         f'dl_dsx={g[0]:+.3e}  dl_dsy={g[1]:+.3e}')
    _log(f"  [{t['mode']}]  fwd={t['fwd']:.2f}s  "
         f"grad={t['grad']:.2f}s  total={t['total']:.2f}s")
    history.append({'iter': 0, 'x': x.copy(), 'loss': loss, 'grad': g.copy(),
                     'timings': t})

    step = float(args.step)

    for it in range(1, args.maxiter + 1):
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-9:
            _log(f'\n[converged] |grad|={gnorm:.3e} < 1e-9')
            break
        direction = -g / gnorm  # unit step in steepest-descent direction

        # Backtracking: shrink step until loss strictly decreases.
        trial_step = step
        accepted = None
        bt = 0
        bt_fwd_time = 0.0   # accumulated forward time spent on rejected trials
        while bt < 6:
            x_try = clip(x + trial_step * direction)
            l_try, g_try, t_try = loss_and_grad(
                x_try, gt_by_frame, keyframes, mode=args.grad_mode)
            if l_try < loss - 1e-12:
                accepted = (x_try, l_try, g_try, t_try, trial_step, bt)
                break
            bt_fwd_time += t_try['total']
            trial_step *= args.shrink
            bt += 1

        if accepted is None:
            _log(f'\n[stop] backtracking exhausted at iter {it}; '
                 f'|grad|={gnorm:.3e}, step shrunk to {trial_step:.4g}')
            break

        x, loss, g, t, used_step, bt_used = accepted
        _log(f'\n--- iter {it:3d} ---')
        _log(f'  sx={x[0]:.4f}  sy={x[1]:.4f}  loss={loss:.6f}  '
             f'|grad|={np.linalg.norm(g):.3e}  '
             f'dl_dsx={g[0]:+.3e}  dl_dsy={g[1]:+.3e}  '
             f'step={used_step:.4f}  bt={bt_used}')
        if t['mode'] == 'fd':
            _log(f"  [fd]  fwd={t['fwd']:.2f}s  grad={t['grad']:.2f}s  "
                 f"(dx={t['grad_dx']:.2f}s  dy={t['grad_dy']:.2f}s)  "
                 f"total={t['total']:.2f}s  bt_extra={bt_fwd_time:.2f}s")
        else:
            _log(f"  [bwd]  fwd={t['fwd']:.2f}s  grad={t['grad']:.2f}s  "
                 f"total={t['total']:.2f}s  bt_extra={bt_fwd_time:.2f}s")
        history.append({'iter': it, 'x': x.copy(),
                         'loss': loss, 'grad': g.copy(),
                         'step': used_step, 'bt': bt_used,
                         'timings': t,
                         'bt_fwd_time': bt_fwd_time})
        # Grow step slightly if no backtracking needed.
        if bt_used == 0:
            step = min(step * 1.3, args.step * 2.0)

    _log(f'\nOptimization finished in {time.time() - t0:.1f}s')
    _log(f'  final sx={x[0]:.4f}  sy={x[1]:.4f}  loss={loss:.6f}')

    with (OUT_DIR / 'history.pkl').open('wb') as f:
        pickle.dump({
            'history': history, 'result_x': x.copy(),
            'result_loss': float(loss),
            'ref_slab_center': REF_SLAB_CENTER,
            'fixed': {'E': E_FIXED, 'nu': NU_FIXED,
                       'poke_amp': POKE_AMP, 'dt': DT}
        }, f)

    # ---- Plot the descent curve ------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        losses = [h['loss'] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(losses) + 1), losses, 'o-')
        plt.xlabel('iter'); plt.ylabel('loss')
        plt.title('slab_center (sx, sy) optimization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'loss_curve.png', dpi=120)
        plt.close()
        _log(f'Loss curve saved → {OUT_DIR / "loss_curve.png"}')
    except Exception as e:
        _log(f'Plot failed: {e}')


if __name__ == '__main__':
    main()
