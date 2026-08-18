"""Optimize the dice's (E, nu) so that the simulated poke deformation
matches the Atlas mesh ground truth (frames 18..53).

Mirrors plant_3d.py's structure: scipy L-BFGS-B on (log E, log nu) with
gradients coming from DiffPD's PyBackward chain. The sim setup itself is
exactly the one in ur5_poke_demo.py — frame-18 convex-hull rest state,
homogeneous neo-Hookean dice + rigid cylinder, mesh_boundary NCP contact,
per-frame Dirichlet update on the cylinder from FK.

Loss = recon (sim→GT chamfer) + lambda_g · geom (GT→sim chamfer).
The bidirectional chamfer with separable backward serves both as
"reconstruction loss" (sim surface lies on GT) and "geometry loss"
(GT surface is covered by sim).

Run from python/example/:
    python dice_material_optimize.py
"""

import sys
sys.path.append('../')

import time
from pathlib import Path

import numpy as np
import scipy.optimize
import trimesh
from scipy.spatial import cKDTree

from py_diff_pd.common.common import (
    create_folder, ndarray, print_info, print_ok)
from py_diff_pd.common.tet_mesh import generate_tet_mesh
from py_diff_pd.core.py_diff_pd_core import (
    TetMesh3d, TetDeformable, StdRealVector, StdIntVector)

# Reuse the dice + cylinder builders + FK from the demo.
from ur5_poke_demo import (
    ATLAS_BASE, ATLAS_FRAME, ATLAS_ROT_RPY,
    READY_POSE, compute_fk,
    prep_slab_tet, prep_cylinder_tet,
    _extract_boundary_faces, rpy_to_R)

def _out_dir(contact_mode: str) -> Path:
    """Each contact mode has its own output dir (no cross-contamination of
    log / history / videos between mesh_contact and mesh_boundary runs)."""
    suffix = {'mesh_contact': 'mc', 'mesh_boundary': 'mb'}[contact_mode]
    p = Path(f'dice_material_opt_{suffix}')
    p.mkdir(exist_ok=True)
    return p

# These two get RE-BOUND inside main() once the contact_mode CLI flag is
# parsed. Kept at module level so _log_line / loss_and_grad etc. can read
# them without plumbing extra arguments everywhere.
OUT_DIR  = _out_dir('mesh_contact')
LOG_PATH = OUT_DIR / 'optimize.log'


def _log_line(s: str):
    """Append a line to the optimization log AND echo to stdout."""
    print(s)
    with open(LOG_PATH, 'a') as fh:
        fh.write(s + '\n')


# ---------------------------------------------------------------------------
# Asymmetric single-poke waveform matched to the GT atlas timing.
# GT atlas frames 18..53 deform with deepest dent at frame 40 (= sim step 22).
# Down phase 22 frames, up phase 13 frames — NOT a full sine.
# ---------------------------------------------------------------------------
GT_DEEPEST_STEP   = 23         # peak depth at sim step 23
GT_CONTACT_STEP   = 8         # first contact at sim step 13
GT_RECOVER_STEPS  = 35         # one cycle total

# Shoulder-lift offset at which cyl_tip just kisses the slab top, for
# cyl_length=0.10 + 5mm rest gap. slope ≈ 0.526 m/rad → 0.005/0.526.
CONTACT_OFFSET    = 0.0095


def asym_poke_keyframes(n_per_cycle: int = GT_RECOVER_STEPS,
                        poke_amp: float = 0.18,
                        peak_step: int = GT_DEEPEST_STEP,
                        contact_step: int = GT_CONTACT_STEP,
                        contact_offset: float = CONTACT_OFFSET,
                        ready: dict = None):
    """Three-phase smoothstep poke aligned to user-specified timing:

      phase 1 (slow approach)  steps 0..contact_step-1
          shoulder_lift offset: 0 → contact_offset
          (cylinder descends through the rest gap onto the dice surface)
      phase 2 (push into dice) steps contact_step..peak_step-1
          shoulder_lift offset: contact_offset → poke_amp
      phase 3 (release)        steps peak_step..n_per_cycle-1
          shoulder_lift offset: poke_amp → 0

    Each phase uses a cosine smoothstep so velocity is zero at every phase
    boundary — no jerk between approach / push / release.
    """
    if ready is None:
        ready = READY_POSE

    def smoothstep(s):
        s = max(0.0, min(1.0, s))
        return (1.0 - np.cos(np.pi * s)) / 2.0

    kf = []
    for i in range(n_per_cycle):
        if i < contact_step:
            s = i / max(contact_step, 1)
            d = contact_offset * smoothstep(s)
        elif i < peak_step:
            s = (i - contact_step) / max(peak_step - contact_step, 1)
            d = contact_offset + (poke_amp - contact_offset) * smoothstep(s)
        else:
            denom = max(n_per_cycle - 1 - peak_step, 1)
            s = (i - peak_step) / denom
            d = poke_amp * (1.0 - smoothstep(s))
        q = dict(ready)
        q['shoulder_lift_joint'] = ready['shoulder_lift_joint'] + d
        kf.append(q)
    return kf


# ---------------------------------------------------------------------------
# Build the combined deformable for a given (E_dice, nu_dice).
# Returns everything the forward/backward loops need.
# ---------------------------------------------------------------------------
def _auto_slab_center(slab_size, cyl_length, poke_amp=0.15):
    """Aligns slab_center so the dent actually forms at slab center:

      xy  ← cyl_tip xy at PEAK (where the dent is)
      z   ← cyl_tip z at REST minus (size_z/2 + 5mm gap)
            (so the cylinder is 5mm above slab_top at frame 0)

    Without this, slab_center sits under the REST cyl_tip but the cylinder
    drifts +x as it bends during the poke, so the dent forms ~3cm forward
    of the dice center.
    """
    fk_rest = compute_fk(READY_POSE)
    pos_rest, R_rest = fk_rest['ee_link']
    rest_tip_z = (pos_rest + R_rest @ np.array([cyl_length, 0., 0.]))[2]

    q_peak = dict(READY_POSE)
    q_peak['shoulder_lift_joint'] += poke_amp
    fk_peak = compute_fk(q_peak)
    pos_peak, R_peak = fk_peak['ee_link']
    tip_peak = pos_peak + R_peak @ np.array([cyl_length, 0., 0.])

    return (float(tip_peak[0]), float(tip_peak[1]),
            float(rest_tip_z) - slab_size[2] * 0.5 - 0.005)


def build_sim(E_dice: float, nu_dice: float,
              slab_size=(0.30, 0.30, 0.06),
              # Aligned under new READY_POSE cyl_tip (0.791, 0.109, -0.135).
              # slab_top sits 5mm below cyl_tip at rest → center_z = -0.17.
              slab_center=(0.70, 0.089, -0.170),
              cyl_radius=0.01, cyl_length=0.10, cyl_sections=24, # 要改
              # Cylinder is deformable (free interior DOFs so mesh_boundary
               # NCP can solve contact), but extremely stiff (1e13 Pa) so it
               # behaves like a rigid stick — deforms < 1 µm under any
               # realistic poke force.
              cyl_youngs=1e13, nu_share=0.45, rho=800.0,
              # mesh_contact (penalty) tuned with 30× substepping in
              # simulate_and_loss: per-substep cyl motion is ~0.17mm so a
              # 5mm contact_radius gives the dice surface enough margin to
              # track the cylinder side without phantom-contact at rest
              # (5mm > rest gap of 4mm, but the cyl is moving away during
              # rest so transient phantom force damps out quickly).
              contact_mode='mesh_contact',         # or 'mesh_boundary'
              contact_radius=5.5e-4,
              contact_kn_slab=5e3, contact_kn_cyl=5e4,
              # contact_kf adds tangential friction damping at the contact
              # interface (mirrors google_robot's planar_contact kf=50).
              # Helps damp rebound oscillation when cyl is fully embedded.
              contact_kf=50.0, contact_mu=0.5,
              # velocity_damping per-substep tuned so per-second decay
              # matches google_robot (~200/s). 0.80 per dt_sub=1.1ms.
              velocity_damping=0.30):
    if slab_center is None:
        slab_center = _auto_slab_center(slab_size, cyl_length)
    cv_slab, ce_slab, surf_slab, bottom_idx = prep_slab_tet(
        size=slab_size, center=slab_center)
    cv_cyl_local, ce_cyl, surf_cyl = prep_cylinder_tet(
        radius=cyl_radius, length=cyl_length, sections=cyl_sections)

    Ns, Nc = len(cv_slab), len(cv_cyl_local)
    OFF_C = Ns
    n_ts, n_tc = len(ce_slab), len(ce_cyl)

    # Initial FK gives the cylinder its world-space rest pose.
    fk0 = compute_fk(READY_POSE)
    pos_ee0, R_ee0 = fk0['ee_link']
    cv_cyl_world0 = (R_ee0 @ cv_cyl_local.T).T + pos_ee0

    all_v = np.vstack([cv_slab, cv_cyl_world0])
    all_e = np.vstack([ce_slab, ce_cyl + OFF_C])
    slab_surf_g = surf_slab.copy()
    cyl_surf_g  = surf_cyl + OFF_C

    # Material — DICE is what we optimize; cylinder is stiff filler.
    def lame(E, nu):
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return la, mu
    la_s, mu_s = lame(E_dice, nu_dice)
    la_c, mu_c = lame(cyl_youngs, nu_share)
    nh_params = []
    for _ in range(n_ts):
        nh_params.append(float(mu_s)); nh_params.append(float(la_s))
    for _ in range(n_tc):
        nh_params.append(float(mu_c)); nh_params.append(float(la_c))

    tmp_bin = str(OUT_DIR / '_combined.bin')
    generate_tet_mesh(all_v, all_e, tmp_bin)
    mesh = TetMesh3d(); mesh.Initialize(tmp_bin)
    deformable = TetDeformable()
    E_avg = (E_dice + cyl_youngs) / 2.0
    deformable.Initialize(tmp_bin, rho, 'none', E_avg, nu_share)
    deformable.AddPdEnergy('pd_neohookean', nh_params, [])

    # Cylinder cap (x_local ≈ 0) → Dirichlet-driven by FK; the rest of the
    # cylinder is free so mesh_contact has free DOFs on both sides to push
    # against (mirrors google_robot's fingertip with root-only Dirichlet).
    cyl_x_min = cv_cyl_local[:, 0].min()
    cyl_root_local_idx = np.where(
        cv_cyl_local[:, 0] - cyl_x_min < 1e-4)[0].astype(int)

    # Branch on contact_mode.
    slab_surf_int = slab_surf_g.astype(int)
    cyl_surf_int  = cyl_surf_g.astype(int)
    if contact_mode == 'mesh_contact':
        # penalty-based bidirectional contact.
        mesh_params = [
            contact_radius,
            contact_kn_slab,    # kn_left  → dice verts pushing cyl faces
            contact_kn_cyl,     # kn_right → cyl  verts pushing dice faces
            contact_kf,
            contact_mu,
            float(len(slab_surf_int)),
            float(len(cyl_surf_int)),
        ]
        mesh_params.extend(slab_surf_int.ravel().astype(float).tolist())
        mesh_params.extend(cyl_surf_int.ravel().astype(float).tolist())
        deformable.AddStateForce('mesh_contact', mesh_params)
    elif contact_mode == 'mesh_boundary':
        # NCP hard constraint via SetFrictionalBoundary.
        boundary_params = [
            contact_radius, float(Ns),
            float(len(slab_surf_int)), float(len(cyl_surf_int)),
        ]
        boundary_params.extend(slab_surf_int.ravel().astype(float).tolist())
        boundary_params.extend(cyl_surf_int.ravel().astype(float).tolist())
        cand_all = np.unique(np.concatenate([
            np.unique(slab_surf_int.ravel()),
            np.unique(cyl_surf_int.ravel()),
        ])).astype(int)
        # Exclude pinned (Dirichlet) verts from candidates so the NCP
        # solver only resolves contact at free DOFs.
        pinned = (set(int(i) for i in bottom_idx)
                  | set(int(i) + OFF_C for i in cyl_root_local_idx))
        cand = [int(v) for v in cand_all if int(v) not in pinned]
        deformable.SetFrictionalBoundary('mesh', boundary_params, cand)
    else:
        raise ValueError(f'unknown contact_mode {contact_mode!r}')

    def set_all_dirichlet(R_ee, pos_ee):
        for li in bottom_idx:
            target = cv_slab[li]
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * li + k), float(target[k]))
        for li in cyl_root_local_idx:
            target = R_ee @ cv_cyl_local[li] + pos_ee
            gi = int(li) + OFF_C
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
    set_all_dirichlet(R_ee0, pos_ee0)

    return {
        'deformable':    deformable,
        'all_v':         all_v,
        'Ns':            Ns,
        'Nc':            Nc,
        'OFF_C':         OFF_C,
        'cv_slab':       cv_slab,
        'cv_cyl_local':  cv_cyl_local,
        'bottom_idx':    bottom_idx,
        'slab_surf_g':   slab_surf_g,
        'slab_surf_verts': np.unique(slab_surf_g.ravel()).astype(int),
        'n_slab_tets':   n_ts,
        'n_cyl_tets':    n_tc,
        'set_all_dirichlet': set_all_dirichlet,
        'velocity_damping': velocity_damping,
        'slab_center':   slab_center,
        'slab_size':     slab_size,
        'cyl_length':    cyl_length,
    }


# ---------------------------------------------------------------------------
# Load GT mesh-fXXX vertex positions in the SAME world frame as the sim.
# Apply the same rotation / translation / scale that prep_slab_tet uses
# inside ur5_poke_demo, so chamfer compares apples-to-apples.
# ---------------------------------------------------------------------------
def _atlas_to_world_transform(slab_size, slab_center):
    """Replicate prep_slab_tet's transform EXACTLY (load → hull → pyvista
    decimate → centroid/scale/translation from the decimated point cloud),
    then return the parameters needed to lift any GT frame's raw vertices
    into the same world frame the SIM lives in."""
    import os
    import pyvista as pv

    raw = trimesh.load(str(ATLAS_BASE / f'mesh-f{ATLAS_FRAME:05d}.obj'),
                       process=False)
    hull = trimesh.convex.convex_hull(np.asarray(raw.vertices))
    tmp_stl = '/tmp/_atlas_hull_xform.stl'
    hull.export(tmp_stl)
    m = pv.read(tmp_stl)
    m = m.decimate(0.85).clean()
    if os.path.exists(tmp_stl):
        os.remove(tmp_stl)

    pts = np.asarray(m.points, dtype=np.float64)
    centroid_local = pts.mean(axis=0)
    R_atlas = rpy_to_R(*np.deg2rad(ATLAS_ROT_RPY))
    pts_r = (R_atlas @ (pts - centroid_local).T).T + centroid_local
    ext = pts_r.max(axis=0) - pts_r.min(axis=0)
    target_max = max(slab_size[0], slab_size[1])
    scale = target_max / max(ext)
    pts_s = pts_r * scale
    top_z = slab_center[2] + slab_size[2] * 0.5
    pts_max = pts_s.max(axis=0)
    pts_mean_xy = pts_s.mean(axis=0)[:2]
    dx = slab_center[0] - pts_mean_xy[0]
    dy = slab_center[1] - pts_mean_xy[1]
    dz = top_z - pts_max[2]
    return centroid_local, R_atlas, scale, np.array([dx, dy, dz])


def _xform_frame(verts, centroid_local, R_atlas, scale, t_world):
    p = (R_atlas @ (verts - centroid_local).T).T + centroid_local
    p = p * scale + t_world
    return p


def load_gt_surfaces(frame_start: int, frame_end: int,
                     slab_size, slab_center):
    """Returns dict frame_idx → world-frame GT vertex positions.

    Alignment: frame-18 (rest) decimated-hull centroid → slab_center xy.
    All other frames use the SAME world transform, so the dice's natural
    drift in the 4D capture is preserved. Sim dice and GT dice overlap
    fully at rest; they diverge during the poke according to their own
    dynamics — that's the signal the optimizer minimizes.
    """
    centroid, R_atlas, scale, t_world = _atlas_to_world_transform(
        slab_size, slab_center)
    gt = {}
    for f in range(frame_start, frame_end + 1):
        p = ATLAS_BASE / f'mesh-f{f:05d}.obj'
        if not p.exists():
            continue
        m = trimesh.load(str(p), process=False)
        v = _xform_frame(np.asarray(m.vertices, dtype=np.float64),
                          centroid, R_atlas, scale, t_world)
        gt[f] = v
    return gt


# ---------------------------------------------------------------------------
# Chamfer loss (bidirectional) on the dice surface only.
#   recon: sim_surf → nearest GT vert (sim should sit on GT surface)
#   geom : GT vert  → nearest sim_surf vert (GT should be covered)
# Returns scalar loss and a dl/dq[:3Ns] gradient on the FULL state vector
# (cylinder grads are zero — its dofs are Dirichlet-driven).
# ---------------------------------------------------------------------------
def chamfer_loss_and_grad_q(q_full, surf_v_idx, gt_verts, Ns,
                            lambda_recon=1.0, lambda_geom=1.0):
    pts_all = q_full.reshape(-1, 3)
    sim_surf = pts_all[surf_v_idx]
    gt = gt_verts

    # If forward solver diverged we'll see NaN / inf in sim positions —
    # cKDTree barfs on those. Return a finite "barrier" loss + zero
    # gradient so L-BFGS-B simply rejects this trial and backs off.
    if not (np.isfinite(sim_surf).all() and np.isfinite(gt).all()):
        return 1.0, np.zeros_like(q_full), (1.0, 0.0)

    tree_gt  = cKDTree(gt)
    d1, i1   = tree_gt.query(sim_surf, k=1)
    tree_sim = cKDTree(sim_surf)
    d2, i2   = tree_sim.query(gt, k=1)

    n_s, n_g = len(sim_surf), len(gt)
    loss_recon = 0.5 * np.sum(d1 ** 2) / n_s
    loss_geom  = 0.5 * np.sum(d2 ** 2) / n_g
    loss = lambda_recon * loss_recon + lambda_geom * loss_geom

    # Gradients on sim_surf positions.
    grad_surf = lambda_recon * (sim_surf - gt[i1]) / n_s
    # gt → sim: each gt vert pulls its nearest sim vert.
    sim_pull = np.zeros_like(sim_surf)
    np.add.at(sim_pull, i2, sim_surf[i2] - gt)
    grad_surf += lambda_geom * sim_pull / n_g

    # Scatter into the full state grad.
    dl_dq = np.zeros_like(q_full)
    dl_dq_pts = dl_dq.reshape(-1, 3)
    dl_dq_pts[surf_v_idx] += grad_surf
    # Zero out cylinder grads (it's Dirichlet-driven; gradient is taken
    # care of by PyBackward via the contact path).
    dl_dq_pts[Ns:] = 0.0
    return loss, dl_dq, (loss_recon, loss_geom)


# ---------------------------------------------------------------------------
# Forward sim with per-step loss accumulation.
# ---------------------------------------------------------------------------
def simulate_and_loss(sim, keyframes, gt_by_frame, frame_start, dt,
                      method, opt, lambda_recon=1.0, lambda_geom=1.0,
                      n_substeps: int = 30):
    """Substepping forward: each atlas frame is split into `n_substeps`
    PyForward calls of dt/n_substeps. Keyframes are linearly interpolated
    between adjacent atlas keyframes so the cylinder moves smoothly and
    mesh_contact never sees more than ~Δt_sub * cyl_vel ≈ 0.2 mm per
    sub-step (well within contact_radius=0.55mm).

    All sub-step trajectories are kept so backward can replay them.
    Loss is computed only at atlas frame boundaries (matching GT timing).
    """
    deformable = sim['deformable']
    set_all_dirichlet = sim['set_all_dirichlet']
    Ns = sim['Ns']
    surf_v_idx = sim['slab_surf_verts']

    dofs     = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_phys   = sim['all_v'].ravel().copy()
    v_phys   = np.zeros(dofs)
    act      = np.zeros(act_dofs)
    dt_sub   = dt / n_substeps
    n_atlas  = len(keyframes)

    from py_diff_pd.common.common import copy_std_int_vector
    ci = StdIntVector(0)

    # ALL sub-step states are kept so backward can replay.
    q_traj  = [q_phys.copy()]
    v_traj  = [v_phys.copy()]
    ci_traj = [StdIntVector(0)]
    f_ext_traj = []
    # `keyframe_per_substep` records which keyframe drove each forward
    # substep — backward replays the same Dirichlet update.
    kf_per_substep = []

    total_loss = 0.0
    recon_sum = 0.0
    geom_sum  = 0.0
    # dl_dq is injected at atlas frame boundaries — i.e. at indices
    # n_substeps, 2*n_substeps, ... in the substep grid.
    n_total = n_atlas * n_substeps
    dl_dq_traj = [np.zeros(dofs) for _ in range(n_total + 1)]

    def _blend(qa, qb, alpha):
        return {k: (1 - alpha) * qa[k] + alpha * qb[k] for k in qb}

    t0 = time.time()
    for atlas_step in range(n_atlas):
        kf_prev = keyframes[atlas_step - 1] if atlas_step > 0 else keyframes[0]
        kf_curr = keyframes[atlas_step]

        for sub in range(n_substeps):
            alpha = (sub + 1) / n_substeps
            kf_blend = _blend(kf_prev, kf_curr, alpha)
            fk = compute_fk(kf_blend)
            pos_ee, R_ee = fk['ee_link']
            set_all_dirichlet(R_ee, pos_ee)

            f_ext = np.zeros(dofs)
            q_next  = StdRealVector(dofs)
            v_next  = StdRealVector(dofs)
            ci_next = copy_std_int_vector(ci)
            deformable.PyForward(method, q_phys, v_phys, act, f_ext,
                                 dt_sub, opt, q_next, v_next, ci_next)
            q_phys = ndarray(q_next)
            v_phys = ndarray(v_next) * sim['velocity_damping']
            ci     = ci_next

            q_traj.append(q_phys.copy())
            v_traj.append(v_phys.copy())
            ci_traj.append(ci_next)
            f_ext_traj.append(f_ext.copy())
            kf_per_substep.append(kf_blend)

            if not (np.isfinite(q_phys).all() and np.isfinite(v_phys).all()):
                return {
                    'loss': 1.0, 'recon': 1.0, 'geom': 0.0,
                    'q': q_traj, 'v': v_traj, 'ci': ci_traj,
                    'f_ext': f_ext_traj, 'kf_per_substep': kf_per_substep,
                    'dl_dq': dl_dq_traj, 'n_substeps': n_substeps,
                    'forward_time': time.time() - t0,
                    'diverged_at_step': len(q_traj) - 1,
                }

        # END of atlas frame — accumulate chamfer loss at the atlas boundary
        # (dl_dq_traj is indexed by substep, so injection goes at the LAST
        # substep state of this atlas frame).
        gt_idx = frame_start + atlas_step + 1
        if gt_idx in gt_by_frame:
            loss_i, grad_qi, (lr, lg) = chamfer_loss_and_grad_q(
                q_phys, surf_v_idx, gt_by_frame[gt_idx], Ns,
                lambda_recon, lambda_geom)
            total_loss += loss_i
            recon_sum  += lr
            geom_sum   += lg
            dl_dq_traj[(atlas_step + 1) * n_substeps] += grad_qi

    fwd_time = time.time() - t0
    return {
        'loss':        total_loss,
        'recon':       recon_sum,
        'geom':        geom_sum,
        'q':           q_traj,
        'v':           v_traj,
        'ci':          ci_traj,
        'f_ext':       f_ext_traj,
        'kf_per_substep': kf_per_substep,
        'n_substeps':  n_substeps,
        'dl_dq':       dl_dq_traj,
        'forward_time': fwd_time,
    }


# ---------------------------------------------------------------------------
# Adjoint backward chain → dl/d(E, nu) for the DICE tets only.
# ---------------------------------------------------------------------------
def backward_material_grad(sim, keyframes, fwd, dt, method, opt_cpp,
                           E_dice, nu_dice, grad_clip_norm: float = 1e3):
    """Reverse adjoint chain over `n_steps` PyBackward calls, with per-step
    L2-clip on (dl_dq_next, dl_dv_next).

    Matches env_base.simulate's backward structure exactly (line 339-371),
    plus the same clip on adjoints (line 327-337) so a partially-converged
    forward step cannot inflate dl/dq_next into inf across the chain."""
    deformable = sim['deformable']
    set_all_dirichlet = sim['set_all_dirichlet']

    dofs     = deformable.dofs()
    act_dofs = deformable.act_dofs()
    mat_w_dofs = deformable.NumOfPdElementEnergies()
    act_w_dofs = deformable.NumOfPdMuscleEnergies()
    state_p_dofs = deformable.NumOfStateForceParameters()

    def _clip_adj(g):
        if grad_clip_norm is None:
            return g
        g = np.nan_to_num(g, nan=0.0,
                          posinf=grad_clip_norm, neginf=-grad_clip_norm)
        n = np.linalg.norm(g)
        if n > grad_clip_norm:
            g = g * (grad_clip_norm / n)
        return g

    # Substep-aware backward. fwd['q'][i] for i in 0..n_total covers every
    # substep state. PyBackward uses dt_sub between consecutive substep
    # states; dl_dq is injected only at atlas frame boundaries (substep
    # indices that are multiples of n_substeps).
    n_substeps = fwd['n_substeps']
    n_total    = len(keyframes) * n_substeps
    dt_sub     = dt / n_substeps
    kf_per_substep = fwd['kf_per_substep']

    dl_dmat_w   = np.zeros(mat_w_dofs)
    dl_dstate_p = np.zeros(state_p_dofs)   # mesh_contact's [cr, kn_l, kn_r, kf, mu, ...] grads
    dl_dq_next = _clip_adj(fwd['dl_dq'][n_total].copy())
    dl_dv_next = np.zeros(dofs)

    t0 = time.time()
    for i in reversed(range(n_total)):
        # Restore Dirichlet for substep i (matches forward exactly).
        kf_blend = kf_per_substep[i]
        fk = compute_fk(kf_blend)
        pos_ee, R_ee = fk['ee_link']
        set_all_dirichlet(R_ee, pos_ee)

        dl_dq   = StdRealVector(dofs)
        dl_dv_c = StdRealVector(dofs)
        dl_da   = StdRealVector(act_dofs)
        dl_df   = StdRealVector(dofs)
        dl_dmw  = StdRealVector(mat_w_dofs)
        dl_daw  = StdRealVector(act_w_dofs)
        dl_dsp  = StdRealVector(state_p_dofs)

        act_i  = np.zeros(act_dofs)
        f_ext_i = fwd['f_ext'][i]
        deformable.PyBackward(
            method, fwd['q'][i], fwd['v'][i], act_i, f_ext_i, dt_sub,
            fwd['q'][i + 1], fwd['v'][i + 1], fwd['ci'][i + 1],
            dl_dq_next, dl_dv_next,
            opt_cpp, dl_dq, dl_dv_c, dl_da, dl_df, dl_dmw, dl_daw, dl_dsp)

        # Carry to the previous step: chain term (PyBackward output) +
        # stepwise loss at q[i] (chamfer applied at forward step i).
        dl_dq_next = _clip_adj(ndarray(dl_dq) + fwd['dl_dq'][i])
        dl_dv_next = _clip_adj(ndarray(dl_dv_c))
        dl_dmat_w   += ndarray(dl_dmw)
        dl_dstate_p += ndarray(dl_dsp)

    bwd_time = time.time() - t0

    # Sum dice-tet entries only. mat_w layout = [mu_0, la_0, mu_1, la_1, ...]
    # First 2 * n_slab_tets entries belong to the dice.
    n_dice = sim['n_slab_tets']
    dl_dmu = float(dl_dmat_w[0:2 * n_dice:2].sum())
    dl_dla = float(dl_dmat_w[1:2 * n_dice:2].sum())

    # Chain rule (mu, la) → (E, nu).  Same formulas as env_base._material_jacobian:
    #   d(mu)/dE  = 1/(2(1+nu))
    #   d(la)/dE  = nu / ((1+nu)(1-2nu))
    #   d(mu)/dnu = -E / (2(1+nu)²)
    #   d(la)/dnu = E (1 + 2nu²) / ((1+nu)(1-2nu))²
    E, nu = E_dice, nu_dice
    dmu_dE  = 1.0 / (2 * (1 + nu))
    dla_dE  = nu / ((1 + nu) * (1 - 2 * nu))
    dmu_dnu = -E / (2 * (1 + nu) ** 2)
    dla_dnu = E * (1 + 2 * nu * nu) / (((1 + nu) * (1 - 2 * nu)) ** 2)

    dl_dE  = dl_dmu * dmu_dE  + dl_dla * dla_dE
    dl_dnu = dl_dmu * dmu_dnu + dl_dla * dla_dnu

    # mesh_contact state-force parameters layout (DiffPD C++):
    #   [0] contact_radius
    #   [1] kn_left   (= contact_kn_slab)
    #   [2] kn_right  (= contact_kn_cyl)
    #   [3] kf
    #   [4] mu
    #   [5..] face counts + indices (integers; grad is meaningless)
    # For mesh_boundary the layout is shorter ([radius, split, n_a, n_b, faces])
    # and only [0] = radius is differentiable.
    dl_dcr      = float(dl_dstate_p[0]) if state_p_dofs >= 1 else 0.0
    dl_dkn_slab = float(dl_dstate_p[1]) if state_p_dofs >= 2 else 0.0
    dl_dkn_cyl  = float(dl_dstate_p[2]) if state_p_dofs >= 3 else 0.0
    dl_dkf      = float(dl_dstate_p[3]) if state_p_dofs >= 4 else 0.0
    dl_dmu_c    = float(dl_dstate_p[4]) if state_p_dofs >= 5 else 0.0

    return {
        'dl_dE':       dl_dE,
        'dl_dnu':      dl_dnu,
        'dl_dcr':      dl_dcr,
        'dl_dkn_slab': dl_dkn_slab,
        'dl_dkn_cyl':  dl_dkn_cyl,
        'dl_dkf':      dl_dkf,
        'dl_dmu_c':    dl_dmu_c,
    }, bwd_time


# ---------------------------------------------------------------------------
# Top-level: simulate + (optionally) backprop for one (E, nu).
# ---------------------------------------------------------------------------
THREAD_CT = 8
METHOD = 'pd_eigen_alg4'
OPT_FWD = {
    'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9,
    'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': THREAD_CT,
    'use_bfgs': 0, 'use_acc': 1, 'use_sparse': 0,
    'project_newton': 1, 'use_abs': 1, 'aa_window': 5,
    # Backward adjoint clamp — avoids inf in the chain when forward only
    # converged partially. See env_base.simulate() around line 327.
    'backward_grad_clip_norm': 1e3,
}


FD_EPS_LOG = 1e-2     # finite-diff step in log space for damping & poke_amp


def _silent_forward_loss(E, nu, contact_mode, cr, kn_s, kn_c, kf, mu_c,
                          damping, poke_amp, gt_by_frame, frame_start, dt,
                          lambda_recon, lambda_geom, slab_center=None):
    """Forward-only pass: returns scalar loss for FD gradients of damping
    & poke_amp (parameters DiffPD's backward can't see)."""
    import os
    build_kwargs = dict(
        E_dice=E, nu_dice=nu, contact_mode=contact_mode,
        contact_radius=cr,
        contact_kn_slab=kn_s, contact_kn_cyl=kn_c,
        contact_kf=kf, contact_mu=mu_c,
        velocity_damping=damping)
    if slab_center is not None:
        build_kwargs['slab_center'] = tuple(slab_center)
    sim = build_sim(**build_kwargs)
    kfs = asym_poke_keyframes(poke_amp=poke_amp)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        fwd = simulate_and_loss(sim, kfs, gt_by_frame, frame_start, dt,
                                METHOD, OPT_FWD,
                                lambda_recon=lambda_recon,
                                lambda_geom=lambda_geom)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr); os.close(devnull)
    return fwd['loss']


def loss_and_grad(x, gt_by_frame, keyframes, frame_start, dt,
                  lambda_recon=1.0, lambda_geom=0.5, require_grad=True,
                  contact_mode='mesh_contact', joint=False,
                  slab_center=None):
    """If joint=False:  x = [log E, log nu]  (2D).
    If joint=True:   x = [log E, log nu, log cr, log kn_slab, log kn_cyl,
                          log kf, log mu_c, log damping, log poke_amp]  (9D).
                    damping & poke_amp use forward-diff (DiffPD's backward
                    can't see them); the other 7 use analytical gradients.
    Returns (loss, grad, fwd_info).
    """
    import os
    E   = float(np.exp(x[0]))
    nu  = float(np.exp(x[1]))
    sc_kw = ({'slab_center': tuple(slab_center)}
             if slab_center is not None else {})
    if joint:
        cr        = float(np.exp(x[2]))
        kn_s      = float(np.exp(x[3]))
        kn_c      = float(np.exp(x[4]))
        kf        = float(np.exp(x[5]))
        mu_c      = float(np.exp(x[6]))
        damping   = float(np.exp(x[7]))
        poke_amp  = float(np.exp(x[8]))
        # Regenerate keyframes for the current poke_amp.
        keyframes = asym_poke_keyframes(poke_amp=poke_amp)
        sim = build_sim(E_dice=E, nu_dice=nu, contact_mode=contact_mode,
                        contact_radius=cr,
                        contact_kn_slab=kn_s, contact_kn_cyl=kn_c,
                        contact_kf=kf, contact_mu=mu_c,
                        velocity_damping=damping, **sc_kw)
    else:
        sim = build_sim(E_dice=E, nu_dice=nu, contact_mode=contact_mode,
                        **sc_kw)

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        fwd = simulate_and_loss(sim, keyframes, gt_by_frame, frame_start,
                                dt, METHOD, OPT_FWD,
                                lambda_recon=lambda_recon,
                                lambda_geom=lambda_geom)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)
    loss = fwd['loss']
    if not require_grad:
        return loss, None, fwd

    if 'diverged_at_step' in fwd:
        _log_line(f'  E={E:.4e}  nu={nu:.4f}  '
                  f'FORWARD DIVERGED at step {fwd["diverged_at_step"]}  '
                  f'→ returning barrier loss=1.0, zero grad')
        return loss, np.zeros(len(x)), fwd

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        grads, bwd_time = backward_material_grad(
            sim, keyframes, fwd, dt, METHOD, OPT_FWD, E, nu)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)
    # x = log(theta) ⇒ d(loss)/d(x_i) = d(loss)/d(theta_i) * theta_i.
    if joint:
        # Forward-diff for damping & poke_amp (NOT visible to PyBackward).
        # eps in log-space: theta * (exp(eps) - 1) ≈ theta * eps when eps small.
        eps = FD_EPS_LOG
        loss_d = _silent_forward_loss(
            E, nu, contact_mode, cr, kn_s, kn_c, kf, mu_c,
            damping*np.exp(eps), poke_amp, gt_by_frame, frame_start, dt,
            lambda_recon, lambda_geom, slab_center=slab_center)
        loss_p = _silent_forward_loss(
            E, nu, contact_mode, cr, kn_s, kn_c, kf, mu_c,
            damping, poke_amp*np.exp(eps), gt_by_frame, frame_start, dt,
            lambda_recon, lambda_geom, slab_center=slab_center)
        # d(loss)/d(log θ) ≈ (loss_pert - loss_base) / eps.
        dl_dlog_damping  = (loss_d - loss) / eps
        dl_dlog_pokeamp  = (loss_p - loss) / eps

        grad = np.array([
            grads['dl_dE']       * E,
            grads['dl_dnu']      * nu,
            grads['dl_dcr']      * cr,
            grads['dl_dkn_slab'] * kn_s,
            grads['dl_dkn_cyl']  * kn_c,
            grads['dl_dkf']      * kf,
            grads['dl_dmu_c']    * mu_c,
            dl_dlog_damping,                       # already in log space (FD)
            dl_dlog_pokeamp,
        ])
        _log_line(
            f'  E={E:.3e}  nu={nu:.3f}  cr={cr*1000:.2f}mm  '
            f'kn_s={kn_s:.2e}  kn_c={kn_c:.2e}  kf={kf:.1f}  mu={mu_c:.3f}  '
            f'damp={damping:.3f}  amp={poke_amp:.3f}  '
            f'loss={loss:.6f}  '
            f'|grad|={np.linalg.norm(grad):.3e}  '
            f'fwd={fwd["forward_time"]:.1f}s  bwd={bwd_time:.1f}s')
    else:
        grad = np.array([grads['dl_dE'] * E, grads['dl_dnu'] * nu])
        _log_line(
            f'  E={E:.4e}  nu={nu:.4f}  '
            f'loss={loss:.6f}  recon={fwd["recon"]:.6f}  geom={fwd["geom"]:.6f}  '
            f'|grad|={np.linalg.norm(grad):.3e}  '
            f'dl_dE={grads["dl_dE"]:+.3e}  dl_dnu={grads["dl_dnu"]:+.3e}  '
            f'fwd={fwd["forward_time"]:.1f}s  bwd={bwd_time:.1f}s')
    return loss, grad, fwd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--contact-mode',
                    choices=['mesh_contact', 'mesh_boundary'],
                    default='mesh_contact',
                    help='mesh_contact = penalty force; '
                         'mesh_boundary = NCP hard constraint. '
                         'Each writes to its own dice_material_opt_<mc|mb>/ dir.')
    ap.add_argument('--no-render', action='store_true',
                    help='Skip the final sim+GT video render after optimization')
    ap.add_argument('--maxiter', type=int, default=30,
                    help='Max L-BFGS-B outer iterations.')
    ap.add_argument('--maxfun', type=int, default=200,
                    help='Max func evals. Joint mode does 3 fwds/iter via FD; '
                         'set generously (≥ 3 * maxiter).')
    ap.add_argument('--ftol', type=float, default=1e-4,
                    help='Stop when relative loss change < ftol. '
                         'Looser (1e-2) = fewer iters; tighter (1e-5) = more.')
    ap.add_argument('--gtol', type=float, default=1e-6,
                    help='Stop when |grad|_inf < gtol.')
    ap.add_argument('--maxls', type=int, default=20,
                    help='Max line-search steps per iter. Bump if noisy.')
    ap.add_argument('--maxcor', type=int, default=15,
                    help='L-BFGS Hessian-pair history. Higher = better for '
                         '>5 vars; default 15 covers our 9-D joint mode.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--joint', action='store_true',
                    help='Also jointly optimize contact (cr, kn_slab, kn_cyl, '
                         'kf, mu) + sim params (damping, poke_amp) — 9 vars.')
    ap.add_argument('--warm-start', action='store_true',
                    help='If history.pkl exists in OUT_DIR, use its result_x '
                         'as x_init (extends to 9-D in joint mode by padding '
                         'remaining vars with their defaults). Useful for '
                         '(E,nu)-then-(joint) staged optimization.')
    ap.add_argument('--warm-from', type=str, default=None,
                    help='Explicit history.pkl path to warm-start from. '
                         'Overrides --warm-start auto-detection. Example: '
                         'multi_start/seed_007/history.pkl')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Override the default output dir '
                         '(dice_material_opt_<mc|mb>). Stage 2 → keep stage 1 '
                         'results: --out-dir dice_material_opt_mc_stage2')
    ap.add_argument('--sim-slab-center', type=float, nargs=3,
                    metavar=('SX', 'SY', 'SZ'), default=None,
                    help='Fix the dice rest center used by build_sim, '
                         'while GT keeps the reference anchor (0.70, '
                         '0.089, -0.170). Use after an xy-opt to lock in '
                         'the best slab_center, e.g. '
                         '"--sim-slab-center 0.6999 0.0761 -0.170".')
    args = ap.parse_args()
    if args.joint and args.contact_mode != 'mesh_contact':
        raise SystemExit('--joint only supported for --contact-mode mesh_contact')

    # Rebind module globals so all logging / saving lands in the right dir.
    global OUT_DIR, LOG_PATH
    canonical_dir = _out_dir(args.contact_mode)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUT_DIR = canonical_dir
    LOG_PATH = OUT_DIR / 'optimize.log'

    seed = args.seed
    np.random.seed(seed)

    # Fresh log file per run.
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    _log_line(f'# dice material optimization log  (seed={seed})')
    _log_line(f'# header: iter  E  nu  loss  recon  geom  |grad|  '
              f'dl_dE  dl_dnu  fwd_time  bwd_time')

    # ---- 1) Build keyframes for the poke cycle (atlas 18..53 = 35 frames).
    frame_start = ATLAS_FRAME       # 18
    frame_end   = 53
    n_per_cycle = frame_end - frame_start   # 35
    # Asymmetric single-poke aligned to GT: peak at sim step 22 (= atlas
    # frame 40), poke_amp tuned so cyl_tip reaches ~z=0.14 (3cm below the
    # deepest GT dent at z=0.18).
    keyframes = asym_poke_keyframes(n_per_cycle=n_per_cycle, poke_amp=0.18) 
    dt = 1.0 / 30.0
    print_info(f'Atlas frames {frame_start}..{frame_end}  ({n_per_cycle} steps)')

    # ---- 2) Load GT meshes for those atlas frames.
    slab_size_default = (0.30, 0.30, 0.06)
    sc_auto = _auto_slab_center(slab_size_default, cyl_length=0.10) #要改
    slab_center_default = (0.70, 0.089, -0.170)
    gt_by_frame = load_gt_surfaces(frame_start, frame_end,
                                    slab_size=slab_size_default,
                                    slab_center=slab_center_default)
    if args.sim_slab_center is not None:
        print_info(f'sim slab_center OVERRIDDEN to {tuple(args.sim_slab_center)} '
                   f'(GT anchor stays at {slab_center_default})')
        _log_line(f'sim slab_center={tuple(args.sim_slab_center)}  '
                  f'gt_anchor={slab_center_default}')
    print_info(f'Loaded {len(gt_by_frame)} GT meshes (frames '
               f'{min(gt_by_frame.keys())}..{max(gt_by_frame.keys())})')

    # ---- 3) Optimization bounds.
    # E:        [2e5, 2e6]   stay above PD stability floor
    # nu:       [0.25, 0.42] avoid (1-2ν) singularity at 0.5
    # cr:       [1e-3, 2e-2] 1mm — 20mm
    # kn_slab:  [1e3, 1e6]
    # kn_cyl:   [1e3, 1e6]
    if args.joint:
        # 9-D bounds: E, nu, cr, kn_s, kn_c, kf, mu, damping, poke_amp
        # cr/damping lower bounds widened so build_sim defaults (stage-1
        # operating point: cr=0.55mm, damp=0.30) are in-bounds.
        x_lb = ndarray([np.log(2e5),  np.log(0.25),
                        np.log(1e-4), np.log(1e3), np.log(1e3),
                        np.log(1.0),  np.log(0.05),
                        np.log(0.20), np.log(0.05)])
        x_ub = ndarray([np.log(2e6),  np.log(0.42),
                        np.log(5e-2), np.log(1e6), np.log(1e6),
                        np.log(500.), np.log(2.0),
                        np.log(0.99), np.log(0.30)])
        # Start at build_sim's hardcoded defaults — the operating point at
        # which stage 1 (2-var E/ν) was tuned. Lets stage 2 warm-start
        # reproduce stage 1's loss before exploring 9D.
        x_init = ndarray([np.log(5e5),    np.log(0.40),
                          np.log(5.5e-4), np.log(5e3), np.log(5e4),
                          np.log(50.),    np.log(0.5),
                          np.log(0.30),   np.log(0.18)])
        # Warm-start: pull whatever optimal vars exist in prev history.pkl.
        if args.warm_start or args.warm_from:
            if args.warm_from:
                hpkl = Path(args.warm_from)
            elif (OUT_DIR / 'history.pkl').exists():
                hpkl = OUT_DIR / 'history.pkl'
            else:
                hpkl = canonical_dir / 'history.pkl'
            if hpkl.exists():
                import pickle as _pk
                prev = _pk.loads(hpkl.read_bytes())
                px = prev['result_x']
                n = min(len(px), len(x_init))
                x_init[:n] = px[:n]
                _log_line(f'warm-start: read {n} vars from {hpkl}')
            else:
                _log_line(f'warm-start: {hpkl} not found, using defaults')
        print_info(f'init  E={np.exp(x_init[0]):.3e}  nu={np.exp(x_init[1]):.4f}  '
                   f'cr={np.exp(x_init[2])*1000:.2f}mm  '
                   f'kn_s={np.exp(x_init[3]):.2e}  kn_c={np.exp(x_init[4]):.2e}  '
                   f'kf={np.exp(x_init[5]):.1f}  mu={np.exp(x_init[6]):.3f}  '
                   f'damp={np.exp(x_init[7]):.3f}  amp={np.exp(x_init[8]):.3f}')
    else:
        x_lb = ndarray([np.log(2e5), np.log(0.25)])
        x_ub = ndarray([np.log(2e6), np.log(0.42)])
        x_init = np.random.uniform(x_lb, x_ub)
        if args.warm_start or args.warm_from:
            if args.warm_from:
                hpkl = Path(args.warm_from)
            elif (OUT_DIR / 'history.pkl').exists():
                hpkl = OUT_DIR / 'history.pkl'
            else:
                hpkl = canonical_dir / 'history.pkl'
            if hpkl.exists():
                import pickle as _pk
                prev = _pk.loads(hpkl.read_bytes())
                px = prev['result_x']
                x_init[:min(2, len(px))] = px[:min(2, len(px))]
                _log_line(f'warm-start: read {min(2, len(px))} vars from {hpkl}')
        print_info(f'init  E={np.exp(x_init[0]):.3e}  nu={np.exp(x_init[1]):.4f}')

    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    history = []
    iter_counter = {'i': 0}
    def wrap(x):
        iter_counter['i'] += 1
        _log_line(f'\n--- iter {iter_counter["i"]:3d} ---')
        loss, grad, _ = loss_and_grad(
            x, gt_by_frame, keyframes, frame_start, dt,
            contact_mode=args.contact_mode, joint=args.joint,
            slab_center=args.sim_slab_center)
        history.append({'iter': iter_counter['i'], 'x': x.copy(),
                        'loss': loss, 'grad': grad.copy()})
        return loss, grad

    t0 = time.time()
    result = scipy.optimize.minimize(
        wrap, np.copy(x_init), method='L-BFGS-B', jac=True, bounds=bounds,
        options={
            'ftol':    args.ftol,
            'gtol':    args.gtol,
            'maxiter': args.maxiter,
            'maxfun':  args.maxfun,
            'maxls':   args.maxls,
            'maxcor':  args.maxcor,
            'disp':    False,
        })
    _log_line(f'  L-BFGS-B termination: {result.message}')
    _log_line(f'  iters: {result.nit}  func_evals: {result.nfev}')
    t1 = time.time()

    _log_line(f'\nOptimization finished in {t1 - t0:.1f}s')
    _log_line(f'  final E   = {np.exp(result.x[0]):.4e}')
    _log_line(f'  final nu  = {np.exp(result.x[1]):.4f}')
    if args.joint:
        _log_line(f'  final cr      = {np.exp(result.x[2])*1000:.2f}mm')
        _log_line(f'  final kn_slab = {np.exp(result.x[3]):.4e}')
        _log_line(f'  final kn_cyl  = {np.exp(result.x[4]):.4e}')
        _log_line(f'  final kf      = {np.exp(result.x[5]):.4e}')
        _log_line(f'  final mu      = {np.exp(result.x[6]):.4f}')
        _log_line(f'  final damping = {np.exp(result.x[7]):.4f}')
        _log_line(f'  final amp     = {np.exp(result.x[8]):.4f}')
    _log_line(f'  final loss = {result.fun:.6f}')

    import pickle
    with open(OUT_DIR / 'history.pkl', 'wb') as f:
        pickle.dump({'history': history, 'result_x': result.x,
                     'result_fun': result.fun, 'joint': args.joint}, f)

    # ---- 5) Auto-render loss curves + sim-vs-GT trajectory video.
    if not args.no_render:
        try:
            from dice_vis import plot_curves, render_trajectory
            _log_line('\n--- Rendering loss curves + trajectory ---')
            plot_curves(OUT_DIR / 'history.pkl',
                        OUT_DIR / 'loss_curves.png')
            E_final  = float(np.exp(result.x[0]))
            nu_final = float(np.exp(result.x[1]))
            kw = {'contact_mode': args.contact_mode, 'out_dir': OUT_DIR}
            if args.joint:
                kw['contact_radius'] = float(np.exp(result.x[2]))
                kw['contact_kn']     = float(np.exp(result.x[3]))  # avg of slab/cyl
            render_trajectory(E_final, nu_final, sample=16, fps=20, **kw)
        except Exception as e:
            _log_line(f'  render skipped: {e}')


if __name__ == '__main__':
    main()
