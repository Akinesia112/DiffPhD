"""
Google Robot arm — FK animation + DiffPD soft-body grasping simulation.

Architecture (rewritten to mirror gatorman_dual_env contact energy):
  * One TetDeformable holds three bodies, in a single global vertex numbering:
        body 0 : soft ball     (heterogeneous A-B-A: hard / soft / hard
                                 along the gripper closing axis)
        body 1 : left fingertip (convex-hull tetrahedralised, uniform stiff
                                 pd_neohookean)
        body 2 : right fingertip (same as body 1)
  * Each fingertip's "root face" (the proximal cap that connects to
    link_finger_*) is held by per-step Dirichlet BCs whose target xyz is
    recomputed every frame from the URDF FK.  The C++ side updates only
    `dirichlet_` values (no new dof keys), so the PD prefactor is preserved.
  * Contact between the ball surface and BOTH fingertip surfaces is handled
    by a single mesh_contact StateForce
        faces_a = ball surface,
        faces_b = concat(L fingertip surface, R fingertip surface).
    Forces are part of the implicit solve, fixing the convergence problems
    and the visual gap that came from external f_ext.
  * Gravity is a state force, on for the entire physics phase.
  * The rest of the arm (torso..link_finger_*) stays a kinematic FK rigid
    mesh — used only for rendering, never for contact.

Keyframe sequence (N_PRESKIP=6 fast-approach + n_descent slow descent + 30 action):
  0-5          : fast approach (not rendered)
  6..+n_descent: slow descent
  +0..+9       : finger close
  +10..+19     : arm lift
  +20..+29     : finger open
  +30..end     : gravity / fall

Run from python/example/:
    python google_robot_static_mesh_contact.py \\
        --ball-youngs-hard 5e4 --ball-youngs-soft 5e3 \\
        --fingertip-youngs 1e6 \\
        --contact-radius 1.5e-3 \\
        --contact-kn-ball 1e4 --contact-kn-tip 1e4 \\
        --dt 1e-3 --method pd_eigen_alg4 --max-iter 5000 --rel-tol 1e-4 \\
        --finger-close 0.45 --n-descent 10 --n-gravity 60

Minimal run (all defaults):
    cd python/example/
    python google_robot_static_mesh_contact.py

Common flags:
  Ball:        --ball-radius 0.04  --ball-subdiv 3
               --ball-youngs-hard 5e4  --ball-youngs-soft 5e3
               --ball-soft-low-q 0.333  --ball-soft-high-q 0.667
               --grasp-z-offset 0.0
  Fingertip:   --fingertip-youngs 1e6
  Contact:     --contact-radius 1.5e-3
               --contact-kn-ball 1e4  --contact-kn-tip 1e4
               --contact-kf 0.0  --contact-mu 0.5
  Animation:   --n-descent 10  --n-gravity 60  --finger-close 0.45
               --physics-start <int>   (default: N_PRESKIP + n_descent)
  Solver:      --method pd_eigen_alg4  (or pd_eigen / newton_pcg / newton_cholesky)
               --max-iter 5000  --rel-tol 1e-4  --dt 1e-3  --threads 8
  Other:       --no-vis   (skip rendering, sim only)
"""

import sys
sys.path.append('../')

import os
import shutil
import subprocess
import time
import numpy as np
import trimesh
from pathlib import Path

from py_diff_pd.common.renderer import PbrtRenderer

# ---- Directories ----------------------------------------------------------
BASE = (Path(__file__).resolve().parent.parent.parent
        / 'asset' / 'mesh' / 'google_robot')
MESH_3OBJ   = BASE / '3obj_recolor_tabletop_visual_matching_1'
MESH_COARSE = BASE / 'coarse_meshes'

OUT_DIR = Path('google_robot_grip_anim_mesh_contact')
OUT_DIR.mkdir(exist_ok=True)

# ---- FK helpers -----------------------------------------------------------
def rpy_to_R(roll, pitch, yaw):
    cx, sx = np.cos(roll),  np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def axis_angle_to_R(axis, angle):
    ax = np.array(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = ax
    return np.array([
        [t*x*x + c,     t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x,  t*z*z + c  ],
    ])

I3 = np.eye(3)

# ---- URDF joint chain -----------------------------------------------------
_JOINT_CHAIN = [
    ('link_torso',           'base',              [0., 0., 0.5945],              [0,0,0],         [0,0,1]),
    ('link_shoulder',        'link_torso',         [0.,-0.16728852885, 0.113],    [0,0,0],         [0,1,0]),
    ('link_bicep',           'link_shoulder',      [0., 2.289e-3, 0.21502852885], [0,0,0],         [0,0,1]),
    ('link_elbow',           'link_bicep',         [0.,-0.064, 0.1849715],        [0,0,0],         [0,1,0]),
    ('link_forearm',         'link_elbow',         [0.,-0.101, 9.15e-2],          [0,0,0],         [0,0,1]),
    ('link_wrist',           'link_forearm',       [0.,-6.04e-3, 0.2735],         [0,0,0],         [0,1,0]),
    ('link_gripper',         'link_wrist',         [0., 6.038e-3, 0.10911],       [0,0,0],         [0,0,1]),
    ('link_finger_right',    'link_gripper',       [0.,-0.025, 5.861e-2],         [-1.15,0,np.pi], [1,0,0]),
    ('link_finger_tip_right','link_finger_right',  [0.,-1.03567e-2, 6.41556e-2],  [0.45,0,0],      None),
    ('link_finger_left',     'link_gripper',       [0., 0.025, 5.861e-2],         [-1.15,0,0],     [1,0,0]),
    ('link_finger_tip_left', 'link_finger_left',   [0.,-1.03567e-2, 6.41556e-2],  [0.45,0,0],      None),
]

_JOINT_NAME = {
    'link_torso':        'joint_torso',
    'link_shoulder':     'joint_shoulder',
    'link_bicep':        'joint_bicep',
    'link_elbow':        'joint_elbow',
    'link_forearm':      'joint_forearm',
    'link_wrist':        'joint_wrist',
    'link_gripper':      'joint_gripper',
    'link_finger_right': 'joint_finger_right',
    'link_finger_left':  'joint_finger_left',
}

_VISUAL_RPY = {'link_gripper': [0., 0., -np.pi/2]}

def compute_fk(q: dict) -> dict:
    frames = {'base': (np.zeros(3), I3.copy())}
    for child, parent, xyz, jrpy, axis in _JOINT_CHAIN:
        p_pos, p_R = frames[parent]
        pos = p_pos + p_R @ np.array(xyz)
        R_fixed = rpy_to_R(*jrpy)
        if axis is not None:
            qa = q.get(_JOINT_NAME.get(child, ''), 0.)
            R = p_R @ R_fixed @ axis_angle_to_R(axis, qa)
        else:
            R = p_R @ R_fixed
        frames[child] = (pos, R)
    return frames

# ---- Mesh paths -----------------------------------------------------------
def mp(d, n):
    p = d / n; return p if p.exists() else None

_LINK_MESHES = {
    'link_torso':            mp(MESH_COARSE, 'link_torso.obj'),
    'link_shoulder':         mp(MESH_3OBJ,   'link_shoulder.3obj.obj'),
    'link_bicep':            mp(MESH_3OBJ,   'link_bicep.3obj.obj'),
    'link_elbow':            mp(MESH_3OBJ,   'link_elbow.3obj.obj'),
    'link_forearm':          mp(MESH_3OBJ,   'link_forearm.3obj.obj'),
    'link_wrist':            mp(MESH_3OBJ,   'link_wrist.3obj.obj'),
    'link_gripper':          mp(MESH_3OBJ,   'link_gripper.3obj.obj'),
    'link_finger_right':     mp(MESH_3OBJ,   'link_finger_base_simplified.3obj.obj'),
    'link_finger_tip_right': mp(MESH_3OBJ,   'link_finger_tip_simplified.3obj.obj'),
    'link_finger_left':      mp(MESH_3OBJ,   'link_finger_base_simplified.3obj.obj'),
    'link_finger_tip_left':  mp(MESH_3OBJ,   'link_finger_tip_simplified.3obj.obj'),
}

# Fingertips become deformable bodies → exclude from rigid FK render
_RIGID_RENDER_LINKS = [k for k in _LINK_MESHES
                      if not k.startswith('link_finger_tip_')]

_LINK_RAW_CACHE: dict = {}

def build_arm_mesh(q: dict) -> trimesh.Trimesh:
    """FK rigid mesh for everything EXCEPT fingertips (those are simulated)."""
    frames = compute_fk(q)
    all_v, all_f, offset = [], [], 0
    for lname in _RIGID_RENDER_LINKS:
        mpath = _LINK_MESHES.get(lname)
        if mpath is None:
            continue
        if lname not in _LINK_RAW_CACHE:
            tri = trimesh.load(str(mpath), force='mesh', process=False)
            _LINK_RAW_CACHE[lname] = (tri.vertices.copy(), tri.faces.copy()) \
                if (tri is not None and len(tri.vertices) > 0) else None
        entry = _LINK_RAW_CACHE[lname]
        if entry is None:
            continue
        local_v, local_f = entry
        pos, R_link = frames[lname]
        R_vis = rpy_to_R(*_VISUAL_RPY.get(lname, [0, 0, 0]))
        verts = (R_link @ R_vis @ local_v.T).T + pos
        all_v.append(verts)
        all_f.append(local_f + offset)
        offset += len(verts)
    return trimesh.Trimesh(vertices=np.vstack(all_v),
                           faces=np.vstack(all_f), process=False)

# ---- Boundary face extraction --------------------------------------------
def _extract_boundary_faces(elements, vertices):
    """Outward-oriented surface faces from a tet mesh (same logic as gatorman)."""
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_dict = {}
    for tet in np.asarray(elements, dtype=int):
        for face in faces:
            key = tuple(sorted((int(tet[face[0]]), int(tet[face[1]]), int(tet[face[2]]))))
            if key in face_dict:
                del face_dict[key]
            else:
                face_dict[key] = (int(tet[face[0]]), int(tet[face[1]]), int(tet[face[2]]))
    if not face_dict:
        return np.zeros((0, 3), dtype=int)
    vertices = np.asarray(vertices)
    centroid = vertices.mean(axis=0)
    oriented = []
    for vidx in face_dict.values():
        a, b, c = vertices[vidx[0]], vertices[vidx[1]], vertices[vidx[2]]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, (a + b + c) / 3.0 - centroid) < 0:
            oriented.append((vidx[0], vidx[2], vidx[1]))
        else:
            oriented.append(vidx)
    return np.array(oriented, dtype=int)


# ---- Fingertip preprocessing ----------------------------------------------
def prep_fingertip_tet(obj_path: Path, root_quantile: float = 0.10):
    """Load fingertip mesh → convex hull → tet mesh.

    Convex hull is used because the raw mesh is not watertight; the fingertip is
    nearly convex anyway (a small rounded block) so this preserves the contact
    surface well.

    Returns
    -------
    cv_local : (N, 3) — tet vertices in fingertip LOCAL frame.
    ce       : (M, 4) — tet element indices.
    root_idx : 1-D int array — local indices of the proximal "root" nodes
                (smallest z values; these stay glued to link_finger_*).
    surf_faces : (F, 3) — surface triangle indices, local numbering.
    """
    from py_diff_pd.common.tet_mesh import tetrahedralize

    raw = trimesh.load(str(obj_path), force='mesh', process=False)
    hull = raw.convex_hull
    tmp = '/tmp/_finger_tip_hull.obj'
    hull.export(tmp)
    try:
        cv, ce = tetrahedralize(tmp, normalize_input=False,
                                options={'minratio': 1.5})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    cv = np.asarray(cv, dtype=np.float64)
    ce = np.asarray(ce, dtype=int)

    z_thresh = np.quantile(cv[:, 2], root_quantile)
    root_idx = np.where(cv[:, 2] <= z_thresh)[0].astype(int)

    surf_faces = _extract_boundary_faces(ce, cv)
    return cv, ce, root_idx, surf_faces


def prep_rigid_tet(obj_path, visual_rpy=None):
    """Convex-hull tetrahedralization for a fully-rigid link body.

    All vertices will be Dirichlet-constrained every step to follow FK.
    Returns (cv_local, ce, surf_faces, R_vis).
    """
    from py_diff_pd.common.tet_mesh import tetrahedralize
    raw = trimesh.load(str(obj_path), force='mesh', process=False)
    hull = raw.convex_hull
    tmp = '/tmp/_rigid_link_hull.obj'
    hull.export(tmp)
    try:
        cv, ce = tetrahedralize(tmp, normalize_input=False,
                                options={'minratio': 1.5})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    cv = np.asarray(cv, dtype=np.float64)
    ce = np.asarray(ce, dtype=int)
    surf_faces = _extract_boundary_faces(ce, cv)
    R_vis = rpy_to_R(*visual_rpy) if visual_rpy is not None else np.eye(3)
    return cv, ce, surf_faces, R_vis


# ---- Helpers --------------------------------------------------------------
def gripper_midpoint(q: dict) -> np.ndarray:
    fk = compute_fk(q)
    return (fk['link_finger_tip_right'][0] + fk['link_finger_tip_left'][0]) / 2.0


# ---- Colors / rendering ---------------------------------------------------
COLOR_BALL_HARD = (0xcd/255, 0x8d/255, 0xdd/255)   # #cd8ddd dark orchid (A — outer hard regions)
COLOR_BALL_SOFT = (0x36/255, 0xc0/255, 0xe3/255)   # #36c0e3 cyan-blue   (B — inner soft band)
COLOR_TIP       = (0.30, 0.30, 0.32)   # dark grey
COLOR_ARM       = (0.25, 0.25, 0.27)   # dark grey


def _add_ball_meshes_aba(renderer, ball_pts, split_info, stem):
    """Export A-B-A ball halves and add to renderer."""
    objs = []
    items = (
        ('hardA_faces', COLOR_BALL_HARD),
        ('soft_faces',  COLOR_BALL_SOFT),
        ('hardB_faces', COLOR_BALL_HARD),
    )
    for key, color in items:
        faces = split_info.get(key)
        if faces is None or len(faces) == 0:
            continue
        m = trimesh.Trimesh(vertices=ball_pts, faces=faces, process=False)
        obj = stem.parent / f'_{stem.name}_{key}.obj'
        m.export(str(obj))
        renderer.add_tri_mesh(obj, transforms=[('s', 1.0)],
                              render_tet_edge=False, color=color)
        objs.append(obj)
    return objs


def _add_fingertip_mesh(renderer, tip_pts, tip_faces, stem, color=COLOR_TIP):
    if tip_pts is None or tip_faces is None or len(tip_faces) == 0:
        return []
    m = trimesh.Trimesh(vertices=tip_pts, faces=tip_faces, process=False)
    obj = stem.parent / f'_{stem.name}.obj'
    m.export(str(obj))
    renderer.add_tri_mesh(obj, transforms=[('s', 1.0)],
                          render_tet_edge=False, color=color)
    return [obj]


def render_frame(q_kf, out_png, camera_pos, camera_lookat,
                 ball_pts=None, ball_split=None,
                 ltip_pts=None, ltip_faces=None,
                 rtip_pts=None, rtip_faces=None):
    """Render arm (FK rigid, no fingertips) + deformable ball + fingertips."""
    arm = build_arm_mesh(q_kf)
    arm_obj = out_png.with_suffix('.obj')
    arm.export(str(arm_obj))

    renderer = PbrtRenderer({
        'file_name':     str(out_png),
        'light_map':     'uffizi-large.exr',
        'sample':        256,
        'max_depth':     2,
        'camera_pos':    camera_pos,
        'camera_lookat': camera_lookat,
        'resolution':    (800, 800),
        'bg_color':      (1.0, 1.0, 1.0),  # solid white background
    })
    renderer.add_tri_mesh(arm_obj, transforms=[('s', 1.0)],
                          render_tet_edge=False, color=COLOR_ARM)

    tmp_objs = []
    if ball_pts is not None and ball_split is not None:
        tmp_objs += _add_ball_meshes_aba(renderer, ball_pts, ball_split, out_png)
    tmp_objs += _add_fingertip_mesh(renderer, ltip_pts, ltip_faces,
                                    out_png.with_name(out_png.stem + '_ltip'))
    tmp_objs += _add_fingertip_mesh(renderer, rtip_pts, rtip_faces,
                                    out_png.with_name(out_png.stem + '_rtip'))
    renderer.render()

    for o in tmp_objs:
        if Path(o).exists():
            Path(o).unlink()
    if arm_obj.exists():
        arm_obj.unlink()


# ---- Keyframe generation --------------------------------------------------
def lerp(a, b, t): return a + (b - a) * t

def make_q(torso=0., shoulder=0., elbow=0., wrist=0., finger=0.):
    return {
        'joint_torso':        torso,
        'joint_shoulder':     shoulder,
        'joint_bicep':        0.,
        'joint_elbow':        elbow,
        'joint_forearm':      0.,
        'joint_wrist':        wrist,
        'joint_gripper':      0.,
        'joint_finger_right': finger,
        'joint_finger_left':  finger,
    }

N_PRESKIP = 6   # frames 0..N_PRESKIP-1: fast approach (not rendered)

def gen_keyframes(finger_close: float = 0.9, n_descent: int = 10, n_close: int = 10):
    kf = []
    _E_GRASP = 0.2
    _W_GRASP = np.pi - np.pi/2 - _E_GRASP
    FC = finger_close

    _T_ENTRY = 6.0 / 9.0
    sh_entry = lerp(0., np.pi / 2, _T_ENTRY)
    el_entry = lerp(0., _E_GRASP,  _T_ENTRY)
    wr_entry = lerp(np.pi, _W_GRASP, _T_ENTRY)

    for i in range(N_PRESKIP):
        t = i / max(N_PRESKIP - 1, 1) * _T_ENTRY
        kf.append(make_q(torso=0.,
            shoulder=lerp(0., np.pi / 2, t),
            elbow   =lerp(0., _E_GRASP,  t),
            wrist   =lerp(np.pi, _W_GRASP, t),
            finger  =0.))

    for i in range(n_descent):
        t = i / max(n_descent - 1, 1)
        kf.append(make_q(torso=0.,
            shoulder=lerp(sh_entry, np.pi / 2, t),
            elbow   =lerp(el_entry, _E_GRASP,  t),
            wrist   =lerp(wr_entry, _W_GRASP,  t),
            finger  =0.))

    for i in range(n_close):
        t = i / max(n_close - 1, 1)
        kf.append(make_q(torso=0., shoulder=np.pi/2, elbow=_E_GRASP,
                         wrist=_W_GRASP, finger=lerp(0., FC, t)))

    for i in range(10):
        t = i / 9.0
        kf.append(make_q(torso=0., shoulder=np.pi/2,
                         elbow=lerp(_E_GRASP, 0., t),
                         wrist=lerp(_W_GRASP, np.pi/2, t), finger=FC))

    for i in range(10):
        t = i / 9.0
        kf.append(make_q(torso=0., shoulder=np.pi/2, elbow=0., wrist=np.pi/2,
                         finger=lerp(FC, 0., t)))

    return kf


# ---- DiffPD soft-body simulation ------------------------------------------
def run_sim(keyframes: list,
            n_gravity: int = 1, thread_ct: int = 8,
            contact_radius: float = 1.5e-3,
            contact_kn_ball: float = 1e4,
            contact_kn_tip:  float = 1e4,
            contact_kf: float = 0.0,
            contact_mu: float = 0.5,
            ball_youngs_hard: float = 1e8,
            ball_youngs_soft: float = 5e5,
            ball_soft_low_q:  float = 0.333,
            ball_soft_high_q: float = 0.667,
            fingertip_youngs: float = 1e13,
            method: str = 'pd_eigen_alg4',
            max_iter: int = 5000,
            rel_tol: float = 1e-4,
            physics_start: int = 0,
            gravity_start: int = None,
            grasp_z_offset: float = 0.02,
            dt: float = 1e-4,
            ball_radius: float = 0.05,
            ball_subdiv: int = 5,
            velocity_damping: float = 0.9,
            ):
    """One TetDeformable holds ball + L tip + R tip.

    Forces / energies (all in C++ implicit solve):
      * pd_neohookean per element — hetero ABA on ball, uniform stiff on tips
      * gravity (state force)
      * planar_contact at z=0 (state force, floor)
      * mesh_contact (state force) — ball surface ↔ both fingertip surfaces

    Drive:
      * Per-step Dirichlet on the proximal cap of each fingertip, target xyz
        recomputed from URDF FK every frame.

    Returns dict with all data needed for rendering each frame.
    """
    from py_diff_pd.common.common import ndarray, create_folder, print_info, \
        print_ok, copy_std_int_vector
    from py_diff_pd.common.tet_mesh import generate_tet_mesh, tetrahedralize
    from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable, \
        StdRealVector, StdIntVector

    sim_folder = Path('google_robot_gravity_sim_mesh_contact')
    create_folder(str(sim_folder), exist_ok=True)

    DT            = dt
    N_FK          = len(keyframes)
    total_steps   = N_FK + n_gravity
    PHYSICS_START = physics_start
    LIFT_START    = PHYSICS_START + 10

    nu  = 0.45
    rho = 800.0

    # ---- 1) Fingertip tets (left + right, in LOCAL frame) ----------------
    ltip_obj = _LINK_MESHES['link_finger_tip_left']
    rtip_obj = _LINK_MESHES['link_finger_tip_right']
    if ltip_obj is None or rtip_obj is None:
        raise RuntimeError('fingertip OBJ not found.')
    cv_l, ce_l, root_l, surf_l = prep_fingertip_tet(ltip_obj)
    cv_r, ce_r, root_r, surf_r = prep_fingertip_tet(rtip_obj)
    print_info(f'L tip tet: {len(cv_l)} nodes, {len(ce_l)} tets, '
               f'{len(root_l)} root nodes, {len(surf_l)} surface faces')
    print_info(f'R tip tet: {len(cv_r)} nodes, {len(ce_r)} tets, '
               f'{len(root_r)} root nodes, {len(surf_r)} surface faces')

    # ---- 1b) Near-gripper rigid links: finger bases + gripper housing --------
    fb_obj = _LINK_MESHES['link_finger_right']   # same mesh for L and R bases
    gr_obj = _LINK_MESHES['link_gripper']
    if fb_obj is None or gr_obj is None:
        raise RuntimeError('finger base / gripper OBJ not found.')
    cv_fb, ce_fb, surf_fb, _ = prep_rigid_tet(fb_obj)
    cv_fbl, ce_fbl, surf_fbl = cv_fb.copy(), ce_fb.copy(), surf_fb.copy()
    cv_fbr, ce_fbr, surf_fbr = cv_fb.copy(), ce_fb.copy(), surf_fb.copy()
    cv_gr,  ce_gr,  surf_gr, R_vis_gr = prep_rigid_tet(
        gr_obj, _VISUAL_RPY.get('link_gripper'))
    Nfbl, Nfbr, Ngr = len(cv_fbl), len(cv_fbr), len(cv_gr)
    print_info(f'Finger base tet: {len(cv_fb)} nodes, {len(ce_fb)} tets, '
               f'{len(surf_fb)} surface faces (shared L/R)')
    print_info(f'Gripper housing tet: {Ngr} nodes, {len(ce_gr)} tets, '
               f'{len(surf_gr)} surface faces')

    # ---- 2) Initial FK + grasp_pos (distal-tip midpoint) -----------------
    fk_init = compute_fk(keyframes[PHYSICS_START - 1])
    pos_l0, R_l0 = fk_init['link_finger_tip_left']
    pos_r0, R_r0 = fk_init['link_finger_tip_right']
    cv_l_world0 = (R_l0 @ cv_l.T).T + pos_l0
    cv_r_world0 = (R_r0 @ cv_r.T).T + pos_r0

    pos_fbl0, R_fbl0 = fk_init['link_finger_left']
    pos_fbr0, R_fbr0 = fk_init['link_finger_right']
    pos_gr0,  R_gr0  = fk_init['link_gripper']
    cv_fbl_world0 = (R_fbl0 @ cv_fbl.T).T + pos_fbl0
    cv_fbr_world0 = (R_fbr0 @ cv_fbr.T).T + pos_fbr0
    cv_gr_world0  = (R_gr0  @ R_vis_gr @ cv_gr.T).T + pos_gr0

    # Ball center is aligned with the midpoint of the fingertip MESH distal
    # ends (largest local z, opposite the root cap), not the link origins —
    # otherwise the gripper has to reach further down past the tips.
    def _distal_pt(cv_local, top_q=0.9):
        thr = np.quantile(cv_local[:, 2], top_q)
        return cv_local[cv_local[:, 2] >= thr].mean(axis=0)
    distal_l_w = R_l0 @ _distal_pt(cv_l) + pos_l0
    distal_r_w = R_r0 @ _distal_pt(cv_r) + pos_r0
    grasp_pos = (distal_l_w + distal_r_w) / 2.0 \
                - np.array([0., 0., grasp_z_offset])
    print_info(f'Grasp position (distal-tip midpoint): {np.round(grasp_pos, 3)}')

    # ---- 3) Ball tet mesh -------------------------------------------------
    ball_mesh = trimesh.creation.icosphere(subdivisions=ball_subdiv,
                                            radius=ball_radius)
    ball_mesh.apply_translation(grasp_pos)
    tmp_stl = str(sim_folder / '_ball.stl')
    ball_mesh.export(tmp_stl)
    try:
        cv_ball, ce_ball = tetrahedralize(tmp_stl, normalize_input=False,
                                          options={'minratio': 1.2})
    finally:
        if os.path.exists(tmp_stl):
            os.remove(tmp_stl)
    cv_ball = np.asarray(cv_ball, dtype=np.float64)
    ce_ball = np.asarray(ce_ball, dtype=int)
    ball_surf_local = _extract_boundary_faces(ce_ball, cv_ball)
    print_info(f'Ball tet: {len(cv_ball)} nodes, {len(ce_ball)} tets, '
               f'{len(ball_surf_local)} surface faces')

    # ---- 4) Combine into single mesh -------------------------------------
    Nb, Nl, Nr = len(cv_ball), len(cv_l), len(cv_r)
    OFF_L   = Nb
    OFF_R   = Nb + Nl
    OFF_FBL = Nb + Nl + Nr
    OFF_FBR = Nb + Nl + Nr + Nfbl
    OFF_GR  = Nb + Nl + Nr + Nfbl + Nfbr

    all_v = np.vstack([cv_ball, cv_l_world0, cv_r_world0,
                       cv_fbl_world0, cv_fbr_world0, cv_gr_world0])
    all_e = np.vstack([ce_ball,
                       ce_l   + OFF_L,
                       ce_r   + OFF_R,
                       ce_fbl + OFF_FBL,
                       ce_fbr + OFF_FBR,
                       ce_gr  + OFF_GR])

    # surface faces in GLOBAL numbering
    ball_surf_g = ball_surf_local.copy()
    l_surf_g    = surf_l   + OFF_L
    r_surf_g    = surf_r   + OFF_R
    fbl_surf_g  = surf_fbl + OFF_FBL
    fbr_surf_g  = surf_fbr + OFF_FBR
    gr_surf_g   = surf_gr  + OFF_GR
    tip_surf_g  = np.vstack([l_surf_g, r_surf_g, fbl_surf_g, fbr_surf_g, gr_surf_g])

    # ---- 5) Material assignment ------------------------------------------
    # Closing axis: +x in world after FK at grasp pose (right tip - left tip)
    closing_axis = pos_r0 - pos_l0
    closing_axis /= (np.linalg.norm(closing_axis) + 1e-12)

    ball_c = cv_ball.mean(axis=0)
    vert_proj = (cv_ball - ball_c) @ closing_axis
    q_lo = np.quantile(vert_proj, ball_soft_low_q)
    q_hi = np.quantile(vert_proj, ball_soft_high_q)
    print_info(f'Ball ABA split: low={q_lo:+.4f}, high={q_hi:+.4f} '
               f'(quantiles {ball_soft_low_q:.2f}/{ball_soft_high_q:.2f})')

    tet_proj = vert_proj[ce_ball].mean(axis=1)        # (Mb,)
    ball_is_soft = (tet_proj > q_lo) & (tet_proj < q_hi)
    ball_is_hardA = tet_proj <= q_lo
    ball_is_hardB = tet_proj >= q_hi
    print_info(f'Ball tets: hardA={ball_is_hardA.sum()}, '
               f'soft={ball_is_soft.sum()}, hardB={ball_is_hardB.sum()}')

    # Same split for surface faces (used by renderer)
    face_proj = vert_proj[ball_surf_local].mean(axis=1)
    face_hardA = face_proj <= q_lo
    face_hardB = face_proj >= q_hi
    face_soft  = ~(face_hardA | face_hardB)
    ball_split = {
        'hardA_faces': ball_surf_local[face_hardA],
        'soft_faces':  ball_surf_local[face_soft],
        'hardB_faces': ball_surf_local[face_hardB],
    }

    def lame(E):
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return la, mu

    la_h, mu_h     = lame(ball_youngs_hard)
    la_s, mu_s     = lame(ball_youngs_soft)
    la_tip, mu_tip = lame(fingertip_youngs)

    nh_params = []
    for i in range(len(ce_ball)):
        if ball_is_soft[i]:
            nh_params.append(float(mu_s)); nh_params.append(float(la_s))
        else:
            nh_params.append(float(mu_h)); nh_params.append(float(la_h))
    for _ in range(len(ce_l)):
        nh_params.append(float(mu_tip)); nh_params.append(float(la_tip))
    for _ in range(len(ce_r)):
        nh_params.append(float(mu_tip)); nh_params.append(float(la_tip))
    for _ in range(len(ce_fbl) + len(ce_fbr) + len(ce_gr)):
        nh_params.append(float(mu_tip)); nh_params.append(float(la_tip))

    # ---- 6) Generate tet mesh + deformable -------------------------------
    tmp_bin = str(sim_folder / '_combined.bin')
    generate_tet_mesh(all_v, all_e, tmp_bin)
    mesh = TetMesh3d()
    mesh.Initialize(tmp_bin)
    deformable = TetDeformable()
    E_avg = (ball_youngs_hard + ball_youngs_soft + fingertip_youngs) / 3.0
    deformable.Initialize(tmp_bin, rho, 'none', E_avg, nu)
    os.remove(tmp_bin)

    deformable.AddPdEnergy('pd_neohookean', nh_params, [])

    # ---- 7) State forces -------------------------------------------------
    # Gravity is applied via f_ext (per-node m*g) so we can switch it on
    # mid-simulation; see gravity_start.
    deformable.AddStateForce('planar_contact',
                              [0., 0., 1., 0., 2, 1e4, 50., 0.5])

    mesh_params = [
        contact_radius,
        contact_kn_ball,    # kn_left  → ball verts pushing tip faces
        contact_kn_tip,     # kn_right → tip  verts pushing ball faces
        contact_kf,
        contact_mu,
        float(len(ball_surf_g)),
        float(len(tip_surf_g)),
    ]
    mesh_params.extend(ball_surf_g.ravel().astype(float).tolist())
    mesh_params.extend(tip_surf_g.ravel().astype(float).tolist())
    deformable.AddStateForce('mesh_contact', mesh_params)
    print_info(f'mesh_contact: ball={len(ball_surf_g)} faces, '
               f'tips={len(tip_surf_g)} faces, '
               f'r={contact_radius:.4f} kn_ball={contact_kn_ball:.0f} '
               f'kn_tip={contact_kn_tip:.0f}')

    # ---- 8) Dirichlet on fingertip root nodes + rigid arm links -------------
    root_l_global = (root_l + OFF_L).astype(int)
    root_r_global = (root_r + OFF_R).astype(int)

    def set_root_dirichlet(R_l, pos_l, R_r, pos_r,
                           R_fbl, pos_fbl, R_fbr, pos_fbr, R_gr, pos_gr):
        for li, gi in zip(root_l, root_l_global):
            target = R_l @ cv_l[li] + pos_l
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
        for li, gi in zip(root_r, root_r_global):
            target = R_r @ cv_r[li] + pos_r
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
        for li in range(Nfbl):
            target = R_fbl @ cv_fbl[li] + pos_fbl
            gi = li + OFF_FBL
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
        for li in range(Nfbr):
            target = R_fbr @ cv_fbr[li] + pos_fbr
            gi = li + OFF_FBR
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
        for li in range(Ngr):
            target = R_gr @ R_vis_gr @ cv_gr[li] + pos_gr
            gi = li + OFF_GR
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))

    # Register Dirichlet keys ONCE at init (so PD prefactor stays valid).
    set_root_dirichlet(R_l0, pos_l0, R_r0, pos_r0,
                       R_fbl0, pos_fbl0, R_fbr0, pos_fbr0, R_gr0, pos_gr0)

    # ---- 9) Simulation loop ---------------------------------------------
    dofs     = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_phys   = all_v.ravel().copy()
    v_phys   = np.zeros(dofs)

    # Per-node gravity force (m * g) for f_ext-based gravity.
    GRAVITY_Z = -9.81 * 125
    node_mass = np.zeros(len(all_v))
    for tet in all_e:
        v_t = all_v[tet]
        vol = abs(np.dot(v_t[1] - v_t[0],
                         np.cross(v_t[2] - v_t[0], v_t[3] - v_t[0]))) / 6.0
        node_mass[tet] += rho * vol / 4.0
    gravity_force = np.zeros((len(all_v), 3))
    gravity_force[:, 2] = GRAVITY_Z * node_mass
    gravity_force_flat = gravity_force.ravel()
    g_start = physics_start if gravity_start is None else int(gravity_start)
    print_info(f'Gravity starts at step {g_start} '
               f'(physics_start={physics_start})')

    base_opt = {'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': rel_tol,
                'verbose': 0, 'thread_ct': thread_ct}
    if method.startswith('newton'):
        opt = {**base_opt, 'max_newton_iter': max_iter}
    elif method == 'pd_eigen_alg4':
        opt = {**base_opt, 'max_pd_iter': max_iter,
               'use_bfgs': 0, 'use_acc': 1, 'use_sparse': 0,
               'project_newton': 1, 'use_abs': 1, 'aa_window': 5}
    else:  # pd_eigen
        opt = {**base_opt, 'max_pd_iter': max_iter,
               'use_bfgs': 1, 'bfgs_history_size': 10,
               'project_newton': 1, 'use_abs': 1}

    act = np.zeros(act_dofs)
    ci  = StdIntVector(0)

    # Pre-physics frames: rest pose (everything still).
    all_q = [all_v.copy() for _ in range(PHYSICS_START)]

    print_info(f'Running DiffPD: steps {PHYSICS_START}..{total_steps} '
               f'({total_steps - PHYSICS_START + 1} physics steps)...')
    t0 = time.time()
    for step in range(PHYSICS_START, total_steps + 1):
        kf_idx = min(step, N_FK - 1)
        q_kf   = keyframes[kf_idx]
        fk     = compute_fk(q_kf)
        pos_l,   R_l   = fk['link_finger_tip_left']
        pos_r,   R_r   = fk['link_finger_tip_right']
        pos_fbl, R_fbl = fk['link_finger_left']
        pos_fbr, R_fbr = fk['link_finger_right']
        pos_gr,  R_gr  = fk['link_gripper']
        set_root_dirichlet(R_l, pos_l, R_r, pos_r,
                           R_fbl, pos_fbl, R_fbr, pos_fbr, R_gr, pos_gr)

        f_ext = gravity_force_flat.copy() if step >= g_start \
                else np.zeros(dofs)

        q_next  = StdRealVector(dofs)
        v_next  = StdRealVector(dofs)
        ci_next = copy_std_int_vector(ci)
        deformable.PyForward(method, q_phys, v_phys, act, f_ext, DT, opt,
                             q_next, v_next, ci_next)
        q_phys = ndarray(q_next)
        damp = velocity_damping if step < N_FK else 1.0
        v_phys = ndarray(v_next) * damp
        ci     = ci_next
        all_q.append(q_phys.reshape(-1, 3).copy())

        if (step - PHYSICS_START) % 10 == 0:
            ball_z = q_phys.reshape(-1, 3)[:Nb, 2]
            print_info(f'  step {step:3d}  ball z=[{ball_z.min():+.3f}, '
                       f'{ball_z.max():+.3f}]  '
                       f'elapsed={time.time()-t0:.1f}s')

    print_ok(f'DiffPD done in {time.time()-t0:.1f}s')

    return {
        'all_q':         all_q,             # list of (N, 3), len = total_steps + 1
        'ball_rest':     all_v[:Nb].copy(),
        'ball_split':    ball_split,
        'ball_local_faces': ball_surf_local,
        'tip_l_faces':   surf_l,            # local numbering (within tip)
        'tip_r_faces':   surf_r,
        'Nb':            Nb,
        'OFF_L':         OFF_L,
        'OFF_R':         OFF_R,
        'Nl':            Nl,
        'Nr':            Nr,
        'cv_l':          cv_l,              # local-frame fingertip nodes
        'cv_r':          cv_r,
        'PHYSICS_START': PHYSICS_START,
        'N_FK':          N_FK,
    }


# ---- Per-frame world positions for renderer -------------------------------
def slice_frame(sim, q_kf, frame_q_or_none):
    """Return ball_pts, ltip_pts, rtip_pts in world coords for one frame.

    For pre-physics frames (q is None / pre-START), fingertips follow FK rigidly,
    ball stays at rest pos.
    """
    Nb    = sim['Nb']
    OFF_L = sim['OFF_L']
    OFF_R = sim['OFF_R']
    Nl    = sim['Nl']
    Nr    = sim['Nr']
    cv_l  = sim['cv_l']
    cv_r  = sim['cv_r']
    if frame_q_or_none is not None:
        Q = frame_q_or_none
        ball_pts = Q[:Nb]
        ltip_pts = Q[OFF_L:OFF_L + Nl]
        rtip_pts = Q[OFF_R:OFF_R + Nr]
    else:
        ball_pts = sim.get('ball_rest')
        fk = compute_fk(q_kf)
        pos_l, R_l = fk['link_finger_tip_left']
        pos_r, R_r = fk['link_finger_tip_right']
        ltip_pts = (R_l @ cv_l.T).T + pos_l
        rtip_pts = (R_r @ cv_r.T).T + pos_r
    return ball_pts, ltip_pts, rtip_pts


# ---- Main -----------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    from py_diff_pd.common.common import print_info

    p = argparse.ArgumentParser(
        description='Google Robot FK + DiffPD soft-body grasping (mesh contact, ABA ball).')

    # Ball
    p.add_argument('--ball-radius',       type=float, default=0.04)
    p.add_argument('--ball-subdiv',       type=int,   default=6)
    p.add_argument('--ball-youngs-hard',  type=float, default=1e8,
                   help="Young's modulus for the two outer hard regions of the ball.")
    p.add_argument('--ball-youngs-soft',  type=float, default=4.55e5,
                   help="Young's modulus for the inner soft band (between fingers).")
    p.add_argument('--ball-soft-low-q',   type=float, default=0.333,
                   help='Lower quantile of closing-axis projection for the soft band.')
    p.add_argument('--ball-soft-high-q',  type=float, default=0.667,
                   help='Upper quantile of closing-axis projection for the soft band.')
    p.add_argument('--grasp-z-offset',    type=float, default=0.01,
                   help='Ball center below the gripper midpoint (m). 0 = aligned.')

    # Fingertip
    p.add_argument('--fingertip-youngs',  type=float, default=1e13,
                   help="Young's modulus for both fingertip soft bodies.")

    # Contact (mesh_contact state force)
    p.add_argument('--contact-radius',    type=float, default=5.5e-4,
                   help='Contact distance threshold (m). Small → contact only on actual touch.')
    p.add_argument('--contact-kn-ball',   type=float, default=5e3,
                   help='Normal stiffness when ball verts push fingertip faces.')
    p.add_argument('--contact-kn-tip',    type=float, default=2e4,
                   help='Normal stiffness when fingertip verts push ball faces.')
    p.add_argument('--contact-kf',        type=float, default=0.0,
                   help='Friction stiffness cap.')
    p.add_argument('--contact-mu',        type=float, default=0.5,
                   help='Friction coefficient.')
    p.add_argument('--velocity-damping',  type=float, default=0.95,
                   help='Velocity scale applied after each step (1.0=none, 0.95=light, 0.9=medium).')

    # Animation
    p.add_argument('--n-descent',         type=int,   default=60)
    p.add_argument('--n-gravity',         type=int,   default=100)
    p.add_argument('--n-close',           type=int,   default=50)
    p.add_argument('--finger-close',      type=float, default=0.85)
    p.add_argument('--physics-start',     type=int,   default=76,
                   help='Step at which DiffPD physics starts. Default = N_PRESKIP + n_descent.')
    p.add_argument('--gravity-start',     type=int,   default=None,
                   help='Step at which gravity turns on (via f_ext). '
                        'Default = physics_start + n_close (gravity kicks in '
                        'only after the fingers have finished closing).')

    # Solver
    p.add_argument('--method',            type=str,   default='pd_eigen_alg4',
                   choices=['pd_eigen', 'pd_eigen_alg4', 'newton_pcg', 'newton_cholesky'])
    p.add_argument('--max-iter',          type=int,   default=5000)
    p.add_argument('--rel-tol',           type=float, default=1e-4)
    p.add_argument('--dt',                type=float, default=2.5e-4)
    p.add_argument('--threads',           type=int,   default=16)

    # Camera (wide-shot defaults so the whole arm is visible).
    # p.add_argument('--cam-distance', type=float, default=0.42,
    #                help='Camera distance from lookat along +x (m). Larger = farther/wider shot.')
    # p.add_argument('--cam-side',     type=float, default=0.0,
    #                help='Camera lateral offset along +y (m).')
    # p.add_argument('--cam-height',   type=float, default=0.10,
    #                help='Camera height offset above lookat (m).')
    # p.add_argument('--cam-lookat-z', type=float, default=0.4,
    #                help='Absolute z (m) the camera looks at. ~0.55 frames the whole arm; '
    #                     'pass grasp_pos.z (~0.4) for a close-up on the gripper.')
    # room-out
    p.add_argument('--cam-distance', type=float, default=1.6,
                   help='Camera distance from lookat along +x (m). Larger = farther/wider shot.')
    p.add_argument('--cam-side',     type=float, default=0.3,
                   help='Camera lateral offset along +y (m).')
    p.add_argument('--cam-height',   type=float, default=0.20,
                   help='Camera height offset above lookat (m).')
    p.add_argument('--cam-lookat-z', type=float, default=0.55,
                   help='Absolute z (m) the camera looks at. ~0.55 frames the whole arm; '
                        'pass grasp_pos.z (~0.4) for a close-up on the gripper.')

    p.add_argument('--no-vis', action='store_true')
    args = p.parse_args()

    keyframes = gen_keyframes(finger_close=args.finger_close,
                              n_descent=args.n_descent,
                              n_close=args.n_close)
    N_FK = len(keyframes)
    assert N_FK == N_PRESKIP + args.n_descent + args.n_close + 20, N_FK
    physics_start = (args.physics_start
                     if args.physics_start is not None
                     else N_PRESKIP + args.n_descent)
    gravity_start = (args.gravity_start
                     if args.gravity_start is not None
                     else physics_start + args.n_close)

    grasp_pos  = gripper_midpoint(keyframes[N_PRESKIP + args.n_descent - 1])
    lookat_xy  = grasp_pos[:2]
    CAM_LOOKAT = (float(lookat_xy[0]), float(lookat_xy[1]), float(args.cam_lookat_z))
    CAM_POS    = (CAM_LOOKAT[0] + args.cam_distance,
                  CAM_LOOKAT[1] + args.cam_side,
                  CAM_LOOKAT[2] + args.cam_height)
    print(f'Camera → pos={np.round(CAM_POS, 3)}  lookat={np.round(CAM_LOOKAT, 3)}')

    print('=== DiffPD soft-body grasping (ball + 2 fingertips) ===')
    sim = run_sim(
        keyframes,
        n_gravity=args.n_gravity, thread_ct=args.threads,
        contact_radius=args.contact_radius,
        contact_kn_ball=args.contact_kn_ball,
        contact_kn_tip =args.contact_kn_tip,
        contact_kf=args.contact_kf,
        contact_mu=args.contact_mu,
        ball_youngs_hard=args.ball_youngs_hard,
        ball_youngs_soft=args.ball_youngs_soft,
        ball_soft_low_q =args.ball_soft_low_q,
        ball_soft_high_q=args.ball_soft_high_q,
        fingertip_youngs=args.fingertip_youngs,
        method=args.method, max_iter=args.max_iter, rel_tol=args.rel_tol,
        physics_start=physics_start,
        gravity_start=gravity_start,
        grasp_z_offset=args.grasp_z_offset,
        dt=args.dt,
        ball_radius=args.ball_radius, ball_subdiv=args.ball_subdiv,
        velocity_damping=args.velocity_damping,
    )

    if args.no_vis:
        print('--no-vis: skipping rendering.')
        sys.exit(0)

    PHYSICS_START = sim['PHYSICS_START']
    all_q = sim['all_q']
    ball_split = sim['ball_split']
    tip_l_faces = sim['tip_l_faces']
    tip_r_faces = sim['tip_r_faces']

    # ---- Render FK animation frames (skip first N_PRESKIP) ---------------
    print(f'\n=== Part 1: {N_FK - N_PRESKIP} animation frames ===')
    out_idx = 0
    all_pngs = []
    for i in range(N_PRESKIP, N_FK):
        q_kf = keyframes[i]
        # Pre-physics frames have no real sim state — pass Q=None so fingertips
        # render at the FK pose instead of frozen at the PHYSICS_START-1 pose.
        Q = all_q[i] if (PHYSICS_START <= i < len(all_q)) else None
        ball_pts, ltip_pts, rtip_pts = slice_frame(sim, q_kf, Q)

        png = OUT_DIR / f'frame_{out_idx:04d}.png'
        print(f'[kf {i:02d}/{N_FK-1}] out {out_idx:03d}  '
              f'shoulder={q_kf["joint_shoulder"]:.2f}  '
              f'elbow={q_kf["joint_elbow"]:.2f}  '
              f'wrist={q_kf["joint_wrist"]:.2f}  '
              f'finger={q_kf["joint_finger_right"]:.2f}')
        render_frame(q_kf, png, CAM_POS, CAM_LOOKAT,
                     ball_pts=ball_pts, ball_split=ball_split,
                     ltip_pts=ltip_pts, ltip_faces=tip_l_faces,
                     rtip_pts=rtip_pts, rtip_faces=tip_r_faces)
        all_pngs.append(png)
        out_idx += 1

    # ---- Gravity / fall phase --------------------------------------------
    print(f'\n=== Part 2: gravity phase ({args.n_gravity} frames) ===')
    last_kf = keyframes[-1]
    for j in range(args.n_gravity):
        idx = N_FK + j
        if idx >= len(all_q):
            break
        Q = all_q[idx]
        Nb = sim['Nb']
        if Q[:Nb, 2].max() < 0.0:
            print_info(f'Ball passed below floor at gravity frame {j}, stopping.')
            break

        ball_pts, ltip_pts, rtip_pts = slice_frame(sim, last_kf, Q)
        png = OUT_DIR / f'frame_{out_idx:04d}.png'
        out_idx += 1
        render_frame(last_kf, png, CAM_POS, CAM_LOOKAT,
                     ball_pts=ball_pts, ball_split=ball_split,
                     ltip_pts=ltip_pts, ltip_faces=tip_l_faces,
                     rtip_pts=rtip_pts, rtip_faces=tip_r_faces)
        all_pngs.append(png)
        print(f'  gravity [{j:02d}]  ball z=[{Q[:Nb,2].min():+.3f}, '
              f'{Q[:Nb,2].max():+.3f}]')

    print(f'Total frames: {len(all_pngs)}')
    video_path = OUT_DIR / 'full_animation.mp4'
    subprocess.run([
        shutil.which('ffmpeg') or 'ffmpeg', '-y',
        '-framerate', '10',
        '-i', str(OUT_DIR / 'frame_%04d.png'),
        '-vf', 'scale=800:800',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(video_path),
    ], check=True)
    print(f'\nVideo → {video_path}')