"""
UR5 + slender cylinder — DiffPD poke demo against a soft slab.

Mirrors the arm-control / state-force structure of
`google_robot_static_mesh_contact.py`, but with a different robot and
without a gripper.

Architecture
------------
One TetDeformable holds TWO bodies in a single global vertex numbering:
    body 0 : soft slab  (the poked target — soft pd_neohookean, bottom face
                         Dirichlet-clamped to a fixed table position)
    body 1 : slender cylinder (stiff pd_neohookean; ALL nodes are Dirichlet
                               and follow the FK of ee_link every frame, so
                               it behaves like a rigid tool)

State forces (all in the implicit solve):
  * mesh_contact (faces_a = slab surface, faces_b = cylinder surface)

There is no gravity / floor — slab is anchored at the bottom, cylinder is
fully Dirichlet, so neither falls.

UR5 mesh source:
  asset/mesh/ur5/visual/  (corlab/cogimon-gazebo-models, GPL-3.0)

Motion:
  * Move arm to a "ready" pose with the flange (ee_link +x in this URDF)
    pointing down and the cylinder tip parked just above the slab top.
  * Oscillate `shoulder_lift_joint` sinusoidally by ±`poke_amp` radians
    for `n_cycles` cycles → cylinder pokes the slab repeatedly.

Run from python/example/:
    python ur5_poke_demo.py --n-cycles 4 --poke-amp 0.15
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
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UR5_VIS  = _REPO_ROOT / 'asset' / 'mesh' / 'ur5' / 'visual'

OUT_DIR = Path('ur5_poke_demo_out')
OUT_DIR.mkdir(exist_ok=True)

# ---- Atlas mesh (replaces the box slab) ----------------------------------
ATLAS_BASE  = _REPO_ROOT / 'asset' / 'mesh' / 'atlas_dice'
ATLAS_FRAME = 18                               # start sim from frame 18
ATLAS_OBJ   = ATLAS_BASE / f'mesh-f{ATLAS_FRAME:05d}.obj'
ATLAS_TEX   = ATLAS_BASE / f'Atlas-F{ATLAS_FRAME:05d}.png'
# Rotation that puts the "4 dots" face up (deg). Applied to mesh-local pts
# around the mesh centroid before placing in the world.
ATLAS_ROT_RPY = (90.0, 0.0, 0.0)

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

# ---- UR5 joint chain (from ur5robot.urdf) ---------------------------------
# (child, parent, xyz, rpy, axis)  — axis None ⇒ fixed joint
_JOINT_CHAIN = [
    ('base_link',     'world',          [0., 0., 0.01],     [0., 0., 0.],       None),
    ('shoulder_link', 'base_link',      [0., 0., 0.089159], [0., 0., 0.],       [0,0,1]),
    ('upper_arm_link','shoulder_link',  [0., 0.13585, 0.],  [0., np.pi/2, 0.],  [0,1,0]),
    ('forearm_link',  'upper_arm_link', [0.,-0.1197, 0.425],[0., 0., 0.],       [0,1,0]),
    ('wrist_1_link',  'forearm_link',   [0., 0., 0.39225],  [0., np.pi/2, 0.],  [0,1,0]),
    ('wrist_2_link',  'wrist_1_link',   [0., 0.093, 0.],    [0., 0., 0.],       [0,0,1]),
    ('wrist_3_link',  'wrist_2_link',   [0., 0., 0.09465],  [0., 0., 0.],       [0,1,0]),
    ('ee_link',       'wrist_3_link',   [0., 0.0823, 0.],   [0., 0., np.pi/2],  None),
]

_JOINT_NAME = {
    'shoulder_link':  'shoulder_pan_joint',
    'upper_arm_link': 'shoulder_lift_joint',
    'forearm_link':   'elbow_joint',
    'wrist_1_link':   'wrist_1_joint',
    'wrist_2_link':   'wrist_2_joint',
    'wrist_3_link':   'wrist_3_joint',
}

# URDF visual offsets (per <visual><origin>) — applied to local mesh verts.
_VISUAL_XYZ = {
    'base_link':     [0., 0., 0.],
    'shoulder_link': [0., 0., 0.],
    'upper_arm_link':[0., 0., 0.],
    'forearm_link':  [0., 0., 0.],
    'wrist_1_link':  [0., 0.093, 0.],
    'wrist_2_link':  [0., 0., 0.09465],
    'wrist_3_link':  [0., 0., 0.],
}
_VISUAL_RPY = {
    'base_link':     [0., 0., 2.3561944875],
    'shoulder_link': [0., 0., 0.],
    'upper_arm_link':[0., 0., 0.],
    'forearm_link':  [0., 0., 0.],
    'wrist_1_link':  [0., np.pi/2, 0.],
    'wrist_2_link':  [0., 0., 0.],
    'wrist_3_link':  [0., 0., 0.],
}

_LINK_MESHES = {
    'base_link':      UR5_VIS / 'Base.dae',
    'shoulder_link':  UR5_VIS / 'Shoulder.dae',
    'upper_arm_link': UR5_VIS / 'UpperArm.dae',
    'forearm_link':   UR5_VIS / 'Forearm.dae',
    'wrist_1_link':   UR5_VIS / 'Wrist1.dae',
    'wrist_2_link':   UR5_VIS / 'Wrist2.dae',
    'wrist_3_link':   UR5_VIS / 'Wrist3.dae',
}

def compute_fk(q: dict) -> dict:
    frames = {'world': (np.zeros(3), I3.copy())}
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

# ---- DAE loading + arm render --------------------------------------------
_LINK_RAW_CACHE: dict = {}

def _load_link_mesh(path: Path, mesh_scale: float = 1.0):
    """Load a DAE / OBJ link mesh into a single Trimesh.

    Manually concatenate Scene geometries (verts + faces, with the
    per-node 4x4 graph transform applied so arm tubes land at the right
    offsets) to avoid trimesh's texture-packing path which fails on UR5
    DAE files.
    """
    loaded = trimesh.load(str(path), process=False)
    if loaded is None:
        return None
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            return None
        v_list, f_list, off = [], [], 0
        for node_name in loaded.graph.nodes_geometry:
            try:
                T, geom_name = loaded.graph[node_name]
            except Exception:
                continue
            g = loaded.geometry.get(geom_name)
            if g is None or not hasattr(g, 'vertices') or len(g.vertices) == 0:
                continue
            v_local = np.asarray(g.vertices)
            # Apply 4x4 scene-graph transform to vertices.
            v = (T[:3, :3] @ v_local.T).T + T[:3, 3]
            f = np.asarray(g.faces)
            v_list.append(v)
            f_list.append(f + off)
            off += len(v)
        if not v_list:
            return None
        verts = np.vstack(v_list)
        faces = np.vstack(f_list)
    else:
        verts = np.asarray(loaded.vertices)
        faces = np.asarray(loaded.faces)
    if mesh_scale != 1.0:
        verts = verts * mesh_scale
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def build_arm_mesh(q: dict, mesh_scale: float = 1.0) -> trimesh.Trimesh:
    """FK rigid render mesh for all UR5 links."""
    frames = compute_fk(q)
    all_v, all_f, offset = [], [], 0
    for lname, mpath in _LINK_MESHES.items():
        if mpath is None or not mpath.exists():
            continue
        if lname not in _LINK_RAW_CACHE:
            tri = _load_link_mesh(mpath, mesh_scale=mesh_scale)
            _LINK_RAW_CACHE[lname] = (
                (tri.vertices.copy(), tri.faces.copy())
                if (tri is not None and len(tri.vertices) > 0) else None)
        entry = _LINK_RAW_CACHE[lname]
        if entry is None:
            continue
        local_v, local_f = entry
        pos, R_link = frames[lname]
        R_vis = rpy_to_R(*_VISUAL_RPY[lname])
        t_vis = np.array(_VISUAL_XYZ[lname])
        verts = (R_link @ (R_vis @ local_v.T + t_vis.reshape(3,1))).T + pos
        all_v.append(verts)
        all_f.append(local_f + offset)
        offset += len(verts)
    return trimesh.Trimesh(vertices=np.vstack(all_v),
                           faces=np.vstack(all_f), process=False)

# ---- Boundary face extraction --------------------------------------------
def _extract_boundary_faces(elements, vertices):
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

# ---- Cylinder & slab tet prep --------------------------------------------
def prep_cylinder_tet(radius=0.01, length=0.20, sections=24):
    """Slender cylinder, axis along +x in LOCAL ee_link frame.

    Returns (cv_local, ce, surf_faces) — cap at x=0 (flange) → tip at x=L.
    """
    from py_diff_pd.common.tet_mesh import tetrahedralize
    cyl = trimesh.creation.cylinder(radius=radius, height=length,
                                    sections=sections)
    # cyl default: axis along z, centered at origin.
    # Rotate so its axis aligns with +x in ee_link frame.
    Ry = trimesh.transformations.rotation_matrix(np.pi/2, [0., 1., 0.])
    cyl.apply_transform(Ry)
    # Now along +x, centered at origin. Shift so cap is at x=0, tip at x=L.
    cyl.apply_translation([length / 2., 0., 0.])

    tmp = '/tmp/_ur5_cyl.stl'
    cyl.export(tmp)
    try:
        cv, ce = tetrahedralize(tmp, normalize_input=False,
                                options={'minratio': 1.2})
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    cv = np.asarray(cv, dtype=np.float64)
    ce = np.asarray(ce, dtype=int)
    surf = _extract_boundary_faces(ce, cv)
    return cv, ce, surf

def prep_slab_tet(size=(0.30, 0.30, 0.06), center=(0.5, 0.0, 0.05),
                  bottom_quantile=0.10):
    """Atlas mesh substituted for the original box slab.

    The high-res Atlas OBJ (~15k verts, non-watertight) is decimated with
    pyvista and tetrahedralized via tetgen (bypassing diff_pd's watertight
    assert). The mesh is scaled so its longest XY extent matches `size[0]`
    and translated so its centroid lands at `center`. Bottom-z nodes are
    Dirichlet-clamped (so the head sits anchored on the table).

    Returns (cv_world, ce, surf_faces, bottom_idx).
    """
    import pyvista as pv
    import tetgen

    # The Atlas 4D mesh has multiple overlapping shell components per frame
    # (e.g. frame 18 has 4 disconnected pieces — top half + bottom half +
    # extras). tetgen.make_manifold() keeps only the largest, so half the
    # dice would go missing. The dice is essentially convex, so we wrap
    # ALL vertices in a convex hull (guaranteed watertight, covers full
    # bbox) and then decimate that for the sim mesh.
    raw = trimesh.load(str(ATLAS_OBJ), process=False)
    hull = trimesh.convex.convex_hull(np.asarray(raw.vertices))
    tmp_stl = '/tmp/_ur5_atlas_hull.stl'
    hull.export(tmp_stl)
    m = pv.read(tmp_stl)
    m = m.decimate(0.85).clean()
    if os.path.exists(tmp_stl):
        os.remove(tmp_stl)

    pts = np.asarray(m.points, dtype=np.float64)
    # Apply orientation calibration in mesh-local space (around the centroid)
    # so the chosen face ends up on top in world frame.
    centroid_local = pts.mean(axis=0)
    R_atlas = rpy_to_R(*np.deg2rad(ATLAS_ROT_RPY))
    pts = (R_atlas @ (pts - centroid_local).T).T + centroid_local

    ext = pts.max(axis=0) - pts.min(axis=0)
    # Uniform scale so the mesh's largest XY extent matches size[0].
    target_max = max(size[0], size[1])
    scale = target_max / max(ext)
    pts = pts * scale
    # Align the mesh TOP to the original slab-top plane (center_z + size_z/2)
    # so the cylinder tip lands right on the head's crown. XY centroid → center.
    top_z = center[2] + size[2] * 0.5
    pts_max = pts.max(axis=0)
    pts_mean_xy = pts.mean(axis=0)[:2]
    pts[:, 0] += center[0] - pts_mean_xy[0]
    pts[:, 1] += center[1] - pts_mean_xy[1]
    pts[:, 2] += top_z - pts_max[2]
    m.points = pts

    tet = tetgen.TetGen(m)
    tet.make_manifold()
    cv, ce = tet.tetrahedralize(minratio=1.3, order=1)
    cv = np.asarray(cv, dtype=np.float64)
    ce = np.asarray(ce, dtype=int)
    if ce.shape[1] > 4:
        ce = ce[:, :4]
    surf = _extract_boundary_faces(ce, cv)
    z_thresh = np.quantile(cv[:, 2], bottom_quantile)
    bottom_idx = np.where(cv[:, 2] <= z_thresh)[0].astype(int)
    return cv, ce, surf, bottom_idx

# ---- Colors / rendering ---------------------------------------------------
COLOR_SLAB     = (0x6e/255, 0xa7/255, 0xd8/255)
COLOR_CYLINDER = (0xcd/255, 0x8d/255, 0xdd/255)
COLOR_ARM      = (0.27, 0.27, 0.29)


def _glass_mat(color):
    """PBRT glass material that tints transmitted light by `color`.
    eta close to 1.0 → minimal refraction so the geometry still reads."""
    return {'name': 'glass',
            'Kr': (0.05, 0.05, 0.05),
            'Kt': tuple(float(c) for c in color),
            'eta': 1.05}


def render_frame(q_kf, out_png, camera_pos, camera_lookat,
                 slab_pts=None, slab_faces=None,
                 cyl_pts=None, cyl_faces=None,
                 mesh_scale=1.0,
                 transparent_arm: bool = False,
                 transparent_cyl: bool = False,
                 hide_arm: bool = False,
                 hide_cyl: bool = False):
    arm_obj = out_png.with_suffix('.obj')
    if not hide_arm:
        arm = build_arm_mesh(q_kf, mesh_scale=mesh_scale)
        arm.export(str(arm_obj))

    # Glass needs deeper ray bounces (refract + reflect) to render correctly.
    max_depth = 8 if (transparent_arm or transparent_cyl) else 2
    renderer = PbrtRenderer({
        'file_name':     str(out_png),
        'light_map':     'uffizi-large.exr',
        'sample':        32,
        'max_depth':     max_depth,
        'camera_pos':    camera_pos,
        'camera_lookat': camera_lookat,
        'resolution':    (800, 800),
        'bg_color':      (1.0, 1.0, 1.0),
    })
    if not hide_arm:
        arm_kw = {'material': _glass_mat(COLOR_ARM)} if transparent_arm \
                 else {'color': COLOR_ARM}
        renderer.add_tri_mesh(arm_obj, transforms=[('s', 1.0)],
                              render_tet_edge=False, **arm_kw)

    tmp_objs = []
    if slab_pts is not None and slab_faces is not None and len(slab_faces) > 0:
        m = trimesh.Trimesh(vertices=slab_pts, faces=slab_faces, process=False)
        p = out_png.parent / f'_{out_png.stem}_slab.obj'
        m.export(str(p))
        renderer.add_tri_mesh(p, transforms=[('s', 1.0)],
                              render_tet_edge=False, color=COLOR_SLAB)
        tmp_objs.append(p)
    if (not hide_cyl
            and cyl_pts is not None and cyl_faces is not None
            and len(cyl_faces) > 0):
        m = trimesh.Trimesh(vertices=cyl_pts, faces=cyl_faces, process=False)
        p = out_png.parent / f'_{out_png.stem}_cyl.obj'
        m.export(str(p))
        cyl_kw = {'material': _glass_mat(COLOR_CYLINDER)} if transparent_cyl \
                 else {'color': COLOR_CYLINDER}
        renderer.add_tri_mesh(p, transforms=[('s', 1.0)],
                              render_tet_edge=False, **cyl_kw)
        tmp_objs.append(p)
    renderer.render()
    for o in tmp_objs:
        if Path(o).exists():
            Path(o).unlink()
    if arm_obj.exists():
        arm_obj.unlink()

# ---- Keyframes ------------------------------------------------------------
def make_q(pan=0., lift=0., elbow=0., w1=0., w2=0., w3=0.):
    return {
        'shoulder_pan_joint':  pan,
        'shoulder_lift_joint': lift,
        'elbow_joint':         elbow,
        'wrist_1_joint':       w1,
        'wrist_2_joint':       w2,
        'wrist_3_joint':       w3,
    }

# Ready pose: EE in front of robot, tool axis (ee +x) pointing roughly -z.
# These angles are a starting guess for the UR5 URDF in this repo —
# tune via CLI flags if the cylinder doesn't end up pointing down.
READY_POSE = make_q(
    pan=0.0, lift=-0.45, elbow=1.10,
    w1=-2.228, w2=-np.pi/2, w3=0.0,
)
# Resulting FK at rest pose (arm extended forward, slight retract, flange low):
#   pos_ee = [0.7902, 0.1091, -0.0350]   (cap at wrist, z ≈ -3.5 cm)
#   cyl_tip= [0.7909, 0.1091, -0.1350]   (with cyl_length=0.10)
# tool axis pointing straight DOWN (-z world).

def gen_keyframes(n_warmup: int, n_cycles: int, n_per_cycle: int,
                  poke_amp: float, ready: dict = None):
    """Warmup (hold ready) + N cycles of sinusoidal shoulder_lift oscillation."""
    if ready is None:
        ready = READY_POSE
    kf = [dict(ready) for _ in range(n_warmup)]
    for c in range(n_cycles):
        for i in range(n_per_cycle):
            t = i / n_per_cycle
            # Positive offset → push down (assuming +lift moves EE further down)
            d = poke_amp * np.sin(2 * np.pi * t)
            q = dict(ready)
            q['shoulder_lift_joint'] = ready['shoulder_lift_joint'] + d
            kf.append(q)
    return kf

# ---- DiffPD simulation ----------------------------------------------------
def run_sim(keyframes,
            mesh_scale: float = 1.0,
            cyl_radius: float = 0.012,
            cyl_length: float = 0.18,
            cyl_sections: int = 24,
            slab_size=(0.30, 0.30, 0.06),
            slab_center=(0.5, 0.0, 0.05),
            slab_youngs: float = 5e5,
            cyl_youngs: float = 1e8,
            contact_radius: float = 1.5e-3,
            contact_kn_slab: float = 1e4,
            contact_kn_cyl:  float = 1e4,
            contact_kf: float = 0.0,
            contact_mu: float = 0.5,
            method: str = 'pd_eigen_alg4',
            max_iter: int = 2000,
            rel_tol: float = 1e-4,
            dt: float = 5e-4,
            thread_ct: int = 8,
            physics_start: int = 0,
            velocity_damping: float = 0.95):
    from py_diff_pd.common.common import ndarray, create_folder, print_info, \
        print_ok, copy_std_int_vector
    from py_diff_pd.common.tet_mesh import generate_tet_mesh
    from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable, \
        StdRealVector, StdIntVector

    sim_folder = Path('ur5_poke_demo_sim')
    create_folder(str(sim_folder), exist_ok=True)

    N_FK = len(keyframes)
    PHYSICS_START = physics_start
    nu  = 0.45
    rho = 800.0

    # ---- 1) Soft slab (world frame) --------------------------------------
    cv_slab, ce_slab, surf_slab, bottom_idx = prep_slab_tet(
        size=slab_size, center=slab_center)
    print_info(f'Slab tet: {len(cv_slab)} nodes, {len(ce_slab)} tets, '
               f'{len(surf_slab)} surface faces, {len(bottom_idx)} bottom nodes')

    # ---- 2) Cylinder (local ee frame) ------------------------------------
    cv_cyl_local, ce_cyl, surf_cyl = prep_cylinder_tet(
        radius=cyl_radius, length=cyl_length, sections=cyl_sections)
    print_info(f'Cylinder tet: {len(cv_cyl_local)} nodes, {len(ce_cyl)} tets, '
               f'{len(surf_cyl)} surface faces (ALL nodes Dirichlet)')

    # ---- 3) Initial FK → world cylinder verts ----------------------------
    fk0 = compute_fk(keyframes[max(PHYSICS_START - 1, 0)])
    pos_ee0, R_ee0 = fk0['ee_link']
    cv_cyl_world0 = (R_ee0 @ cv_cyl_local.T).T + pos_ee0
    tip0 = R_ee0 @ np.array([cyl_length, 0., 0.]) + pos_ee0
    print_info(f'Initial EE pos: {np.round(pos_ee0, 3)}, '
               f'cylinder tip: {np.round(tip0, 3)}')

    # ---- 4) Combine into single mesh -------------------------------------
    Ns, Nc = len(cv_slab), len(cv_cyl_local)
    OFF_C = Ns

    all_v = np.vstack([cv_slab, cv_cyl_world0])
    all_e = np.vstack([ce_slab, ce_cyl + OFF_C])

    slab_surf_g = surf_slab.copy()
    cyl_surf_g  = surf_cyl + OFF_C

    # ---- 5) Material assignment ------------------------------------------
    def lame(E):
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return la, mu
    la_s, mu_s = lame(slab_youngs)
    la_c, mu_c = lame(cyl_youngs)

    # Dice (slab body): homogeneous neo-Hookean — every tet gets the SAME
    # (mu_s, la_s) derived from slab_youngs. The cylinder body is much stiffer
    # and is fully Dirichlet-driven (rigid tool).
    nh_params = []
    for _ in range(len(ce_slab)):
        nh_params.append(float(mu_s)); nh_params.append(float(la_s))
    for _ in range(len(ce_cyl)):
        nh_params.append(float(mu_c)); nh_params.append(float(la_c))

    # ---- 6) Generate combined tet + deformable ---------------------------
    tmp_bin = str(sim_folder / '_combined.bin')
    generate_tet_mesh(all_v, all_e, tmp_bin)
    mesh = TetMesh3d()
    mesh.Initialize(tmp_bin)
    deformable = TetDeformable()
    E_avg = (slab_youngs + cyl_youngs) / 2.0
    deformable.Initialize(tmp_bin, rho, 'none', E_avg, nu)
    os.remove(tmp_bin)
    deformable.AddPdEnergy('pd_neohookean', nh_params, [])

    # ---- 7) mesh_boundary (Alg4 NCP hard constraint) slab ↔ cylinder -----
    # boundary_params = [radius, split_idx, n_faces_a, n_faces_b,
    #                    ...slab_faces..., ...cyl_faces...]
    slab_surf_g_int = slab_surf_g.astype(int)
    cyl_surf_g_int  = cyl_surf_g.astype(int)
    boundary_params = [
        contact_radius,
        float(Ns),                         # split index: verts < Ns are slab
        float(len(slab_surf_g_int)),
        float(len(cyl_surf_g_int)),
    ]
    boundary_params.extend(slab_surf_g_int.ravel().astype(float).tolist())
    boundary_params.extend(cyl_surf_g_int.ravel().astype(float).tolist())

    slab_surf_verts = np.unique(slab_surf_g_int.ravel())
    cyl_surf_verts  = np.unique(cyl_surf_g_int.ravel())
    boundary_indices = np.unique(
        np.concatenate([slab_surf_verts, cyl_surf_verts])).astype(int)
    deformable.SetFrictionalBoundary('mesh', boundary_params,
                                     boundary_indices.tolist())
    print_info(f'mesh_boundary (NCP): radius={contact_radius:.3e}, '
               f'slab_faces={len(slab_surf_g_int)}, '
               f'cyl_faces={len(cyl_surf_g_int)}, '
               f'candidates={len(boundary_indices)}')

    # ---- 8) Dirichlet: slab bottom (fixed) + ALL cylinder nodes (FK) ----
    def set_all_dirichlet(R_ee, pos_ee):
        # Slab bottom — clamped to its initial position.
        for li in bottom_idx:
            target = cv_slab[li]
            gi = int(li)
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))
        # Cylinder — every node follows ee_link.
        for li in range(Nc):
            target = R_ee @ cv_cyl_local[li] + pos_ee
            gi = li + OFF_C
            for k in range(3):
                deformable.SetDirichletBoundaryCondition(
                    int(3 * gi + k), float(target[k]))

    # Register Dirichlet keys ONCE at init (so PD prefactor stays valid).
    set_all_dirichlet(R_ee0, pos_ee0)

    # ---- 9) Simulation loop ---------------------------------------------
    dofs     = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_phys   = all_v.ravel().copy()
    v_phys   = np.zeros(dofs)

    base_opt = {'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': rel_tol,
                'verbose': 0, 'thread_ct': thread_ct}
    if method.startswith('newton'):
        opt = {**base_opt, 'max_newton_iter': max_iter}
    elif method == 'pd_eigen_alg4':
        opt = {**base_opt, 'max_pd_iter': max_iter,
               'use_bfgs': 0, 'use_acc': 1, 'use_sparse': 0,
               'project_newton': 1, 'use_abs': 1, 'aa_window': 5}
    else:
        opt = {**base_opt, 'max_pd_iter': max_iter,
               'use_bfgs': 1, 'bfgs_history_size': 10,
               'project_newton': 1, 'use_abs': 1}

    act = np.zeros(act_dofs)
    ci  = StdIntVector(0)

    all_q = [all_v.copy() for _ in range(PHYSICS_START)]

    print_info(f'Running DiffPD: {N_FK - PHYSICS_START} physics steps...')
    t0 = time.time()
    for step in range(PHYSICS_START, N_FK):
        q_kf = keyframes[step]
        fk = compute_fk(q_kf)
        pos_ee, R_ee = fk['ee_link']
        set_all_dirichlet(R_ee, pos_ee)

        f_ext = np.zeros(dofs)
        q_next  = StdRealVector(dofs)
        v_next  = StdRealVector(dofs)
        ci_next = copy_std_int_vector(ci)
        deformable.PyForward(method, q_phys, v_phys, act, f_ext, dt, opt,
                             q_next, v_next, ci_next)
        q_phys = ndarray(q_next)
        v_phys = ndarray(v_next) * velocity_damping
        ci     = ci_next
        all_q.append(q_phys.reshape(-1, 3).copy())

        if (step - PHYSICS_START) % 10 == 0:
            slab_top_z = q_phys.reshape(-1, 3)[:Ns, 2].max()
            cyl_tip_z  = q_phys.reshape(-1, 3)[OFF_C:, 2].min()
            print_info(f'  step {step:3d}  '
                       f'slab_top_z={slab_top_z:+.4f}  '
                       f'cyl_tip_z={cyl_tip_z:+.4f}  '
                       f'elapsed={time.time()-t0:.1f}s')

    print_ok(f'DiffPD done in {time.time()-t0:.1f}s')

    return {
        'all_q':       all_q,
        'Ns':          Ns,
        'OFF_C':       OFF_C,
        'Nc':          Nc,
        'slab_faces':  surf_slab,
        'cyl_faces':   surf_cyl,
        'cv_cyl':      cv_cyl_local,
        'PHYSICS_START': PHYSICS_START,
    }

# ---- Per-frame slicing for renderer --------------------------------------
def slice_frame(sim, q_kf, frame_q_or_none):
    Ns    = sim['Ns']
    OFF_C = sim['OFF_C']
    Nc    = sim['Nc']
    cv_cyl_local = sim['cv_cyl']
    if frame_q_or_none is not None:
        Q = frame_q_or_none
        slab_pts = Q[:Ns]
        cyl_pts  = Q[OFF_C:OFF_C + Nc]
    else:
        # No physics state yet — slab at rest, cylinder following FK rigidly.
        slab_pts = None  # caller can decide what to do
        fk = compute_fk(q_kf)
        pos_ee, R_ee = fk['ee_link']
        cyl_pts = (R_ee @ cv_cyl_local.T).T + pos_ee
    return slab_pts, cyl_pts

# ---- Main -----------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    from py_diff_pd.common.common import print_info

    p = argparse.ArgumentParser(
        description='UR5 + slender cylinder poking a soft slab (DiffPD).')

    # Geometry
    p.add_argument('--mesh-scale',  type=float, default=1.0,
                   help='Scale factor for UR5 DAE meshes (try 0.001 if they are in mm).')
    p.add_argument('--cyl-radius',  type=float, default=0.012)
    p.add_argument('--cyl-length',  type=float, default=0.18)
    p.add_argument('--cyl-sections',type=int,   default=24)
    p.add_argument('--slab-size',   type=float, nargs=3,
                   default=[0.30, 0.30, 0.06],
                   help='Slab extents (x y z) in metres.')
    p.add_argument('--slab-center', type=float, nargs=3, default=None,
                   help='Slab centre xyz. Default: derived from EE tip at ready pose.')

    # Materials
    p.add_argument('--slab-youngs',  type=float, default=5e5)
    p.add_argument('--cyl-youngs',   type=float, default=1e8)

    # Contact
    p.add_argument('--contact-radius',  type=float, default=1.5e-3)
    p.add_argument('--contact-kn-slab', type=float, default=1e4)
    p.add_argument('--contact-kn-cyl',  type=float, default=1e4)
    p.add_argument('--contact-kf',      type=float, default=0.0)
    p.add_argument('--contact-mu',      type=float, default=0.5)

    # Animation — one poke cycle is aligned to the Atlas mesh sequence:
    # sim frame 0 (= atlas frame 18) → sim frame 35 (= atlas frame 53).
    # That's a full "down → contact → up → leave" cycle in 35 keyframes.
    p.add_argument('--n-warmup',   type=int,   default=0,
                   help='hold ready-pose for this many frames before the first poke')
    p.add_argument('--n-cycles',   type=int,   default=1,
                   help='number of full poke cycles to run')
    p.add_argument('--n-per-cycle',type=int,   default=35,
                   help='frames per cycle; default matches atlas frames 18..53')
    p.add_argument('--poke-amp',   type=float, default=0.22,
                   help='shoulder_lift oscillation amplitude (rad). Larger = deeper poke. '
                        'Default 0.22 → cyl_tip reaches ~z=0.14 at peak, ~3cm below the '
                        'deepest dent observed in atlas frame 40 (z=0.18).')

    # Ready-pose tuning (radians)
    p.add_argument('--q-pan',   type=float, default=0.0)
    p.add_argument('--q-lift',  type=float, default=-1.5)
    p.add_argument('--q-elbow', type=float, default=1.5)
    p.add_argument('--q-w1',    type=float, default=-1.5)
    p.add_argument('--q-w2',    type=float, default=-np.pi/2)
    p.add_argument('--q-w3',    type=float, default=0.0)

    # Solver
    p.add_argument('--method',  type=str, default='pd_eigen_alg4',
                   choices=['pd_eigen', 'pd_eigen_alg4', 'newton_pcg', 'newton_cholesky'])
    p.add_argument('--max-iter',type=int,   default=2000)
    p.add_argument('--rel-tol', type=float, default=1e-4)
    p.add_argument('--dt',      type=float, default=5e-4)
    p.add_argument('--threads', type=int,   default=8)
    p.add_argument('--velocity-damping', type=float, default=0.95)

    # Camera
    p.add_argument('--cam-distance', type=float, default=1.4)
    p.add_argument('--cam-side',     type=float, default=0.4)
    p.add_argument('--cam-height',   type=float, default=0.25)
    p.add_argument('--cam-lookat-z', type=float, default=0.25)

    p.add_argument('--no-vis', action='store_true')
    args = p.parse_args()

    ready = make_q(pan=args.q_pan, lift=args.q_lift, elbow=args.q_elbow,
                   w1=args.q_w1,  w2=args.q_w2,  w3=args.q_w3)
    keyframes = gen_keyframes(n_warmup=args.n_warmup,
                              n_cycles=args.n_cycles,
                              n_per_cycle=args.n_per_cycle,
                              poke_amp=args.poke_amp,
                              ready=ready)
    print_info(f'Total keyframes: {len(keyframes)}')

    # Default slab center: directly under the cylinder tip at ready pose,
    # with its top surface just below the tip.
    fk_ready = compute_fk(ready)
    pos_ee_r, R_ee_r = fk_ready['ee_link']
    tip_r = R_ee_r @ np.array([args.cyl_length, 0., 0.]) + pos_ee_r
    if args.slab_center is None:
        slab_center = (float(tip_r[0]), float(tip_r[1]),
                       float(tip_r[2]) - args.slab_size[2] / 2. - 0.005)
    else:
        slab_center = tuple(args.slab_center)
    print_info(f'EE@ready: {np.round(pos_ee_r, 3)}  '
               f'tip@ready: {np.round(tip_r, 3)}  '
               f'slab_center: {np.round(slab_center, 3)}')

    CAM_LOOKAT = (float(tip_r[0]), float(tip_r[1]), float(args.cam_lookat_z))
    CAM_POS    = (CAM_LOOKAT[0] + args.cam_distance,
                  CAM_LOOKAT[1] + args.cam_side,
                  CAM_LOOKAT[2] + args.cam_height)
    print(f'Camera → pos={np.round(CAM_POS, 3)}  lookat={np.round(CAM_LOOKAT, 3)}')

    print('=== UR5 poke demo (DiffPD) ===')
    sim = run_sim(
        keyframes,
        mesh_scale=args.mesh_scale,
        cyl_radius=args.cyl_radius, cyl_length=args.cyl_length,
        cyl_sections=args.cyl_sections,
        slab_size=tuple(args.slab_size), slab_center=slab_center,
        slab_youngs=args.slab_youngs, cyl_youngs=args.cyl_youngs,
        contact_radius=args.contact_radius,
        contact_kn_slab=args.contact_kn_slab,
        contact_kn_cyl =args.contact_kn_cyl,
        contact_kf=args.contact_kf, contact_mu=args.contact_mu,
        method=args.method, max_iter=args.max_iter, rel_tol=args.rel_tol,
        dt=args.dt, thread_ct=args.threads,
        physics_start=args.n_warmup,
        velocity_damping=args.velocity_damping,
    )

    if args.no_vis:
        print('--no-vis: skipping rendering.')
        sys.exit(0)

    all_q = sim['all_q']
    slab_faces = sim['slab_faces']
    cyl_faces  = sim['cyl_faces']

    print(f'\n=== Rendering {len(all_q)} frames ===')
    all_pngs = []
    for i in range(len(all_q)):
        q_kf = keyframes[i] if i < len(keyframes) else keyframes[-1]
        Q = all_q[i] if i >= sim['PHYSICS_START'] else None
        slab_pts, cyl_pts = slice_frame(sim, q_kf, Q)
        if slab_pts is None:
            # Pre-physics: draw slab at rest (we don't have it here — skip slab).
            slab_pts = np.zeros((0, 3))
            slab_faces_to_use = np.zeros((0, 3), dtype=int)
        else:
            slab_faces_to_use = slab_faces

        png = OUT_DIR / f'frame_{i:04d}.png'
        print(f'[{i:03d}/{len(all_q)-1}]  '
              f'lift={q_kf["shoulder_lift_joint"]:+.3f}  '
              f'elbow={q_kf["elbow_joint"]:+.3f}')
        render_frame(q_kf, png, CAM_POS, CAM_LOOKAT,
                     slab_pts=slab_pts, slab_faces=slab_faces_to_use,
                     cyl_pts=cyl_pts, cyl_faces=cyl_faces,
                     mesh_scale=args.mesh_scale)
        all_pngs.append(png)

    print(f'Total frames: {len(all_pngs)}')
    video_path = OUT_DIR / 'poke_demo.mp4'
    subprocess.run([
        shutil.which('ffmpeg') or 'ffmpeg', '-y',
        '-framerate', '20',
        '-i', str(OUT_DIR / 'frame_%04d.png'),
        '-vf', 'scale=800:800',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(video_path),
    ], check=True)
    print(f'\nVideo → {video_path}')
