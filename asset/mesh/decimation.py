import os
import trimesh
import trimesh.repair
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# .mesh (MEDIT) parser
# ---------------------------------------------------------------------------

def load_medit(path):
    """Parse a MEDIT .mesh file and return (vertices, triangles, tetrahedra)."""
    vertices, triangles, tetrahedra = [], [], []
    section = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            low = line.lower()
            if low.startswith('vertices'):
                section = 'vertices'; continue
            if low.startswith('triangles'):
                section = 'triangles'; continue
            if low.startswith('tetrahedra'):
                section = 'tetrahedra'; continue
            if low.startswith('end') or low.startswith('edges') or low.startswith('normals'):
                section = None; continue
            # skip count lines (single integer)
            parts = line.split()
            if len(parts) == 1 and parts[0].lstrip('-').isdigit():
                continue

            if section == 'vertices' and len(parts) >= 3:
                vertices.append([float(x) for x in parts[:3]])
            elif section == 'triangles' and len(parts) >= 3:
                triangles.append([int(x) - 1 for x in parts[:3]])  # 1-indexed → 0-indexed
            elif section == 'tetrahedra' and len(parts) >= 4:
                tetrahedra.append([int(x) - 1 for x in parts[:4]])

    return (np.array(vertices, dtype=np.float64),
            np.array(triangles, dtype=np.int32) if triangles else np.empty((0, 3), dtype=np.int32),
            np.array(tetrahedra, dtype=np.int32) if tetrahedra else np.empty((0, 4), dtype=np.int32))


def surface_from_tets(tetrahedra):
    """Extract surface triangles: faces shared by exactly one tetrahedron."""
    tet_faces = np.array([
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ])
    all_faces = tetrahedra[:, tet_faces].reshape(-1, 3)  # (4*T, 3)
    all_faces = np.sort(all_faces, axis=1)               # canonical order

    # faces appearing once are on the surface
    _, idx, counts = np.unique(all_faces, axis=0, return_index=True, return_counts=True)
    surface = all_faces[idx[counts == 1]]
    return surface


# ---------------------------------------------------------------------------
# Load (auto-detect format)
# ---------------------------------------------------------------------------

def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mesh':
        verts, tris, tets = load_medit(path)
        if len(tets) > 0:
            surface_faces = surface_from_tets(tets)
        else:
            surface_faces = tris
        mesh = trimesh.Trimesh(vertices=verts, faces=surface_faces, process=False)
    else:
        mesh = trimesh.load(path, force='mesh')
    return mesh


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

def decimate(mesh, target_faces, scale):
    for method in ["simplify_quadric_decimation", "simplify_quadratic_decimation"]:
        fn = getattr(mesh, method, None)
        if fn is None:
            continue
        try:
            return fn(1.0 - scale)      # new API: target_reduction
        except (TypeError, ValueError):
            return fn(target_faces)     # old API: face count
    raise RuntimeError(
        "No decimation method found. trimesh version may be too old — "
        "try: pip install --upgrade trimesh"
    )


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _try(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        pass

def make_watertight(mesh, max_iter=5):
    _try(mesh.merge_vertices)
    mask = mesh.nondegenerate_faces()
    if not mask.all():
        mesh.update_faces(mask)

    for _ in range(max_iter):
        if mesh.is_watertight:
            break
        _try(trimesh.repair.fix_winding,   mesh)
        _try(trimesh.repair.fix_inversion, mesh)
        _try(trimesh.repair.fill_holes,    mesh)
        mask = mesh.nondegenerate_faces()
        if not mask.all():
            mesh.update_faces(mask)

    return mesh.is_watertight


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_mesh(mesh, path, title=""):
    verts = mesh.vertices.copy()
    faces = mesh.faces

    center = verts.mean(axis=0)
    scale  = (verts.max(axis=0) - verts.min(axis=0)).max()
    verts  = (verts - center) / scale

    tris = verts[faces]

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)
    light   = np.array([1.0, 1.0, 2.0])
    light   = light / np.linalg.norm(light)
    shade   = np.clip(normals @ light, 0.15, 1.0)
    colors  = np.stack([shade * 0.6, shade * 0.7, shade * 0.9, np.ones_like(shade)], axis=1)

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    poly = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
    ax.add_collection3d(poly)

    r = 0.55
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title(title, color="white", fontsize=10, pad=6)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    assert 0 < args.scale <= 1, "scale must be in (0, 1]"

    mesh = load_mesh(args.input)
    original_faces = len(mesh.faces)
    original_verts = len(mesh.vertices)
    print(f"Original watertight: {mesh.is_watertight}")

    target_faces = max(4, int(original_faces * args.scale))
    decimated = decimate(mesh, target_faces, args.scale)

    watertight = make_watertight(decimated)

    print(f"Vertices  : {original_verts:>8d} -> {len(decimated.vertices):>8d}")
    print(f"Faces     : {original_faces:>8d} -> {len(decimated.faces):>8d}  (scale={args.scale})")
    print(f"Watertight: {watertight}")
    if not watertight:
        print("WARNING: mesh is still not watertight after repair.")

    decimated.export(args.output)
    print(f"Saved to  : {args.output}")

    img_path = os.path.splitext(args.output)[0] + "_preview.png"
    render_mesh(
        decimated,
        img_path,
        title=f"Decimated  |  faces: {len(decimated.faces)}  |  watertight: {watertight}",
    )
    print(f"Preview   : {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Decimation")
    parser.add_argument("-i", "--input",  type=str,   required=True, help="Input mesh file (.obj or .mesh)")
    parser.add_argument("-o", "--output", type=str,   required=True, help="Output mesh file")
    parser.add_argument("-s", "--scale",  type=float, default=0.5,   help="Fraction of faces to keep (0 < scale <= 1)")

    args = parser.parse_args()
    main(args)
