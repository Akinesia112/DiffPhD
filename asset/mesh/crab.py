import argparse
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import tetgen
import trimesh

try:
    import pymeshfix
except Exception:
    pymeshfix = None


def ndarray(value):
    return np.asarray(value)


def filter_unused_vertices(vertices, elements):
    used = np.zeros(vertices.shape[0], dtype=bool)
    used[elements.reshape(-1)] = True
    remap = -np.ones(vertices.shape[0], dtype=int)
    remap[used] = np.arange(np.count_nonzero(used), dtype=int)
    return vertices[used], remap[elements]


def _load_trimesh(mesh_path):
    tri_mesh = trimesh.load(mesh_path, skip_materials=True)
    if isinstance(tri_mesh, trimesh.Scene):
        geometries = []
        for geom in tri_mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
        if not geometries:
            raise ValueError(f'No Trimesh geometry found in {mesh_path}')
        tri_mesh = trimesh.util.concatenate(geometries)
    if not isinstance(tri_mesh, trimesh.Trimesh):
        raise ValueError(f'Unsupported mesh type in {mesh_path}: {type(tri_mesh)}')
    return tri_mesh


def _try_repair_watertight(tri_mesh):
    repaired = tri_mesh.copy()
    repaired.remove_degenerate_faces()
    repaired.remove_duplicate_faces()
    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()
    trimesh.repair.fix_normals(repaired, multibody=True)

    if repaired.is_watertight:
        return repaired, 'trimesh_cleanup'

    try:
        trimesh.repair.fill_holes(repaired)
    except Exception as exc:
        # fill_holes requires optional dependencies (e.g., networkx).
        print(f'[warn] Skip trimesh.fill_holes: {exc}')
    repaired.remove_unreferenced_vertices()
    if repaired.is_watertight:
        return repaired, 'trimesh_fill_holes'

    if pymeshfix is not None:
        meshfix = pymeshfix.MeshFix(np.asarray(repaired.vertices), np.asarray(repaired.faces))
        meshfix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        repaired = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f, process=False)
        repaired.remove_unreferenced_vertices()
        repaired.merge_vertices()
        trimesh.repair.fix_normals(repaired, multibody=True)
        if repaired.is_watertight:
            return repaired, 'pymeshfix'

    return repaired, None


def _next_nonempty_line(lines, idx):
    n = len(lines)
    while idx < n:
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith('#'):
            continue
        return line, idx
    return None, idx


def load_medit_mesh(mesh_file_name):
    with open(mesh_file_name, 'r') as f:
        lines = f.readlines()

    idx = 0
    vertices = None
    tetrahedra = None

    while True:
        token, idx = _next_nonempty_line(lines, idx)
        if token is None:
            break

        if token == 'Vertices':
            count_line, idx = _next_nonempty_line(lines, idx)
            if count_line is None:
                raise ValueError(f'Invalid .mesh file, missing vertex count: {mesh_file_name}')
            n_vertices = int(count_line)
            verts = np.zeros((n_vertices, 3), dtype=float)
            for i in range(n_vertices):
                row, idx = _next_nonempty_line(lines, idx)
                if row is None:
                    raise ValueError(f'Invalid .mesh file, unexpected EOF in Vertices: {mesh_file_name}')
                items = row.split()
                if len(items) < 3:
                    raise ValueError(f'Invalid vertex row: {row}')
                verts[i, 0] = float(items[0])
                verts[i, 1] = float(items[1])
                verts[i, 2] = float(items[2])
            vertices = verts
        elif token == 'Tetrahedra':
            count_line, idx = _next_nonempty_line(lines, idx)
            if count_line is None:
                raise ValueError(f'Invalid .mesh file, missing tetrahedra count: {mesh_file_name}')
            n_tets = int(count_line)
            tets = np.zeros((n_tets, 4), dtype=int)
            for i in range(n_tets):
                row, idx = _next_nonempty_line(lines, idx)
                if row is None:
                    raise ValueError(f'Invalid .mesh file, unexpected EOF in Tetrahedra: {mesh_file_name}')
                items = row.split()
                if len(items) < 4:
                    raise ValueError(f'Invalid tetra row: {row}')
                tets[i, 0] = int(items[0]) - 1
                tets[i, 1] = int(items[1]) - 1
                tets[i, 2] = int(items[2]) - 1
                tets[i, 3] = int(items[3]) - 1
            tetrahedra = tets

    if vertices is None:
        raise ValueError(f'Vertices section not found in {mesh_file_name}')
    if tetrahedra is None:
        raise ValueError(f'Tetrahedra section not found in {mesh_file_name}')
    return vertices, tetrahedra


def tetrahedralize(triangle_mesh_file_name, visualize=False, normalize_input=False, options=None, allow_repair=True):
    tri_mesh = _load_trimesh(triangle_mesh_file_name)
    if not tri_mesh.is_watertight:
        if allow_repair:
            tri_mesh, repair_method = _try_repair_watertight(tri_mesh)
            if repair_method is None:
                raise ValueError(
                    'Input mesh is not watertight and automatic repair failed: '
                    f'{triangle_mesh_file_name}'
                )
            print(f'[info] Repaired non-watertight mesh using {repair_method}')
        else:
            raise ValueError(f'Input mesh is not watertight: {triangle_mesh_file_name}')

    if normalize_input:
        bbx_offset = np.min(tri_mesh.vertices, axis=0)
        tri_mesh.vertices -= bbx_offset
        bbx_extent = ndarray(tri_mesh.bounding_box.extents)
        tri_mesh.vertices /= np.max(bbx_extent)

    tmp_file_name = '.tmp.stl'
    tri_mesh.export(tmp_file_name)
    mesh = pv.read(tmp_file_name)
    os.remove(tmp_file_name)

    if visualize:
        mesh.plot()

    tet = tetgen.TetGen(mesh)
    tet.make_manifold()
    if options is None:
        nodes, elements = tet.tetrahedralize()
    else:
        nodes, elements = tet.tetrahedralize(**options)

    if visualize:
        tet_grid = tet.grid
        bbx_center = 0.5 * (np.min(tri_mesh.vertices, axis=0) + np.max(tri_mesh.vertices, axis=0))
        mask = tet_grid.points[:, 2] < bbx_center[2]
        half_tet = tet_grid.extract_points(mask)

        plotter = pv.Plotter()
        plotter.add_mesh(half_tet, color='w', show_edges=True)
        plotter.add_mesh(tet_grid, color='r', style='wireframe', opacity=0.2)
        plotter.show()
        plotter.close()

    nodes = ndarray(nodes)
    elements_unsigned = ndarray(elements).astype(int)[:, :4]

    elements = []
    for e in elements_unsigned:
        v = ndarray([nodes[ei] for ei in e])
        v0, v1, v2, v3 = v
        if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
            elements.append(e)
        else:
            elements.append([e[0], e[2], e[1], e[3]])
    elements = ndarray(elements).astype(int)
    return filter_unused_vertices(nodes, elements)


def write_tetgen_files(nodes, elements, node_file, ele_file):
    with open(node_file, 'w') as f:
        f.write(f'{nodes.shape[0]} 3 0 0\n')
        for i, v in enumerate(nodes, start=1):
            f.write(f'{i} {v[0]:.16g} {v[1]:.16g} {v[2]:.16g}\n')

    with open(ele_file, 'w') as f:
        f.write(f'{elements.shape[0]} 4 0\n')
        for i, e in enumerate(elements, start=1):
            f.write(f'{i} {e[0] + 1} {e[1] + 1} {e[2] + 1} {e[3] + 1}\n')


def main():
    parser = argparse.ArgumentParser(description='Convert a watertight surface mesh (.obj, .stl, etc.) to TetGen .node/.ele')
    parser.add_argument('input_mesh', nargs='?', default='crab.mesh__sf.obj', help='Input triangle mesh file')
    parser.add_argument('--out-prefix', default=None, help='Output prefix path (without extension)')
    parser.add_argument('--visualize', action='store_true', help='Visualize triangle/tet mesh while tetrahedralizing')
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization to [0, 1]^3')
    parser.add_argument('--max-volume', type=float, default=None, help='TetGen max tetrahedron volume constraint')
    parser.add_argument('--no-repair', action='store_true', help='Disable automatic watertight mesh repair')
    args = parser.parse_args()

    input_path = Path(args.input_mesh)
    if args.out_prefix is None:
        out_prefix = input_path.with_suffix('')
    else:
        out_prefix = Path(args.out_prefix)

    tet_options = {}
    if args.max_volume is not None:
        tet_options['maxvolume'] = args.max_volume

    if input_path.suffix.lower() == '.mesh':
        nodes, elements = load_medit_mesh(str(input_path))
    else:
        nodes, elements = tetrahedralize(
            str(input_path),
            visualize=args.visualize,
            normalize_input=not args.no_normalize,
            options=tet_options if tet_options else None,
            allow_repair=not args.no_repair,
        )

    node_file = f'{out_prefix}.node'
    ele_file = f'{out_prefix}.ele'
    write_tetgen_files(nodes, elements, node_file, ele_file)
    print(f'Wrote {node_file} ({nodes.shape[0]} vertices)')
    print(f'Wrote {ele_file} ({elements.shape[0]} tetrahedra)')


if __name__ == '__main__':
    main()