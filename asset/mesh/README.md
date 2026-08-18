# Mesh Assets

## Third-Party Mesh Sources

### `ur5/` — Universal Robot UR5 Link Meshes

Visual DAE meshes for the UR5 robot arm (Base, Shoulder, UpperArm, Forearm, Wrist1–3).

- **Source**: [corlab/cogimon-gazebo-models](https://github.com/corlab/cogimon-gazebo-models),
  itself derived from the [Universal Robots ROS Description](https://github.com/ros-industrial/universal_robot)
  package maintained by ROS-Industrial.
- **License**: GPL-3.0 (as distributed by corlab/cogimon-gazebo-models)

> **Note on licensing**: The files under `asset/mesh/ur5/` are distributed under
> the GNU General Public License v3.0 (GPL-3.0), which is different from the MIT
> license that covers the rest of this repository.  If you redistribute or build
> upon these specific mesh files, the GPL-3.0 terms apply to them.  The remainder
> of the codebase is unaffected.  See the full GPL-3.0 text at
> <https://www.gnu.org/licenses/gpl-3.0.html>.

Used by `python/example/ur5_poke_demo.py`.

---

### `google_robot/` — Google RT-1 Robot Link Meshes

Coarse and recoloured OBJ link meshes for the Google RT-1 robot arm and gripper
(torso, shoulder, bicep, elbow, forearm, wrist, gripper, finger links).

- **Source**: [SimplerEnv](https://github.com/simpler-env/SimplerEnv) —
  *Evaluating Real-World Robot Manipulation Policies in Simulation*,
  Xuanlin Li\*, Jinghuan Shang\*, Siyuan Feng, Michael Stark, Shubham Tulsiani,
  Cewu Lu, Masayoshi Tomizuka, Lerrel Pinto, Xiaolong Wang.
  CoRL 2024.
- **License**: MIT (see SimplerEnv repository)

Used by `python/example/google_robot_static_mesh_contact.py`.

---

### `atlas_dice/` — PokeFlex Atlas 4D Scan Ground-Truth Meshes

Per-frame surface OBJ meshes of a deformable dice object extracted from the
Atlas 4D volumetric reconstruction (frames 18–53).  These serve as the
ground-truth shape target for the dice material-parameter optimisation pipeline.

- **Source**: [PokeFlex: A Real-World Dataset of Volumetric Deformable Objects
  for Robotics](https://pokeflex-dataset.github.io/) —
  Timothy Herr, Robin Stecklum, Thomas Bi, Michele Magno, Jenia Jitsev,
  Roland Siegwart, Jen Jen Chung.
  IEEE Robotics and Automation Letters (RA-L) / IROS 2024.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  (see the PokeFlex dataset page for details)

Used by `python/example/ur5_poke_demo.py` and
`python/example/dice_material_optimize.py`.

---

### `crab.*` and `gatorman*` — Crab and Gatorman Character Meshes

Volumetric meshes used for the heterogeneous-material forward-simulation scenes
(`crab.py`) and the contact-rich sword-ball scene (`gatorman_ball.py`).

- **Source**: [otmanon/simkit](https://github.com/otmanon/simkit) — specifically its
  `simkit-data` submodule ([otmanon/simkit-data](https://github.com/otmanon/simkit-data)),
  under `3d/crab/` and `3d/gatorman/`.
- **License**: The `simkit` code repository itself is MIT licensed, but no LICENSE,
  README, or per-mesh credit file could be found in `simkit-data` (at the repo root or
  inside the `crab`/`gatorman` folders) covering these mesh assets specifically.
  **The license/provenance of these two mesh files is therefore unconfirmed** — if this
  repository is redistributed or published beyond coursework/research use, verify
  directly with the `simkit`/`simkit-data` maintainer before relying on them.

Used by `python/example/crab.py` and `python/example/gatorman_ball.py`.

---

## MeshLab Tips

To improve the quality of triangle meshes:

- **Remove duplicates**: `Cleaning and Repairing` → `Remove Duplicated Faces` /
  `Remove Duplicated Vertex`
- **Merge close vertices**: `Cleaning and Repairing` → `Merge Close Vertices`
  (tune the distance threshold; optionally follow with `Subdivision Surfaces: Loop`)
- **Regularise triangles**: `Remeshing, Simplification and Reconstruction` →
  `Uniform Mesh Sampling`, `Iso Parametrization`, or
  `Quadratic Edge Collapse Decimation`