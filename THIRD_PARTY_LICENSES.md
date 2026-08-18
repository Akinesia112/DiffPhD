# Third-Party Asset Licenses

This repository is released under the MIT License (see `LICENSE`).
Certain bundled third-party assets are covered by different licenses, as
documented below.  The scope of each license is limited to the files listed
under that entry; the rest of the codebase remains MIT.

---

## 1. UR5 Visual Meshes — GPL-3.0

**Files**: `asset/mesh/ur5/visual/` (Base.dae, Shoulder.dae, UpperArm.dae,
Forearm.dae, Wrist1.dae, Wrist2.dae, Wrist3.dae)

**Source**: [corlab/cogimon-gazebo-models](https://github.com/corlab/cogimon-gazebo-models),
derived from the [Universal Robots ROS Description](https://github.com/ros-industrial/universal_robot)
package maintained by ROS-Industrial.

**License**: GNU General Public License v3.0 (GPL-3.0)
Full text: <https://www.gnu.org/licenses/gpl-3.0.html>

These mesh files are used solely for visualisation in
`python/example/ur5_poke_demo.py`.  If you redistribute or create derivative
works that include these files, the GPL-3.0 terms apply to those files.

---

## 2. Google RT-1 Robot Meshes — MIT

**Files**: `asset/mesh/google_robot/` (all files)

**Source**: [SimplerEnv](https://github.com/simpler-env/SimplerEnv) —
*Evaluating Real-World Robot Manipulation Policies in Simulation*,
Xuanlin Li\*, Jinghuan Shang\*, Siyuan Feng, Michael Stark, Shubham Tulsiani,
Cewu Lu, Masayoshi Tomizuka, Lerrel Pinto, Xiaolong Wang. CoRL 2024.

**License**: MIT
Full text: <https://github.com/simpler-env/SimplerEnv/blob/main/LICENSE>

---

## 3. PokeFlex Atlas Dice Meshes — CC BY 4.0

**Files**: `asset/mesh/atlas_dice/` (mesh-f00018.obj … mesh-f00053.obj)

**Source**: [PokeFlex: A Real-World Dataset of Volumetric Deformable Objects
for Robotics](https://pokeflex-dataset.github.io/) —
Timothy Herr, Robin Stecklum, Thomas Bi, Michele Magno, Jenia Jitsev,
Roland Siegwart, Jen Jen Chung.
IEEE Robotics and Automation Letters (RA-L) / IROS 2024.

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
Full text: <https://creativecommons.org/licenses/by/4.0/>

**Citation**:
```bibtex
@article{herr2024pokeflex,
  title   = {PokeFlex: A Real-World Dataset of Volumetric Deformable Objects for Robotics},
  author  = {Herr, Timothy and Stecklum, Robin and Bi, Thomas and Magno, Michele
             and Jitsev, Jenia and Siegwart, Roland and Chung, Jen Jen},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2024},
}
```
