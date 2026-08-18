# DiffPhD: A Unified Differentiable Solver for Projective Heterogeneous Materials in Elastodynamics with Contact-Rich GPU-Acceleration

A GPU-accelerated, differentiable soft-body simulator built on
[DiffPD: Differentiable Projective Dynamics](https://github.com/mit-gfx/diff_pd_public)
(ACM TOG / SIGGRAPH 2022). It adds a sparse-inverse ("Algorithm 4" of
[Fast But Accurate: A Real-Time Hyperelastic Simulator with Robust Frictional Contact](https://dl.acm.org/doi/10.1145/3730834),
ACM Transactions on Graphics (Proc. ACM SIGGRAPH 2025)) global solve that runs on the
GPU through cuSPARSE, an NCP contact solver with Coulomb friction, and support for
spatially heterogeneous materials.

## Disclaimer

The code in this repository is derived from the publicly available publication and
source code of *DiffPD: Differentiable Projective Dynamics*. This project is used
solely for academic communication and educational purposes. No commercial use is
intended or involved.

---

## Overview

Differentiable PD is attractive because the global stiffness matrix `A` is constant
across iterations, so one factorisation can be amortised over a whole rollout. That
advantage collapses in exactly the regimes applications need: stiffness contrasts of
10x or more across a single mesh, Neo-Hookean energies whose local Hessians are
indefinite, and contact-rich rollouts that force the backward pass to re-apply `A^-1`
against changing contact Jacobians.

DiffPhD keeps `A` cheap to invert and robust under contrast through three contributions:

- **Heterogeneity via stiffness-aware projective assembly.** Per-element material
  parameters enter through the projective weights `w_e ∝ mu_e` in the global operator
  `A`, not in the per-element local proximal map. Material contrast therefore changes
  `A` while every element still runs the same Newton-on-stretches, so the PD fixed
  point stays contractive.

- **Differentiable hyperelasticity via proximal-map trust-region filtering.**
  Eigenvalue filtering is lifted onto PD's *proximal-map* Hessian in the **backward**
  pass only, under a state-adaptive rule that interpolates between the unprojected
  implicit-function-theorem operator, eigenvalue clamping and absolute-value filtering.
  The forward fixed point is left untouched, so Anderson Acceleration is not
  destabilised. Convergence uses bounded-history Type-II AA with a dual-gate criterion
  on the position and projection residuals.

- **Unified contact-rich GPU forward/backward loop.** A single persistent
  sparse-inverse factor pair `(S, S^T)` obtained by METIS nested dissection is shared
  across the forward global solve, the contact Delassus operator and the backward
  adjoint, so `A^-1 v = S^T (S v)` costs two sparse matrix-vector products and the
  dense inverse is never materialised. Contact uses a Signorini-Coulomb non-smooth NCP
  formulation with Coulomb friction.

Reported in the paper: up to **33x** end-to-end speedup over prior differentiable
solvers, convergence maintained at stiffness contrasts up to **100x** where prior PD
solvers degrade, and a **23.5x** backward-pass speedup on the densest contact
benchmark. Because the sparsity pattern is fixed by mesh topology alone, a
heterogeneous rollout costs exactly what its homogeneous counterpart does.

---

## Requirements

The configuration below is what the install has been verified against. Other versions
may work but are untested.

| | |
|---|---|
| OS | Linux (x86-64) |
| GPU | NVIDIA, CUDA 12 capable (verified on RTX 3080 Ti, `sm_86`) |
| Conda | Miniconda/Anaconda, any prefix |
| System compiler | `gcc`/`g++` in `/usr/bin` (verified with 7.3.0) |
| Python | 3.8.3 (from `environment.yml`) |
| CUDA | nvcc 12.4.131, cuSPARSE 12.3.1.170 (from the conda env) |

Two things are worth stating up front, because both are easy to get wrong:

- **All CUDA libraries come from the conda environment**, not from a system CUDA install.
- **The C++/CUDA host compiler must be the system `g++`, not conda's.** Conda's
  `gcc_linux-64 7.3.0` ships a cos6 sysroot (glibc 2.12) that cannot link against the
  environment's newer `libstdc++`.

---

## Installation

From the project root:

```bash
conda env create -f environment.yml
conda activate diff_phd
./install.sh
```

That is the whole install. `install.sh` fetches the submodules, builds the vendored
RealSim dependencies (oneTBB / GKlib / METIS), generates the SWIG bindings, builds the
CUDA extension, builds the pbrt renderer, and registers the package on the Python path.
It is idempotent — re-running it skips whatever is already done.

Options:

| flag | effect |
|---|---|
| `--no-renderer` | skip pbrt and ffmpeg; use this if you never pass a `vis_folder` |
| `--jobs N` | parallel build jobs (default: `nproc`) |
| `--arch NN` | CUDA architecture (default: autodetected from `nvidia-smi`, else `86`) |

After it finishes, `conda activate diff_phd` is the only setup step you ever need —
no `PYTHONPATH`, no `LD_LIBRARY_PATH`. The CUDA and TBB paths are baked into the
library's `RPATH`, and `install.sh` writes a `.pth` file so `import py_diff_pd` works
from any directory.

<details>
<summary>What <code>install.sh</code> does, and why each step is needed</summary>

Each of these is a step that silently breaks the build if skipped — they are the reason
the installer exists rather than a list of commands in this file.

1. **CUDA static libraries.** `environment.yml` does not ship the static libraries that
   `nvcc` links by default. Without them CMake cannot even identify the CUDA compiler:
   its try-compile links `-lcudadevrt -lcudart_static` and fails with
   `The CUDA compiler identification is unknown`. The installer runs
   `conda install -c nvidia --freeze-installed cuda-cudart-static=12.4.127`.

2. **`external/eigen`.** Required to compile; without it every translation unit fails
   with `fatal error: Eigen/Dense: No such file or directory`.

3. **RealSim dependencies built from source.** They are deliberately not shipped as
   binaries. Linking against a copy built on another machine makes the extension die
   with `Fatal Python error: Illegal instruction` (SIGILL) inside `PyForward`.
   `GKlib`'s `string.c` is compiled with the *system* `gcc`, because conda's sysroot
   `signal.h` is not C99-compatible with how GKlib uses it.

4. **SWIG before CMake.** `cpp/CMakeLists.txt` collects sources with
   `file(GLOB_RECURSE ...)`, which is evaluated at *configure* time. If `wrap.cxx` does
   not exist yet it is silently dropped, the build still succeeds, and you get a
   ~3.8 MB binding-less `.so` instead of ~5.4 MB — failing only later at import with
   `ModuleNotFoundError: No module named 'py_diff_pd.core.py_diff_pd_core'`. The
   installer verifies the built library actually exports the bindings.

5. **System `g++` as the C++/CUDA host compiler.** Conda's `gcc_linux-64 7.3.0` ships a
   cos6 sysroot (glibc 2.12) that cannot link the environment's newer `libstdc++`
   (`undefined reference to memcpy@GLIBC_2.14`). The env's CUDA libraries are still
   used, because `CMakeLists.txt` resolves them from `$CONDA_PREFIX`.

6. **pbrt is built completely outside the conda toolchain.** Setting the compiler is
   not enough, because activating the environment also exports
   `CFLAGS`/`CXXFLAGS`/`LDFLAGS`/`CPPFLAGS` and puts conda's binutils first on `PATH`,
   and CMake folds those into the link line regardless of which compiler you name.
   `LDFLAGS` carries `-L$CONDA_PREFIX/lib`, so `-lstdc++` resolves to conda's copy,
   which still lists `libdl.so.2` in `DT_NEEDED` — a library that no longer exists
   separately on glibc >= 2.34. conda's `ld` then satisfies it from its cos6 sysroot
   with a glibc 2.12 `libdl`, whose `_dl_vsym@GLIBC_PRIVATE` the host glibc does not
   provide. The installer therefore scrubs those four variables and prefers the system
   binutils for pbrt only. (This also drops conda's `-std=c++17`, under which the
   dynamic exception specifications in pbrt's bundled OpenEXR are a hard error.)

7. **`CMAKE_CUDA_ARCHITECTURES` on the command line.** `CMakeLists.txt` sets it *after*
   `project(... LANGUAGES CUDA)`, so it is empty during compiler detection and configure
   dies with `CUDA_ARCHITECTURES is empty for target "cmTC_xxxxx"`.

8. **A static `ffmpeg`.** `display.py` shells out to a bare `ffmpeg`, and the system one
   usually breaks under conda's libraries (e.g.
   `undefined symbol: FT_Get_Transform`). The installer symlinks `imageio-ffmpeg`'s
   static build into the environment's `bin`, where it takes priority on `PATH`.

</details>

## Verifying the installation

`python/example/napkin_3d_test.py` is a fast end-to-end check. It drops a
**heterogeneous** napkin (stiff half / soft half at 0.1x stiffness) onto a spherical
obstacle and runs the same scene through two solvers, so it exercises the GPU path,
the NCP contact solver, heterogeneous materials, the renderer and MP4 export:

```bash
conda activate diff_phd
cd python/example
python napkin_3d_test.py             # both solvers, with rendering (~5 min)
python napkin_3d_test.py --no-vis    # simulation only (~10 s)
```

It exits non-zero if either solver fails, and prints a summary like:

```
=== summary ===
  pd_eigen_alg_phd         OK         0.994s  lands and bends (max z_std after landing = 0.06671)
  pd_eigen_pcg_original    OK         0.424s  lands and bends (max z_std after landing = 0.06569)
  max |z_alg_phd - z_baseline| = 1.7370e-02
```

The GPU self-check should also appear early in the output:

```
[AlgPhd-GPU-VERIFY] dim=0 GPU S^T*S*v err=4.77057e-16  OK
[AlgPhd-GPU-VERIFY] dim=1 GPU S^T*S*v err=4.01634e-16  OK
[AlgPhd-GPU-VERIFY] dim=2 GPU S^T*S*v err=4.15905e-16  OK
```

With rendering enabled you get, under `napkin_3d_test/ratio_0.400000/`, 126 `.bin` +
126 `.png` frames, a `.data` pickle and an `.mp4` per solver. In the final frames the
stiff half of the napkin holds its shape while the soft half droops — that asymmetry is
the heterogeneous material working.

---

## Examples

```bash
conda activate diff_phd
cd python/example
python <example_name>.py
```

Simulations use 8 OpenMP threads by default; most scripts expose a `thread_ct`
variable. Keep it **strictly below** your core count. Scripts prefixed `print_` produce
the tables and figures for a scene once it has been run; scripts prefixed `render_`
produce its mesh sequence and video.

### Solvers

A solver is selected by name in each script's `methods` tuple.

| method | what it does |
|---|---|
| `pd_eigen_alg_phd` | **DiffPhD.** Stiffness-aware projective assembly, trust-region filtered backward pass, and one persistent sparse-inverse factor shared across the forward solve, the contact Delassus operator and the backward adjoint. |
| `pd_eigen` and other `pd_eigen_*` names | Baseline Projective Dynamics; any other `pd_eigen`-prefixed name falls through to this path. |
| `newton_pcg`, `newton_cholesky` | Newton baselines, used as accuracy references. |

### Heterogeneous forward simulation (Sec. 5.2)

Scenes carrying 10x-100x stiffness contrast, where prior PD solvers lose spectral
conditioning.

- **Cantilever** — `cantilever_3d.py`, `render_cantilever_3d.py`. A beam split into a stiff middle third between two soft
  ends; isolates the heterogeneous PD energy from contact and hyperelasticity.
- **Armadillo** — `armadillo.py`. A twisted Armadillo partitioned
  by height into three stiffness bands, driven through the per-element C++ material
  interface. Prior PD fails to converge beyond roughly 50x contrast; DiffPhD remains
  stable across the full 10x-100x sweep.
- **Crab** — `crab.py`. A shell-joint composite at 172,587 DoF with heterogeneous
  material. This is the scene where a dense `A^-1` no longer fits in GPU memory at all.

### Contact-rich forward simulation (Sec. 5.3)

- **Gatorman** — `gatorman_ball.py`. Complex mesh-to-mesh contact at
  60x contrast, where penalty-based and pure-PD contact models leave residual surface
  separation or interpenetration.
- **Napkin** — `napkin_3d_25x25.py`, `napkin_3d_50x50.py`, `napkin_3d_75x75.py`,
  `napkin_3d_100x100.py`, with `render_napkin_3d.py`. Codimensional cloth draping over a
  sphere. As the resolution rises the contact patch grows from a few percent of the mesh
  to roughly half of it, which is where the backward-pass speedup is largest.

### Differentiable inverse problems (Sec. 5.6)

**System identification.** Recover material parameters from a single trajectory.

- `bouncing_ball_3d.py` (single material) and `bouncing_ball_3d_heterogeneous.py`
  (three sub-regions at 30x contrast), with `render_bouncing_ball_3d.py`. The asymmetric
  mass distribution of the heterogeneous variant is what makes the per-sector deformation
  under impact identifiable.
- `plant_3d.py`, with `render_plant_3d.py`. An articulated potted
  plant whose branches deform with distinct curvature signatures under gravity-driven
  oscillation.

**Initial-state optimisation.**

- `bunny.py`, with `render_bunny_3d.py`. Optimises the
  bunny's initial position, orientation and velocity so that its centre of mass reaches
  a target after 100 frames of free fall and bouncing contact. The trajectory crosses
  several contact make/break transitions, at each of which the active set changes
  discontinuously and the Delassus operator is rebuilt — the regime that separates
  DiffPhD from solvers without contact-aware factor reuse. The stiffness-contrast sweep
  behind the 10x and 100x rows of the paper is driven by the `contrast_factor` option of
  `bunny_env`.
- `routing_tendon_3d.py`, with `render_routing_tendon_3d.py`. Muscle-energy backward pass (energy routing).

**Trajectory optimisation.**

- `torus_3d.py`, with `render_torus_3d.py`. Rolling locomotion.

The robot-manipulation and Real2Sim results (Sec. 5.7) depend on recorded hardware data
and are not reproducible from a single script.

For the upstream examples and their options, see the original DiffPD repository:
<https://github.com/mit-gfx/diff_pd_public>.

---

## Attribution

Built on [DiffPD](https://github.com/mit-gfx/diff_pd_public) (Du et al., ACM TOG 2022). `external/pbrt-v3` is
[pbrt-v3](https://github.com/mmp/pbrt-v3); `external/eigen` is
[Eigen](https://gitlab.com/libeigen/eigen). See `LICENSE` for this repository's terms.
