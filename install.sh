#!/usr/bin/env bash
#
# DiffPD-GPU installer. Run with the conda environment already active:
#
#     conda env create -f environment.yml
#     conda activate diff_phd
#     ./install.sh
#
# Options:
#   --no-renderer   skip building pbrt (only needed to render frames / MP4s)
#   --jobs N        parallel build jobs (default: nproc)
#   --arch NN       CUDA architecture (default: autodetected, else 86)
#
# The script is idempotent: re-running it skips work that is already done.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BUILD_RENDERER=1
JOBS="$(nproc)"
CUDA_ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-renderer) BUILD_RENDERER=0; shift ;;
        --jobs)        JOBS="$2"; shift 2 ;;
        --arch)        CUDA_ARCH="$2"; shift 2 ;;
        -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
step "Checking prerequisites"

[[ -n "${CONDA_PREFIX:-}" ]] || die "No conda environment is active.
    Run:  conda env create -f environment.yml && conda activate diff_phd"

ENV_NAME="$(basename "$CONDA_PREFIX")"
[[ "$ENV_NAME" == "diff_phd" ]] || \
    info "warning: active env is '$ENV_NAME', expected 'diff_phd' — continuing anyway"

# The C++/CUDA host compiler must be the SYSTEM g++. Conda's gcc 7.3.0 ships a
# cos6 sysroot (glibc 2.12) that cannot link the environment's newer libstdc++
# (undefined reference to memcpy@GLIBC_2.14).
[[ -x /usr/bin/gcc && -x /usr/bin/g++ ]] || \
    die "system gcc/g++ not found in /usr/bin (required to build the CUDA host code)"

for tool in cmake make swig; do
    command -v "$tool" >/dev/null || die "'$tool' not found in the active environment"
done
command -v nvcc >/dev/null || die "nvcc not found — is this the right conda environment?"

info "env      : $CONDA_PREFIX"
info "python   : $(python --version 2>&1)"
info "nvcc     : $(nvcc --version | tail -1)"
info "system g++: $(/usr/bin/g++ --version | head -1)"

if [[ -z "$CUDA_ARCH" ]]; then
    if command -v nvidia-smi >/dev/null; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
    fi
    CUDA_ARCH="${CAP:-86}"
fi
info "cuda arch: sm_${CUDA_ARCH}"

# ── 1. CUDA static libraries ─────────────────────────────────────────────────
# environment.yml does not ship the static CUDA libs that nvcc links by default.
# Without them CMake cannot even identify the CUDA compiler: its try-compile
# links -lcudadevrt -lcudart_static and fails with
# "The CUDA compiler identification is unknown".
step "CUDA static libraries"
if [[ -f "$CONDA_PREFIX/lib/libcudart_static.a" ]]; then
    info "already present"
else
    info "installing cuda-cudart-static (this is required, see comment in install.sh)"
    conda install -p "$CONDA_PREFIX" -y -c nvidia --freeze-installed \
        cuda-cudart-static=12.4.127
fi

# ── 2. Submodules and RealSim ────────────────────────────────────────────────
step "Submodules"
if [[ ! -f external/eigen/Eigen/Dense ]]; then
    info "fetching external/eigen (required to compile)"
    git submodule update --init --depth 1 external/eigen
else
    info "external/eigen present"
fi

if [[ $BUILD_RENDERER -eq 1 && ! -d external/pbrt-v3/src ]]; then
    info "fetching external/pbrt-v3 (renderer)"
    git submodule update --init --recursive external/pbrt-v3
fi

# RealSim's sparse-LDLT sources and its dependencies (oneTBB / GKlib / METIS)
# are vendored under RealSim/, so nothing is fetched here — only their build
# output is produced in the next step.
for f in RealSim/src/Scomponent/tools/math/SparseLDLT.cpp \
         RealSim/src/Scomponent/tools/math/SparseMatrix.cpp \
         RealSim/include/Scomponent/tools/math/SparseLDLT.h; do
    [[ -f "$f" ]] || die "missing $f — the checkout is incomplete."
done
info "RealSim sources present"

# ── 3. RealSim dependencies ──────────────────────────────────────────────────
# These are deliberately NOT shipped as binaries: they are machine-specific, and
# linking against a copy built elsewhere makes the extension die with SIGILL
# inside PyForward.
step "RealSim dependencies (oneTBB, GKlib, METIS)"

if [[ -f RealSim/deps/onetbb_install/lib/libtbb.so ]]; then
    info "oneTBB already built"
else
    info "building oneTBB"
    ( cd RealSim/deps/oneTBB && mkdir -p build && cd build \
      && cmake -DCMAKE_INSTALL_PREFIX=../../onetbb_install -DTBB_TEST=OFF .. >/dev/null \
      && cmake --build . -j"$JOBS" >/dev/null && cmake --install . >/dev/null )
fi

if [[ -f RealSim/deps/METIS/GKlib_build/lib/libGKlib.a ]]; then
    info "GKlib already built"
else
    info "building GKlib"
    (
      cd RealSim/deps/METIS/GKlib
      GKLIB_INSTALL="$(pwd)/../GKlib_build"
      make config prefix="$GKLIB_INSTALL" openmp=set >/dev/null
      # string.c must be compiled with the system gcc: conda's sysroot signal.h
      # is not C99-compatible with GKlib's usage.
      /usr/bin/gcc -I. -Itest -fPIC -DLINUX -D_FILE_OFFSET_BITS=64 \
        -std=c99 -O3 -DNDEBUG -DNDEBUG2 -DHAVE_EXECINFO_H -DHAVE_GETLINE \
        -o build/Linux-x86_64/CMakeFiles/GKlib.dir/string.c.o -c string.c
      cd build/Linux-x86_64 && make install >/dev/null
    )
fi

if [[ -f RealSim/deps/METIS_install/lib/libmetis.a ]]; then
    info "METIS already built"
else
    info "building METIS"
    (
      cd RealSim/deps/METIS
      DEPS_ADDR="$(dirname "$(pwd)")"
      make config prefix="${DEPS_ADDR}/METIS_install" \
           gklib_path="${DEPS_ADDR}/METIS/GKlib_build" >/dev/null
      # libmetis.a builds; the cmpfillin tool may fail, which is harmless.
      make -j"$JOBS" >/dev/null 2>&1 || true
      mkdir -p "${DEPS_ADDR}/METIS_install/lib" "${DEPS_ADDR}/METIS_install/include"
      cp build/libmetis/libmetis.a "${DEPS_ADDR}/METIS_install/lib/"
      cp include/metis.h           "${DEPS_ADDR}/METIS_install/include/"
    )
    [[ -f RealSim/deps/METIS_install/lib/libmetis.a ]] || die "METIS build failed"
fi

# ── 4. SWIG bindings ─────────────────────────────────────────────────────────
# This must happen BEFORE cmake. cpp/CMakeLists.txt collects sources with
# file(GLOB_RECURSE ...), evaluated at configure time, so a missing wrap.cxx is
# silently dropped: the build still succeeds but produces a binding-less .so
# that only fails later at import.
step "Python bindings (SWIG)"
( cd cpp/core/src && swig -c++ -python py_diff_pd_core.i )
mv -f cpp/core/src/py_diff_pd_core.py python/py_diff_pd/core/
printf "root_path = '%s'\n" "$ROOT" > python/py_diff_pd/common/project_path.py
info "generated wrap.cxx + py_diff_pd_core.py"

# ── 5. Build the extension ───────────────────────────────────────────────────
step "Building py_diff_pd_core"
mkdir -p cpp/build
(
  cd cpp/build
  # All four compiler flags are required — see the comments above about the
  # system g++, and CMAKE_CUDA_ARCHITECTURES which CMakeLists.txt sets too late
  # for CUDA compiler detection.
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    | grep -E 'cuSPARSE|cuBLAS|cudart|TBB found|Configuring|Generating' || true
  make py_diff_pd_core -j"$JOBS" | tail -2
)

SO=python/py_diff_pd/core/_py_diff_pd_core.so
[[ -f "$SO" ]] || die "build did not produce $SO"

# Note: use grep -c, not grep -q, in these pipelines. Under `set -o pipefail`,
# grep -q exits on the first match and closes the pipe, the producer dies of
# SIGPIPE, and the whole pipeline reports failure.
if [[ "$(nm -D "$SO" | grep -c '_wrap_\|PyInit')" -eq 0 ]]; then
    die "$SO has no Python bindings — the SWIG step did not take effect"
fi
if [[ "$(ldd "$SO" | grep -c "$CONDA_PREFIX/lib/libcudart")" -eq 0 ]]; then
    info "warning: $SO does not link cudart from this environment"
fi
info "built $(du -h "$SO" | cut -f1) with bindings, CUDA linked"

# ── 6. Renderer + video export ───────────────────────────────────────────────
if [[ $BUILD_RENDERER -eq 1 ]]; then
    step "Renderer (pbrt) and MP4 export"
    if [[ -x external/pbrt_build/pbrt && -x external/pbrt_build/obj2pbrt ]]; then
        info "pbrt already built"
    else
        info "building pbrt (this takes a few minutes)"
        # pbrt is a self-contained renderer: it needs nothing from the conda
        # environment, and must be built completely outside its toolchain.
        # Activating the env sets CFLAGS/CXXFLAGS/LDFLAGS/CPPFLAGS (from the
        # `compilers` meta-package) and puts conda's binutils first on PATH.
        # Both leak into this build even when the compiler is set explicitly:
        #
        #   * LDFLAGS carries -L$CONDA_PREFIX/lib, so -lstdc++ resolves to
        #     conda's libstdc++. That one still has DT_NEEDED libdl.so.2, which
        #     no longer exists as a separate library on glibc >= 2.34. conda's
        #     ld then satisfies it from its cos6 sysroot with a glibc 2.12
        #     libdl, whose _dl_vsym/_dl_addr/_dl_sym@GLIBC_PRIVATE the host
        #     glibc does not provide:
        #       undefined reference to `_dl_vsym@GLIBC_PRIVATE'
        #     Only executables fail this way -- undefined symbols are legal in
        #     a shared object -- which is why the CUDA extension links fine and
        #     pbrt's little eLut/toFloat tools are what break.
        #     (On older hosts it instead surfaces as memcpy@GLIBC_2.14.)
        #
        #   * CXXFLAGS forces -std=c++17, under which the dynamic exception
        #     specifications in pbrt's bundled OpenEXR are a hard error.
        #
        # Scrubbing those four variables and preferring the system binutils
        # fixes both. cmake/make are resolved first so the PATH change cannot
        # silently swap them for different system copies.
        PBRT_CMAKE="$(command -v cmake)"
        PBRT_MAKE="$(command -v make)"
        rm -rf external/pbrt_build
        ( cd external && mkdir -p pbrt_build && cd pbrt_build \
          && env -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS \
                 PATH="/usr/bin:/bin:$PATH" \
             "$PBRT_CMAKE" ../pbrt-v3 -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_C_COMPILER=/usr/bin/gcc \
               -DCMAKE_CXX_COMPILER=/usr/bin/g++ >/dev/null \
          && env -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS \
                 PATH="/usr/bin:/bin:$PATH" \
             "$PBRT_MAKE" -j"$JOBS" >/dev/null )
    fi

    # display.py shells out to a bare `ffmpeg`; the system one usually breaks
    # under conda's libraries (undefined symbol: FT_Get_Transform).
    if [[ ! -e "$CONDA_PREFIX/bin/ffmpeg" ]]; then
        info "installing a static ffmpeg into the environment"
        python -m pip install --quiet imageio-ffmpeg==0.4.9
        FFMPEG_BIN="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
        ln -sfn "$FFMPEG_BIN" "$CONDA_PREFIX/bin/ffmpeg"
    else
        info "ffmpeg already available in the environment"
    fi
else
    step "Renderer skipped (--no-renderer)"
fi

# ── 7. Make the package importable ───────────────────────────────────────────
step "Registering py_diff_pd on the Python path"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$ROOT/python" > "$SITE/py_diff_pd.pth"
info "wrote $SITE/py_diff_pd.pth"

python -c "import py_diff_pd.core.py_diff_pd_core" \
    || die "py_diff_pd still does not import"

# ── Done ─────────────────────────────────────────────────────────────────────
printf '\n\033[1;32m==> Installation complete.\033[0m\n\n'
cat <<'EOM'
Verify it with:

    cd python/example
    python napkin_3d_test.py             # both solvers, with rendering
    python napkin_3d_test.py --no-vis    # simulation only, ~10 s

EOM
