cmake_minimum_required(VERSION 3.18)

project(diff_pd LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
# Pardiso requires stack variables to be aligned, hence
# -mpreferred-stack-boundary=4. This magic number is suitable only for
# libpardiso600-GNU720-X86-64.so
add_definitions("-Wall -mpreferred-stack-boundary=4")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
# Force system libdl/libpthread to avoid conda sysroot GLIBC_PRIVATE conflicts.
# --allow-shlib-undefined lets the CUDA .so load its internal deps at runtime.
set(CUDA_LINK_FLAGS "-L/lib/x86_64-linux-gnu -ldl -lpthread -lrt -Wl,--allow-shlib-undefined")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CUDA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${CUDA_LINK_FLAGS}")

# CUDA toolkit (cuBLAS, cuSPARSE, cudart).
find_package(CUDAToolkit REQUIRED)

# OpenMP.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# ── RealSim sparse LDLT (truly sparse L^{-1} via METIS nested dissection) ──
set(REALSIM_DIR "${CMAKE_SOURCE_DIR}/../RealSim")
set(REALSIM_DEPS "${CMAKE_SOURCE_DIR}/../RealSim/deps")
include_directories("${REALSIM_DIR}/include")
include_directories("${REALSIM_DEPS}/METIS_install/include")
include_directories("${REALSIM_DEPS}/METIS/GKlib_build/include")
# METIS type width definitions (required by metis.h)
add_definitions(-DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=32)
include_directories("${REALSIM_DEPS}/METIS/GKlib_build/include")  # needed for idx_t etc.
include_directories("${REALSIM_DEPS}/onetbb_install/include")
set(REALSIM_SRCS
    "${REALSIM_DIR}/src/Scomponent/tools/math/SparseLDLT.cpp"
    "${REALSIM_DIR}/src/Scomponent/tools/math/SparseMatrix.cpp"
)
set(REALSIM_LIBS
    "${REALSIM_DEPS}/METIS_install/lib/libmetis.a"
    "${REALSIM_DEPS}/METIS/GKlib_build/lib/libGKlib.a"
)
# TBB: use find_library so CMake doesn't treat it as a build target
find_library(TBB_LIB tbb
    HINTS "${REALSIM_DEPS}/onetbb_install/lib"
    NO_DEFAULT_PATH)
if(TBB_LIB)
    message(STATUS "TBB found: ${TBB_LIB}")
    list(APPEND REALSIM_LIBS ${TBB_LIB})
else()
    message(WARNING "TBB not found in onetbb_install/lib")
endif()
add_definitions(-DREALSIM_SPARSE_LDLT)

# cuSPARSE / cuBLAS / cuDart  (C API only — no nvcc needed)
# Use conda environment's CUDA 12 libraries (headers + .so all match).
set(CONDA_LIB "$ENV{CONDA_PREFIX}/lib")
set(CONDA_INC "$ENV{CONDA_PREFIX}/include")

set(CUSPARSE_LIB "${CONDA_LIB}/libcusparse.so.12.3.1.170")
set(CUBLAS_LIB   "${CONDA_LIB}/libcublas.so.12.4.5.8")
set(CUDART_LIB   "${CONDA_LIB}/libcudart.so.12.4.127")

if(EXISTS "${CUSPARSE_LIB}" AND EXISTS "${CUDART_LIB}")
    message(STATUS "cuSPARSE found: ${CUSPARSE_LIB}")
    message(STATUS "cuBLAS   found: ${CUBLAS_LIB}")
    message(STATUS "cudart   found: ${CUDART_LIB}")
    add_definitions(-DCUDA_AVAILABLE)
    include_directories("${CONDA_INC}")
    set(CUDA_LIBS ${CUSPARSE_LIB} ${CUDART_LIB})
    if(EXISTS "${CUBLAS_LIB}")
        list(APPEND CUDA_LIBS ${CUBLAS_LIB})
    endif()
else()
    message(STATUS "cuSPARSE not found — GPU path disabled, using CPU fallback.")
    set(CUDA_LIBS "")
endif()

# Pardiso.
option(PARDISO_AVAILABLE "Pardiso" OFF)
if(PARDISO_AVAILABLE)
    add_definitions(-DPARDISO_AVAILABLE)
    add_library(pardiso SHARED IMPORTED)
    set_target_properties(pardiso PROPERTIES
        IMPORTED_LOCATION "$ENV{PARDISO_LIC_PATH}/libpardiso600-GNU720-X86-64.so"
    )
else()
    message("Pardiso not detected.")
endif()

# Include directories.
include_directories("../external/eigen")
include_directories("./core/include")
include_directories("../external/preconditioner-for-cloth-and-deformable-body-simulation")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# MAS Preconditioner library.
set(MAS_DIR "../external/preconditioner-for-cloth-and-deformable-body-simulation")
set(MAS_SOURCES
    ${MAS_DIR}/SeSchwarzPreconditioner.cpp
    ${MAS_DIR}/SeOmp.cpp
)
# Enable AVX2+FMA only for MAS source files (not the whole project).
set_source_files_properties(${MAS_SOURCES} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

# CUDA GPU CG solver.
set(CUDA_SOURCES "./core/src/solver/cuda_mas_cg_solver.cu")

# The core library.
file(GLOB_RECURSE CORE_CPP "./core/src/**/*.cpp")
add_library(diff_pd_core SHARED ${CORE_CPP} ${REALSIM_SRCS} ${MAS_SOURCES} ${CUDA_SOURCES})
if(PARDISO_AVAILABLE)
    target_link_libraries(diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m ${CUDA_LIBS} ${REALSIM_LIBS}
                          CUDA::cublas CUDA::cusparse CUDA::cudart)
else()
    target_link_libraries(diff_pd_core OpenMP::OpenMP_CXX ${CUDA_LIBS} ${REALSIM_LIBS}
                          CUDA::cublas CUDA::cusparse CUDA::cudart)
endif()

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./core/src/**/*.cpp" "./core/src/py_diff_pd_core_wrap.cxx")
add_library(py_diff_pd_core SHARED ${CORE_HEADER} ${PYTHON_BINDING} ${REALSIM_SRCS} ${MAS_SOURCES} ${CUDA_SOURCES})
if(PARDISO_AVAILABLE)
    target_link_libraries(py_diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m ${CUDA_LIBS} ${REALSIM_LIBS}
                          CUDA::cublas CUDA::cusparse CUDA::cudart)
else()
    target_link_libraries(py_diff_pd_core OpenMP::OpenMP_CXX ${CUDA_LIBS} ${REALSIM_LIBS}
                          CUDA::cublas CUDA::cusparse CUDA::cudart)
endif()

# Auto-copy built .so to Python module directory after every build.
set(PY_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python/py_diff_pd/core")
add_custom_command(TARGET py_diff_pd_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:py_diff_pd_core>
        "${PY_MODULE_DIR}/_py_diff_pd_core.so"
    COMMENT "Copying py_diff_pd_core → python/py_diff_pd/core/_py_diff_pd_core.so"
)

# Demo.
file(GLOB DEMO_CPP "./main.cpp")
add_executable(diff_pd_demo ${DEMO_CPP})
target_link_libraries(diff_pd_demo diff_pd_core
    -L/lib/x86_64-linux-gnu -ldl -lpthread -lrt)