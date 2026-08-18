#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <iostream>
#include <memory>
#include <cstring>
#include <vector>
#include <utility>
#include <set>
#include <unordered_set>

namespace RealSim
{
    typedef double real;
    typedef int32_t intType;

    // vector types
    typedef Eigen::Vector<intType, Eigen::Dynamic> VecXI;

    typedef Eigen::Vector<real, 2> Vec2R;
    typedef Eigen::Vector<real, 3> Vec3R;
    typedef Eigen::Vector<real, 4> Vec4R;
    typedef Eigen::Vector<real, Eigen::Dynamic> VecXR;

    // matrix types
    typedef Eigen::Matrix<intType, Eigen::Dynamic, Eigen::Dynamic> MatXI;
    typedef Eigen::Matrix<intType, Eigen::Dynamic, 2> MatX2I;
    typedef Eigen::Matrix<intType, Eigen::Dynamic, 3> MatX3I;
    typedef Eigen::Matrix<intType, Eigen::Dynamic, 4> MatX4I;

    typedef Eigen::Matrix<real, 2, 3> Mat2x3R;
    typedef Eigen::Matrix<real, 3, 2> Mat3x2R;
    typedef Eigen::Matrix<real, 4, 3> Mat4x3R;
    typedef Eigen::Matrix<real, 3, 4> Mat3x4R;
    typedef Eigen::Matrix<real, 2, 2> Mat2R;
    typedef Eigen::Matrix<real, 3, 3> Mat3R;
    typedef Eigen::Matrix<real, 4, 4> Mat4R;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 3> MatX3R;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 4> MatX4R;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> MatXR;

    typedef Eigen::SparseMatrix<real> SpMatR;

    // for geometry
    typedef uint32_t Index;
    typedef Eigen::Vector2i EdgeType;
    typedef Eigen::Vector3i TriType;
    typedef Eigen::Vector4i TetType;

    inline static const std::vector<Index> InvalidSet;
    constexpr Index InvalidID = (std::numeric_limits<Index>::max)();

    inline static const VecXI InvalidPointSet;
    inline static const MatX2I InvalidEdgeSet;
    inline static const MatX3I InvalidTriangleSet;
    inline static const MatX4I InvalidTetrahedronSet;

    const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
}
