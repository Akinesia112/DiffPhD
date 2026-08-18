#pragma once

#include "types.h"
#include "printconfig.h"


namespace RealSim::geometry
{


// Geometry Sturcture for BASE Mesh
//
// Original Author: Ziqiu ZENG
class SBaseGeometry
{
public:
    SBaseGeometry(bool dynamic, VecXI vertices, const MatXR& positions)
    : _dynamic(dynamic), _vertices(std::move(vertices)), _positions(positions) {}

    virtual void init() = 0; //build topology structure

    virtual void clearTopology() = 0;

    virtual const VecXI * getSurfacePoints() = 0;

    virtual const MatX2I * getSurfaceEdges() = 0;

    virtual const MatX3I * getSurfaceTriangles() = 0;

    const VecXI * getVertices() const { return &_vertices; }

    const MatX3R * getPositions() const { return &_positions; }

    unsigned int getNumVertices() const { return _positions.rows(); }

    bool isDynamic() const { return _dynamic; }

protected:
    bool _dynamic;

    VecXI _vertices;
    MatX3R _positions;

};

}