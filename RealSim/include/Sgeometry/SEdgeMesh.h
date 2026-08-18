#pragma once

#include "Sgeometry/SBaseGeometry.h"

namespace RealSim::geometry
{

// Geometry Sturcture for Tetrahedra Mesh
//
// Original Author: Ziqiu ZENG
// part of functions imported from SOFA framework
class SEdgeMesh : public SBaseGeometry
{

public:
    SEdgeMesh(bool dynamic, const VecXI& vertices, const MatXR& positions, MatX2I edges = InvalidEdgeSet)
    : SBaseGeometry(dynamic, vertices, positions), _edges(std::move(edges)){}

    void init() override;

    void clearTopology() override;

    const VecXI * getSurfacePoints() override { return &_vertices;}

    const MatX2I * getSurfaceEdges() override { return &_edges;}

    const MatX3I * getSurfaceTriangles() override { return nullptr;}

    const MatX2I * getEdges() { return &_edges; }

    unsigned int getNumEdges() { return _edges.rows(); }

    const VecXI * getEndPoints();

    const std::vector<Index> & getEdgesAroundVertex(Index i);

    Index getEdgeIndex(Index v1, Index v2);

protected:
    MatX2I _edges;
    VecXI _endPoints;
    
    std::vector<std::vector<Index>> _edgesAroundVertex;

    void createEdgesAroundVertexArray();
};

}

