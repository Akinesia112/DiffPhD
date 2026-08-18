#pragma once

#include "Sgeometry/SEdgeMesh.h"

namespace RealSim::geometry
{

// Geometry Sturcture for Tetrahedra Mesh
//
// Original Author: Ziqiu ZENG
// part of functions imported from SOFA framework
class STriangleMesh : public SEdgeMesh
{

public:
    STriangleMesh(bool dynamic, const VecXI& vertices, const MatXR& positions, MatX3I triangles = InvalidTriangleSet, MatX2I edges = InvalidEdgeSet)
    : SEdgeMesh(dynamic, vertices, positions, std::move(edges)), _triangles(std::move(triangles)) {}

    void init() override;

    void clearTopology() override;

    const VecXI * getSurfacePoints() override { return &_vertices;}

    const MatX2I * getSurfaceEdges() override { return &_edges;}

    const MatX3I * getSurfaceTriangles() override { return &_triangles;}

    const MatX3I * getTriangles() { return &_triangles; }

    unsigned int getNumTriangles() { return _triangles.rows(); }
    
    const MatX2I * getBoundaryEdges();

    const std::vector<Index> & getBoundaryEdgesID() {return _boundaryEdgesID;}

    const VecXI * getBoundaryPoints();

    bool checkClosedBoundary() { return (*getEndPoints() == InvalidPointSet); }

    const std::vector<Index> & getTrianglesAroundVertex(Index i);

    const std::vector<Index> & getTrianglesAroundEdge(Index i);

    Index getTriangleIndex(Index v1, Index v2, Index v3);

    const std::vector<std::vector<Index>> & getEdgesInTriangleArray();

protected:
    MatX3I _triangles;
    MatX2I _boundaryEdges;
    VecXI _boundaryPoints;

    std::vector<Index> _boundaryEdgesID;

    std::vector<std::vector<Index>> _trianglesAroundVertex;

    std::vector<std::vector<Index>> _edgesInTriangle;
    std::vector<std::vector<Index>> _trianglesAroundEdge;

    static void addNewEdge(std::vector<EdgeType> &vector_edges, Index n0, Index n1);

    void generateEdgesFromTriangles();

    void createTrianglesAroundVertexArray();

    void createEdgesInTriangleArray();

    void createTrianglesAroundEdgeArray();

    void computeBoundaryPoints();

};

}

