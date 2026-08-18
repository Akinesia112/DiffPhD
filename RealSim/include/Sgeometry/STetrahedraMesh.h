#pragma once

#include "Sgeometry/STriangleMesh.h"

namespace RealSim::geometry
{

// Geometry Sturcture for Tetrahedra Mesh
//
// Original Author: Ziqiu ZENG
// part of functions imported from SOFA framework
class STetrahedraMesh : public STriangleMesh
{

public:
    STetrahedraMesh(bool dynamic, const VecXI& vertices, const MatXR& positions, MatX4I tetrahedra = InvalidTetrahedronSet, MatX3I triangles = InvalidTriangleSet, MatX2I edges = InvalidEdgeSet)
    : STriangleMesh(dynamic, vertices, positions, std::move(triangles), std::move(edges)), _tetrahedra(std::move(tetrahedra)){}

    void init() override;

    void clearTopology() override;

    const MatX4I * getTetrahedra() { return &_tetrahedra; }

    unsigned int getNumTetrahedra() { return _tetrahedra.rows(); }

    const VecXI * getSurfacePoints() override;

    const MatX2I * getSurfaceEdges() override;

    const MatX3I * getSurfaceTriangles() override;

    bool checkClosedSurface() { return (*getBoundaryEdges() == InvalidEdgeSet); }

    const std::vector<Index> & getTetrahedraAroundVertex(Index i);

    Index getTetrahedronIndex(Index v1, Index v2, Index v3,  Index v4);

    const std::vector<std::vector<Index>> & getEdgesInTetrahedronArray();

    const std::vector<std::vector<Index>> & getTrianglesInTetrahedronArray();

protected:
    MatX4I _tetrahedra;

    MatX3I _surfTriangles;
    MatX2I _surfEdges;
    VecXI _surfPoints;

    std::vector<Index> _surfTrianglesID;
    std::vector<Index> _surfEdgesID;

    std::vector<std::vector<Index>> _tetrahedraAroundVertex;

    std::vector<std::vector<Index>> _edgesInTetrahedron;
    std::vector<std::vector<Index>> _tetrahedraAroundEdge;

    std::vector<std::vector<Index>> _trianglesInTetrahedron;
    std::vector<std::vector<Index>> _tetrahedraAroundTriangle;

    static void addNewTriangle(std::vector<TriType> &vector_tris, Index n0, Index n1, Index n2);

    void generateTrianglesFromTetrahedra();

    void createTetrahedraAroundVertexArray ();

    void createEdgesInTetrahedronArray ();

    void createTetrahedraAroundEdgeArray ();

    void createTrianglesInTetrahedronArray ();

    void createTetrahedraAroundTriangleArray ();

    void computeSurfaceEdgesAndPoints();

    static Index findP(const TriType& tri, const TetType& tet);

    TriType ensureOutward(const TriType& tri, const TetType& tet);

};

}

