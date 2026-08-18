#pragma once

#include "Sgeometry/SBaseGeometry.h"

namespace RealSim::geometry
{

//    Geometry Sturcture for Particle Collections
class SParticles : public SBaseGeometry
{

public:
    SParticles(bool dynamic, const VecXI& vertices, MatXR positions, real radius)
            : SBaseGeometry(dynamic, vertices, positions), _radius(radius){}

    real getRadius() { return _radius; }

    void init() override {}

    void clearTopology() override {}

    const VecXI * getSurfacePoints() override { return &_vertices;}

    const MatX2I * getSurfaceEdges() override { return nullptr;}

    const MatX3I * getSurfaceTriangles() override { return nullptr;}

protected:
    real _radius;

};

}

