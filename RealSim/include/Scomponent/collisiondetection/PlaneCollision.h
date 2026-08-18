#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class PlaneCollision : public BaseCollision
{
public:
    PlaneCollision(){}

    PlaneCollision(Vec3R base, Vec3R orientation, real mu)
        :_base(std::move(base)), _orientation(std::move(orientation)), _mu(mu)
    {}

    void init() override;

    void addCollisionObject(RealSim::geometry::SBaseGeometry * mesh);

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real /*dt*/) override;

protected:
    Vec3R _base;       ///< A point in the plane
    Vec3R _orientation;       ///< The normal to the plane. Will be normalized by init().
    real _mu;

    Vec3R _tangent0;       ///< The tangent direction. Will be normalized by init().
    Vec3R _tangent1;       ///< The tangent direction. Will be normalized by init().

    std::vector<const VecXI* > _vertices;

    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p0, const Vec3R &p1, Index pid);

};

}
