#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class TorusCollision : public BaseCollision
{
public:
    TorusCollision(){}

    TorusCollision(Vec3R center, Vec3R axis, real radiusOuter, real radiusInner, real mu)
        :_center(std::move(center)), _axis(axis), _radiusOuter(radiusOuter), _radiusInner(radiusInner), _mu(mu)
    {}

    void addCollisionObject(RealSim::geometry::SBaseGeometry * mesh);

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real /*dt*/) override;

protected:
    Vec3R _center;
    Vec3R _axis;
    real _radiusInner;
    real _radiusOuter;
    real _mu;

    std::vector<const VecXI* > _vertices;

private:
    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p0, const Vec3R &p1, Index pid);

};

}
