#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class SphereCollision : public BaseCollision
{
public:
    SphereCollision(){}

    SphereCollision(Vec3R center, real radius, real mu)
        :_center(std::move(center)), _radius(radius), _mu(mu)
    {}

    void addCollisionObject(RealSim::geometry::SBaseGeometry * mesh);

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real /*dt*/) override;

protected:
    Vec3R _center;
    real _radius;
    real _mu;

    std::vector<const VecXI* > _vertices;

private:
    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p0, const Vec3R &p1, Index pid);

};

}
