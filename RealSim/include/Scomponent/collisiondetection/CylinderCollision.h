#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static and dynamic (rolling) cylinder collision
//
// Original Author: Ziqiu ZENG
//
class CylinderCollision : public BaseCollision
{
public:
    CylinderCollision(){}

    CylinderCollision(Vec3R base, Vec3R axis, real radius, real avel, real mu)
        :_base(std::move(base)), _axis(std::move(axis)), _radius(radius), _avel(avel), _mu(mu)
    {
        _detectAll = true;
    }

    void init() override;

    void set(const Vec3R& base, const Vec3R& orientation, real radius);

    void addCollisionObject(RealSim::geometry::SBaseGeometry * mesh);

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real dt) override;

protected:
    std::vector<const VecXI* > _vertices;

    Vec3R _base;       ///< A point in the plane
    Vec3R _axis;       ///< Orientation of the cylinder axis
    real _radius;
    real _avel; /// angle velocity for rolling cylinders

    real _mu;

    bool _detectAll;

    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p0, const Vec3R &p1, Index pid, real dt);

};

}
