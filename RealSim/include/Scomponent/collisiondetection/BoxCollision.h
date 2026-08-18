#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class BoxCollision : public BaseCollision
{
public:
    BoxCollision(){}

    BoxCollision(Vec3R center, Vec3R rotation, Vec3R scale, real mu)
        :_center(std::move(center)), _rotation(std::move(rotation)), _scale(std::move(scale)), _mu(mu)
    {}

    void init() override;

    void addCollisionObject(RealSim::geometry::SBaseGeometry * mesh);

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real /*dt*/) override;

protected:
    Vec3R _center;
    Vec3R _rotation;
    Vec3R _scale;

    Vec3R _orient0;
    Vec3R _orient1;
    Vec3R _orient2;

    real _mu;

    std::vector<const VecXI* > _vertices;

private:
    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p_t0, const Vec3R &p_t1, Index pid);

};

}
