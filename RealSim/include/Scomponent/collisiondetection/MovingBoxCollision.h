#pragma once
#include <utility>

#include "Scomponent/collisiondetection/BoxCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class MovingBoxCollision : public BoxCollision
{
public:
    MovingBoxCollision(){}

    MovingBoxCollision(Vec3R center, Vec3R rotation, Vec3R scale, real mu,
                       const std::vector<Vec3R> &direction,
                       const std::vector<real> &velocity,
                       const std::vector<real> &max_displacement)
        : BoxCollision(std::move(center), std::move(rotation), std::move(scale), mu),
          _direction(direction),
          _velocity(velocity),
          _max_displacement(max_displacement) {
    }

    void init() override;

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real /*dt*/) override;

protected:
    std::vector<Vec3R> _direction;
    std::vector<real> _velocity;
    std::vector<real> _max_displacement;
    std::vector<real> _current_displacement;

private:
    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p_t0, const Vec3R &p_t1, Index pid, const Vec3R& displacement);

};

}
