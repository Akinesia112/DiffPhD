#pragma once
#include <utility>

#include "Scomponent/collisiondetection/SphereCollision.h"

namespace RealSim::collisiondetection
{

// Implementation of static plane collision
//
// Original Author: Ziqiu ZENG
//
class MovingSphereCollision : public SphereCollision
{
public:
    MovingSphereCollision(){}

    MovingSphereCollision(Vec3R center, real radius, real mu,
                          const std::vector<Vec3R> & direction,
                          const std::vector<real> & velocity,
                          const std::vector<real> & max_displacement)
        : SphereCollision(std::move(center), radius, mu),
          _direction(direction),
          _velocity(velocity),
          _max_displacement(max_displacement) {
        _current_displacement.resize(_direction.size(), 0.0);
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
    void detect_impl(std::vector<RealSim::contact::ContactPair> &output, const Vec3R &p0, const Vec3R &p1, Index pid, const Vec3R& displacement);

};

}
