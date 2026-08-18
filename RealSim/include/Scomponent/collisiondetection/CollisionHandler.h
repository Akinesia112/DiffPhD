#pragma once
#include <memory>

#include "Scomponent/collisiondetection/BaseCollision.h"

#include "Scomponent/collisiondetection/GenericCCD.h"
#include "Scomponent/collisiondetection/GenericDCD.h"
#include "Scomponent/collisiondetection/PlaneCollision.h"
#include "Scomponent/collisiondetection/CylinderCollision.h"
#include "Scomponent/collisiondetection/BoxCollision.h"
#include "Scomponent/collisiondetection/SphereCollision.h"
#include "Scomponent/collisiondetection/TorusCollision.h"
#include "Scomponent/collisiondetection/MovingBoxCollision.h"
#include "Scomponent/collisiondetection/MovingSphereCollision.h"

namespace RealSim::collisiondetection
{

class CollisionHandler
{
public:
    void clearCollisionSet()
    {
        _collisionset.clear();
    }

    void init()
    {
        for(const auto & collision: _collisionset)
        {
            collision->init();
        }
    }

    void addCollision(BaseCollision * collision)
    {
        _collisionset.push_back(static_cast<const std::shared_ptr<BaseCollision>>(collision));
    }

    void addPlaneCollision(const Vec3R& origin, const Vec3R& normal, real mu)
    {
        auto c = std::make_shared<RealSim::collisiondetection::PlaneCollision>(origin, normal, mu);
        _collisionset.push_back(c);
    }

    void addCylinderCollision(const Vec3R& origin, const Vec3R& axis, real radius, real avel, real mu)
    {
        auto c = std::make_shared<RealSim::collisiondetection::CylinderCollision>(origin, axis, radius, avel, mu);
        _collisionset.push_back(c);
    }

    void addSphereCollision(const Vec3R& center, real radius, real mu)
    {
        auto c = std::make_shared<RealSim::collisiondetection::SphereCollision>(center, radius, mu);
        _collisionset.push_back(c);
    }

    void addTorusCollision(const Vec3R& center, const Vec3R& axis, real radius_outer, real radius_inner, real mu)
    {
        auto c = std::make_shared<RealSim::collisiondetection::TorusCollision>(center, axis, radius_outer, radius_inner, mu);
        _collisionset.push_back(c);
    }

    void addMovingSphereCollision(const Vec3R& center, real radius, real mu,
                                  const std::vector<Vec3R> &direction,
                                  const std::vector<real> &velocity,
                                  const std::vector<real> &max_displacement)
    {
        auto c = std::make_shared<RealSim::collisiondetection::MovingSphereCollision>(center, radius, mu, direction, velocity, max_displacement);
        _collisionset.push_back(c);
    }

    void addBoxCollision(const Vec3R& center, const Vec3R& rotation, const Vec3R& scale, real mu)
    {
        auto c = std::make_shared<RealSim::collisiondetection::BoxCollision>(center, rotation, scale, mu);
        _collisionset.push_back(c);
    }

    void addMovingBoxCollision(const Vec3R &center, const Vec3R &rotation,
                               const Vec3R &scale, real mu,
                               const std::vector<Vec3R> &direction,
                               const std::vector<real> &velocity,
                               const std::vector<real> &max_displacement) {
        auto c = std::make_shared<RealSim::collisiondetection::MovingBoxCollision>(
            center, rotation, scale, mu, direction, velocity, max_displacement);
        _collisionset.push_back(c);
    }

    void addGenericDCD(real alarm_distance, real minimum_separation, bool VF, bool EE, bool FV)
    {
        auto c = std::make_shared<RealSim::collisiondetection::GenericDCD>(alarm_distance, minimum_separation, VF, EE, FV);
        _collisionset.push_back(c);
    }

    void addGenericCCD(real alarm_distance, real minimum_separation, bool VF, bool EE, bool FV)
    {
        auto c = std::make_shared<RealSim::collisiondetection::GenericCCD>(alarm_distance, minimum_separation, VF, EE, FV);
        _collisionset.push_back(c);
    }

    void doCollision(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real dt)
    {
        for(const auto & collision: _collisionset)
        {
            collision->detect(output, pos0, pos1, dt);
        }
    }

    const std::vector<std::shared_ptr<BaseCollision>> & collisionSet()
    {
        return _collisionset;
    }

protected:
    std::vector<std::shared_ptr<BaseCollision>> _collisionset;
};

}
