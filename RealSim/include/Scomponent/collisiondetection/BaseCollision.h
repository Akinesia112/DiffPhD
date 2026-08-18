#pragma once
#include <iostream>
#include "types.h"
#include "Scomponent/contact/ContactPair.h"
#include "Sgeometry/SBaseGeometry.h"

namespace RealSim::collisiondetection
{

enum TypeCollision {
    PlaneContact,
    PlaneContactWithFriction
};

class BaseCollision
{
public:
    virtual void init() {}

    virtual void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos, const MatX3R &pos1, real dt)
    {
        std::cout<<"BaseCollision::detect() not implemented"<<std::endl;
    }

    static void generateTangentDirections(Vec3R& out0, Vec3R& out1, const Vec3R& in)
    {
        Vec3R tmp(1.0, 0.0, 0.0);
        if(in == tmp) tmp = Vec3R(0.0, 1.0, 0.0);

        out0 = in.cross(tmp);
        out0.normalize();

        out1 = in.cross(out0);
        out1.normalize();
    }
};

} // namespace RealSim::collisiondetection
