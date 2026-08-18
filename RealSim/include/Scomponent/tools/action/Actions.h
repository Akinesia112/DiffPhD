#pragma once
#include <iostream>
#include <memory>
#include "Eigen/Dense"

namespace RealSim::tools::action
{

class Action
{
public:
    Action(real _diff, real _max) :diff(_diff), max(_max)
    {
        current = 0.0;
        max = std::abs(max);
    }

    real diff;
    real max;
    real current;

    virtual bool isActive() {return (current < max && current > -max);}
    virtual Vec3R action(const Vec3R & pos) = 0;
    virtual void update() { current += diff; }
};

class PullingAction: public Action
{
public:
    PullingAction(real _diff, real _max, Vec3R _direction):
    Action(_diff, _max), direction(_direction){}

    Vec3R direction;

    Vec3R action(const Vec3R & pos) override
    {
        Vec3R res = pos + diff * direction;
        return res;
    }
};

class RollingAction: public Action
{
public:
    RollingAction(real _diff, real _max, Vec3R _center, Vec3R _axis):
    Action(_diff, _max), center(_center), axis(_axis)
    {
        if( _axis.norm() != 0 ) _axis.normalize();

        K.setZero();
        K(0,1) = -_axis[2];
        K(0,2) = _axis[1];
        K(1,0) = _axis[2];
        K(1,2) = -_axis[0];
        K(2,0) = -_axis[1];
        K(2,1) = _axis[0];
    }

    Vec3R center;
    Vec3R axis;

    Vec3R action(const Vec3R & pos) override
    {
        Vec3R base = center + axis.dot( pos - center ) * axis;
        real rad = diff * M_PI / 180.0;
        Mat3R R = Mat3R::Identity() + sin(rad) * K + (1 - cos(rad)) * K;

        Vec3R res = base + R * (pos - base);
        return res;
    }

protected:
    Mat3R K;
};

}
