#pragma once
#include "types.h"

namespace RealSim::integrator
{

class Sactor;

class MechanicalState
{

public:
    void clear(unsigned int dim)
    {
        //allocation for mechanical states
        _position = MatXR::Zero(dim, 3);
        _refposition = MatXR::Zero(dim, 3);
        _velocity = MatXR::Zero(dim, 3);
        _forces = MatXR::Zero(dim, 3);
    }

    unsigned int _dim;
    MatX3R _position;
    MatX3R _refposition;
    MatX3R _velocity;
    MatX3R _forces;

    real _dt;
};

class SBaseIntegrator
{

public:
    virtual void init(const MatX3R & initpos)
    {
        std::cout<<"SBaseIntegrator::init() not implemented"<<std::endl;
    }

    virtual void update(MatXR &pos)
    {
        std::cout<<"SBaseIntegrator::update() not implemented"<<std::endl;
    }

    virtual void onDestroy()
    {
        std::cout<<"SBaseIntegrator::onDestroy() not implemented"<<std::endl;
    }

    Sactor *actor = nullptr;
};

}