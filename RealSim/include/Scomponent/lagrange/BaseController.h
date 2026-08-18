#pragma once
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Scomponent/contact/ContactPair.h"

namespace RealSim::lagrange
{

class BaseController
{
public:
    virtual void init()
    {
//        std::cout<<"BaseController::init() not implemented"<<std::endl;
    }

    virtual void set(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos)
    {
        std::cout<<"BaseController::set() not implemented"<<std::endl;
    }
};
}
