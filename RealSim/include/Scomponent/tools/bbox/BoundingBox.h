#pragma once
#include "types.h"

namespace RealSim::tools
{

class BoundingBox
{

public:
    virtual void init()
    {
        std::cout<<"BoundingBox::init() not implemented"<<std::endl;
    }

    virtual void draw()
    {
        std::cout<<"BoundingBox::draw() not implemented"<<std::endl;
    }

};

}