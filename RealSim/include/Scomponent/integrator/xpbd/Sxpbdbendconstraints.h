#pragma once

#include <iostream>
#include <string>

#include "igl/massmatrix.h"

namespace RealSim
{
    class Scomponent;
    class Sgeometry;
    class Sxpbdstiffness;
    class Sxpbddamping;


    class Sxbpdbendconstraints
    {   
        public:
           Sxbpdbendconstraints(std::shared_ptr<Sxpbdstiffness> stiffness, std::shared_ptr<Sxpbddamping> damping, std::shared_ptr<Scomponent> component, std::shared_ptr<Sgeometry> geometry)
                : stiffness_(std::move(stiffness)), damping_(std::move(damping)), component_(std::move(component)), geometry_(std::move(geometry)) {}
            ~Sxbpdbendconstraints() {}

            void Update() 
            {

            }
            void CalForce() {}

        private:
            std::shared_ptr<Sxpbdstiffness> stiffness_;
            std::shared_ptr<Sxpbddamping> damping_;
            std::shared_ptr<Scomponent> component_;
            std::shared_ptr<Sgeometry> geometry_;
    };
}