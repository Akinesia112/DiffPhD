#pragma once

#include <iostream>
#include <string>

#include "igl/massmatrix.h"

namespace RealSim
{
    class Scomponent;
    class Sgeometry;


    class Sxpbdstiffness
    {   
        public:
            Sxpbdstiffness(std::shared_ptr<Scomponent> component, std::shared_ptr<Sgeometry> geometry)
            : component_(std::move(component)), geometry_(std::move(geometry)) {}
            ~Sxpbdstiffness() {}

            void ImportStiffness(std::string &refstiffness) {}
            void GenerateStiffnessFromType() {}

        private:
            MatXR StiffnessMap;
            std::shared_ptr<Scomponent> component_;
            std::shared_ptr<Sgeometry> geometry_;
    };
}