#pragma once

#include <iostream>
#include <string>

#include "igl/massmatrix.h"

namespace RealSim
{
    class Scomponent;
    class Sgeometry;


    class Sxpbddamping
    {   
        public:
            Sxpbddamping(std::shared_ptr<Scomponent> component, std::shared_ptr<Sgeometry> geometry)
                : component_(std::move(component)), geometry_(std::move(geometry)) {}
            ~Sxpbddamping() {}

            void ImportDamping(std::string &refdamping) {}
            void GenerateDampingFromType() {}

        private:
            MatXR DampingMap;
            std::shared_ptr<Scomponent> component_;
            std::shared_ptr<Sgeometry> geometry_;
    };
}