#pragma once
#include "types.h"

namespace RealSim::tools::laplacebeltrami
{

std::vector<real> getLaplaceBeltramiDiscretisationMeanValueCoefficients(const Vec3R& middle_vertex, const std::vector<Vec3R>& ring_vertices);

real getMeanValue(const Vec3R& v1, const Vec3R& v2, const Vec3R& v3);

real getSine(const Vec3R& v1, const Vec3R& v2, const Vec3R& v3);

real getCosine(const Vec3R& v1, const Vec3R& v2, const Vec3R& v3);

}