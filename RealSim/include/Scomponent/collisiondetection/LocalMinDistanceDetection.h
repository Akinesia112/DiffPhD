#pragma once
#include "Scomponent/collisiondetection/BaseCollision.h"
#include "Sgeometry/SBaseGeometry.h"
#include "Sgeometry/SEdgeMesh.h"
#include "Sgeometry/STriangleMesh.h"

namespace RealSim::collisiondetection::localmindistance
{

bool vertexFaceDCD(Vec3R &normal,
                   real &alpha,
                   real &beta,
                   const Vec3R &v,
                   const Vec3R &f0,
                   const Vec3R &f1,
                   const Vec3R &f2,
                   real alarmDist);

bool edgeEdgeDCD(Vec3R &normal,
                 real &alpha,
                 real &beta,
                 const Vec3R &ea0,
                 const Vec3R &ea1,
                 const Vec3R &eb0,
                 const Vec3R &eb1,
                 real alarmDist);

}
