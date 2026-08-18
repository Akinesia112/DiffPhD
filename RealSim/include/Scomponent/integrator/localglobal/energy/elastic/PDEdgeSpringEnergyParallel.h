#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDEdgeSpringEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct SpringEdgeProjectionInRange {
    std::vector<Vec3R> &res;
    const MatX3R &pos;
    const MatX2I &edges;
    real coeff;
    real weight;
public:
    SpringEdgeProjectionInRange(std::vector<Vec3R> &_res,
                                  const MatX3R &_pos,
                                  const MatX2I &_edges)
            :res(_res),
             pos(_pos),
             edges(_edges){}

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
            const EdgeType &edge = edges.row(i);

            Vec3R spring = pos.row(edge[1]) - pos.row(edge[0]);
            res[i] = spring.normalized();
        }
    }
};

// Implementation of edge spring energies in PD
//
// Original Author: Ziqiu ZENG
//
class PDEdgeSpringEnergyParallel : public PDEdgeSpringEnergy
{

public:
    PDEdgeSpringEnergyParallel(): PDEdgeSpringEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Spring Edge Energy (Parallel Ver.)"<<RESET<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), SpringEdgeProjectionInRange(_normal, pos, *_geometry->getEdges()));

        for(unsigned int  i = 0 ; i<getElementDimension(); i++)
        {
            real wi = coeff * this->_weight;

            const EdgeType &edge = _geometry->getEdges()->row(i);

            rhs.row(edge[0]) += - wi * 0.5 * _length[i] * _normal[i];
            rhs.row(edge[1]) += wi * 0.5 * _length[i] * _normal[i];
        }
    }
};


}
