#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDIsometricBendingEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct IsometricBendingProjectionInRange {
    std::vector<Vec3R> &res;
    const MatX3R &pos;
    const std::vector<std::vector<Index>> &indices;
    const std::vector<std::vector<real>> &weightVertex;
    const std::vector<real> & norm;
public:
    IsometricBendingProjectionInRange(std::vector<Vec3R> &_res,
                                      const MatX3R &_pos,
                                      const std::vector<std::vector<Index>> &_indices,
                                      const std::vector<std::vector<real>> &_weightVertex,
                                      const std::vector<real> & _norm)
            :res(_res),
             pos(_pos),
             indices(_indices),
             weightVertex(_weightVertex),
             norm(_norm){}

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
            if (norm[i] > 1e-6)
            {
                const auto & _indices = indices[i];
                const auto & _weightVertex = weightVertex[i];

                Vec3R e;
                e.setZero();
                for(unsigned j=0; j<4; j++) e += _weightVertex[j] * pos.row(_indices[j]);

                res[i] = e.normalized() * norm[i];
            }
        }
    }
};

// Implementation of isometric bending energies for (triangle) cloth simulations
//
// Original Author: Ziqiu ZENG
//
class PDIsometricBendingEnergyParallel : public PDIsometricBendingEnergy
{

public:
    PDIsometricBendingEnergyParallel(): PDIsometricBendingEnergy()
    {
        std::cout<<GREEN<<"Bending Energy Model: Isometric Bending (Parallel Ver.)"<<RESET<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        real wi = _weight * coeff;

        tbb::parallel_for(tbb::blocked_range<size_t>(0,getSize()), IsometricBendingProjectionInRange(_e, pos, _indices, _weightVertex, _norm));

        for(unsigned i = 0 ; i<getSize(); i++)
        {
            // If the laplace beltrami is the null vector we do not normalize it
            if(_norm[i] > 1e-6)
            {
                const auto & indices = _indices[i];
                const auto & weightVertex = _weightVertex[i];

                real wi = coeff * this->_weight * 3.0 / (_A0[i] + _A1[i]);

                rhs.row(indices[0]) += wi * weightVertex[0] * _e[i];
                rhs.row(indices[1]) += wi * weightVertex[1] * _e[i];
                rhs.row(indices[2]) += wi * weightVertex[2] * _e[i];
                rhs.row(indices[3]) += wi * weightVertex[3] * _e[i];
            }
        }
    }
};


}
