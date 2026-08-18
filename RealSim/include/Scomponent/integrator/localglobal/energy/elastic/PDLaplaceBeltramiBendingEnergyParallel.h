#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDLaplaceBeltramiBendingEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct LaplaceBeltramiProjectionInRange {
    std::vector<Vec3R> &res;
    const MatX3R &pos;
    const std::vector<std::vector<Index>> &indices;
    const std::vector<std::vector<real>> &beltrami_coefficients;
    const std::vector<real> & norm;
public:
    LaplaceBeltramiProjectionInRange(std::vector<Vec3R> &_res,
                                     const MatX3R &_pos,
                                     const std::vector<std::vector<Index>> &_indices,
                                     const std::vector<std::vector<real>> &_beltrami_coefficients,
                                     const std::vector<real> & _norm)
            :res(_res),
             pos(_pos),
             indices(_indices),
             beltrami_coefficients(_beltrami_coefficients),
             norm(_norm){}

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
            if (norm[i] > 1e-6)
            {
                const auto & _indices = indices[i];
                const auto & _beltrami_coefficients = beltrami_coefficients[i];

                Vec3R e;
                e.setZero();

                for(unsigned j = 0; j <_indices.size(); j++)
                {
                    e += _beltrami_coefficients[j] * pos.row(_indices[j]);
                }

                res[i] = e.normalized() * norm[i];
            }
        }
    }
};

// Implementation of Laplace Beltrami bending energy for (triangle) cloth simulations
//
// Original Author: Ziqiu ZENG
//
class PDLaplaceBeltramiBendingEnergyParallel : public PDLaplaceBeltramiBendingEnergy
{

public:
    PDLaplaceBeltramiBendingEnergyParallel(): PDLaplaceBeltramiBendingEnergy()
    {
        std::cout<<GREEN<<"Bending Energy Model: Laplace Beltrami Bending (Parallel Ver.)"<<RESET<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        real wi = _weight * coeff;

        tbb::parallel_for(tbb::blocked_range<size_t>(0,getSize()), LaplaceBeltramiProjectionInRange(_e, pos, _indices, _beltrami_coefficients, _norm));

        for(unsigned i = 0 ; i<getSize(); i++)
        {
            // If the laplace beltrami is the null vector we do not normalize it
            if (_norm[i] > 1e-6)
            {
                const auto & indices = _indices[i];
                const auto & beltrami_coefficients = _beltrami_coefficients[i];

                for(unsigned j = 0; j <indices.size(); j++)
                {
                    rhs.row(indices[j]) += wi * beltrami_coefficients[j] * _e[i];
                }
            }
        }
    }
};


}
