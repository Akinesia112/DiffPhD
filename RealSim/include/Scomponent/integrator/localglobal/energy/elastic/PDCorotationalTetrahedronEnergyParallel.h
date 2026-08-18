#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTetrahedronEnergy.h"
#include "Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "tbb/tbb.h"
#include "Scomponent/tools/math/SVD.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct CorotationalTetrahedronProjectionInRange {
    std::vector<Mat3R> &res;
    const MatX3R &pos;
    const MatX4I &tetrahedra;
    real coeff;
    real weight;
    const std::vector<real> & vol;
    const std::vector<Mat3R> & DmInv;
    const std::vector<CorotProjectionProblem3D> & problem;
    const std::vector<mcl::optlib::LBFGS<double,3>> & solver;
public:
    CorotationalTetrahedronProjectionInRange(std::vector<Mat3R> &_res,
                                           const MatX3R &_pos,
                                           const MatX4I &_tetrahedra,
                                           real _coeff,
                                           real _weight,
                                           const std::vector<real> & _vol,
                                           const std::vector<Mat3R> & _DmInv,
                                           const std::vector<CorotProjectionProblem3D> & _problem,
                                           const std::vector<mcl::optlib::LBFGS<double,3>> & _solver)
            :res(_res),
             pos(_pos),
             tetrahedra(_tetrahedra),
             coeff(_coeff),
             weight(_weight),
             vol(_vol),
             DmInv(_DmInv),
             problem(_problem),
             solver(_solver){}

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
            real wi = coeff * weight * vol[i];

            Mat3R Ds;

            const TetType &tetra = tetrahedra.row(i);
            Index a = tetra[0];
            Index b = tetra[1];
            Index c = tetra[2];
            Index d = tetra[3];

            Ds.col(0) = pos.row(b)-pos.row(a);
            Ds.col(1) = pos.row(c)-pos.row(a);
            Ds.col(2) = pos.row(d)-pos.row(a);

            Mat3R F = Ds * DmInv[i];

            Mat3R U, V;
            Vec3R s;
            RealSim::tools::svd::signedEigenSVD(F, U, s, V);

            Eigen::Vector3d sigma = s.cast<double>();
            Eigen::Vector3d sigma0 = sigma;

            const real eps = 1e-6;
            if (std::abs(sigma[0]) < eps && std::abs(sigma[1]) < eps && std::abs(sigma[2]) < eps) {
                sigma[0] = eps;
                sigma[1] = eps;
                sigma[2] = eps;
            }
            if( sigma[2] < 0.0 )
            {
                sigma[2] = -sigma[2];
            }

            solver[i].minimize(problem[i], sigma, sigma0);

            Mat3R P = U * (sigma.cast<real>()).asDiagonal() * V.transpose();

            res[i] = wi * (DmInv[i] * P.transpose()); // tmp = w_i * D^T * R^T
        }
    }
};


// Implementation of tetrahedron energies with Corotational elasticity
//
// Original Author: Ziqiu ZENG
//
class PDCorotationalTetrahedronEnergyParallel : public PDTetrahedronEnergy
{

public:
    PDCorotationalTetrahedronEnergyParallel(): PDTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Neo-Hookean Tetrahedron Energy (Parallel Ver.)"<<RESET<<std::endl;
    }

    void init() override
    {
        PDTetrahedronEnergy::init();

        _problem.resize(getElementDimension());
        _solver.resize(getElementDimension());
        for(auto & problem : _problem)
        {
            problem.set_lame(this->_mu, this->_lambda);
        }
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), CorotationalTetrahedronProjectionInRange(_proj, pos, *_geometry->getTetrahedra(), coeff, this->_weight, _vol, _DmInv, _problem, _solver));

        for(unsigned int  i = 0 ; i<getElementDimension(); i++)
        {
            const TetType &tetra = _geometry->getTetrahedra()->row(i);

            rhs.row(tetra[0]) += - _proj[i].row(0) -  _proj[i].row(1) - _proj[i].row(2);
            rhs.row(tetra[1]) += _proj[i].row(0);
            rhs.row(tetra[2]) += _proj[i].row(1);
            rhs.row(tetra[3]) += _proj[i].row(2);
        }
    }

protected:
    std::vector<CorotProjectionProblem3D> _problem;
    std::vector<mcl::optlib::LBFGS<double,3>> _solver;
};


}
