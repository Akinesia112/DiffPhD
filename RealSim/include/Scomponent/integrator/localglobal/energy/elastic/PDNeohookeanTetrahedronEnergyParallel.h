#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTetrahedronEnergy.h"
#include "Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct NeohookeanTetrahedronProjectionInRange {
    std::vector<Mat3R> &res;
    const MatX3R &pos;
    const MatX4I &tetrahedra;
    real coeff;
    real weight;
    const std::vector<real> & vol;
    const std::vector<Mat3R> & DmInv;
    const std::vector<NHProjectionProblem3D> & problem;
    const std::vector<mcl::optlib::LBFGS<double,3>> & solver;
public:
    NeohookeanTetrahedronProjectionInRange(std::vector<Mat3R> &_res,
                                           const MatX3R &_pos,
                                           const MatX4I &_tetrahedra,
                                           real _coeff,
                                           real _weight,
                                           const std::vector<real> & _vol,
                                           const std::vector<Mat3R> & _DmInv,
                                           const std::vector<NHProjectionProblem3D> & _problem,
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

            Mat3R U, V, S;
            Vec3R s;
            S.setZero();
            RealSim::tools::svd::signedEigenSVD(F, U, s, V);
//            RealSim::tools::svd::signedMcAdamSVD(F, U, s, V, S);

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

            S(0,0) = sigma[0];
            S(1,1) = sigma[1];
            S(2,2) = sigma[2];

            Mat3R P = U * S * V.transpose();

            res[i] = wi * (DmInv[i] * P.transpose()); // tmp = w_i * D^T * R^T
        }
    }
};


// Implementation of tetrahedron energies with Neo-hookean elasticity
//
// Original Author: Ziqiu ZENG
//
class PDNeohookeanTetrahedronEnergyParallel : public PDTetrahedronEnergy
{

public:
    PDNeohookeanTetrahedronEnergyParallel(): PDTetrahedronEnergy()
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
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), NeohookeanTetrahedronProjectionInRange(_proj, pos, *_geometry->getTetrahedra(), coeff, this->_weight, _vol, _DmInv, _problem, _solver));

        for(unsigned int  i = 0 ; i<getElementDimension(); i++)
        {
            const TetType &tetra = _geometry->getTetrahedra()->row(i);

            rhs.row(tetra[0]) += - _proj[i].row(0) -  _proj[i].row(1) - _proj[i].row(2);
            rhs.row(tetra[1]) += _proj[i].row(0);
            rhs.row(tetra[2]) += _proj[i].row(1);
            rhs.row(tetra[3]) += _proj[i].row(2);
        }
    }

    real getCurrentEnergy(const MatX3R &pos) override
    {
        real E = 0.0;

        Mat3R F;
        real wi;

        for(unsigned int  i = 0 ; i<getElementDimension(); i++) {
            const Eigen::Vector4i &tetra = _geometry->getTetrahedra()->row(i);

            wi = this->_weight * _vol[i];

            computeDeformationGradient(F, pos, i);

            Mat3R U, V, R;
            Vec3R s;
            RealSim::tools::svd::signedEigenSVD(F, U, s, V);

            E += 0.5 * wi * _problem[i].energy_density(s);
        }

        return E;
    }

protected:
    std::vector<NHProjectionProblem3D> _problem;
    std::vector<mcl::optlib::LBFGS<double,3>> _solver;
};


}
