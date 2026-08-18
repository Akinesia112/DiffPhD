#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.h"
#include "Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct CorotationalTriangleProjectionInRange {
    std::vector<Mat2x3R> &res;
    const MatX3R &pos;
    const MatX3I &triangles;
    real coeff;
    real weight;
    const std::vector<real> & area;
    const std::vector<Mat2R> & restMatrix;
    const std::vector<CorotProjectionProblem2D> & problem;
    const std::vector<mcl::optlib::LBFGS<double,2>> & solver;

public:
    CorotationalTriangleProjectionInRange(std::vector<Mat2x3R> &_res,
                                        const MatX3R &_pos,
                                        const MatX3I &_triangles,
                                        real _coeff,
                                        real _weight,
                                        const std::vector<real> & _area,
                                        const std::vector<Mat2R> & _restMatrix,
                                        const std::vector<CorotProjectionProblem2D> & _problem,
                                        const std::vector<mcl::optlib::LBFGS<double,2>> & _solver)
            :res(_res),
             pos(_pos),
             triangles(_triangles),
             coeff(_coeff),
             weight(_weight),
             area(_area),
             restMatrix(_restMatrix),
             problem(_problem),
             solver(_solver){}

    void operator()(const tbb::blocked_range<size_t>& r) const
    {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
            real wi = coeff * weight * area[i];

            Mat3x2R Ds;

            const TriType &tri = triangles.row(i);
            Index a = tri[0];
            Index b = tri[1];
            Index c = tri[2];

            Ds.col(0) = pos.row(b)-pos.row(a);
            Ds.col(1) = pos.row(c)-pos.row(a);

            Mat3x2R F = Ds * restMatrix[i];

            Eigen::JacobiSVD<Mat3x2R > svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

            const Vec2R& s= svd.singularValues();
            Eigen::Vector2d sigma = s.cast<double>();
            Eigen::Vector2d sigma0 = sigma;

            // L-BFGS
            // If everything is very low, It is collapsed to a point and the minimize
            // will likely fail. So we'll just inflate it a bit.
            const real eps = 1e-6;
            if (std::abs(sigma[0]) < eps && std::abs(sigma[1]) < eps) {
                sigma[0] = eps;
                sigma[1] = eps;
            }

            solver[i].minimize(problem[i], sigma, sigma0);

            Mat3x2R P = svd.matrixU() * (sigma.cast<real>()).asDiagonal() * svd.matrixV().transpose();

            res[i] = wi * (restMatrix[i] * P.transpose()); // tmp = w_i * D^T * R^T
        }
    }
};


// Implementation of triangle stretching energies with Corotational elasticity
//
// Original Author: Ziqiu ZENG
//
class PDCorotationalTriangleEnergyParallel : public PDTriangleStretchingEnergy
{

public:
    PDCorotationalTriangleEnergyParallel(): PDTriangleStretchingEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Corotational Triangle Energy (Parallel Ver.)"<<RESET<<std::endl;
    }

    void init() override
    {
        PDTriangleStretchingEnergy::init();

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
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), CorotationalTriangleProjectionInRange(_proj, pos, *_geometry->getTriangles(), coeff, this->_weight, _area, _restMatrix, _problem, _solver));

        for(unsigned int  i = 0 ; i<getElementDimension(); i++)
        {
            const TriType &tri = _geometry->getTriangles()->row(i);

            rhs.row(tri[0]) += - _proj[i].row(0) -  _proj[i].row(1);
            rhs.row(tri[1]) += _proj[i].row(0);
            rhs.row(tri[2]) += _proj[i].row(1);
        }
    }

    std::vector<CorotProjectionProblem2D> _problem;
    std::vector<mcl::optlib::LBFGS<double,2>> _solver;
};


}
