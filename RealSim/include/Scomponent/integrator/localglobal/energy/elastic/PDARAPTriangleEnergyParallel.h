#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct ARAPTriangleProjectionInRange {
    std::vector<Mat2x3R> &res;
    const MatX3R &pos;
    const MatX3I &triangles;
    real coeff;
    real weight;
    const std::vector<real> & area;
    const std::vector<Mat2R> & restMatrix;
public:
    ARAPTriangleProjectionInRange(std::vector<Mat2x3R> &_res,
                                  const MatX3R &_pos,
                                  const MatX3I &_triangles,
                                  real _coeff,
                                  real _weight,
                                  const std::vector<real> & _area,
                                  const std::vector<Mat2R> & _restMatrix)
            :res(_res),
             pos(_pos),
             triangles(_triangles),
             coeff(_coeff),
             weight(_weight),
             area(_area),
             restMatrix(_restMatrix){}

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
            Mat3x2R S = Mat3x2R::Zero();
            S.block<2,2>(0,0) = Mat2R::Identity();
            Mat3x2R P = svd.matrixU() * S * svd.matrixV().transpose();

            res[i] = wi * (restMatrix[i] * P.transpose()); // tmp = w_i * D^T * R^T
        }
    }
};


// Implementation of triangle stretching energies with ARAP model
//
// Original Author: Ziqiu ZENG
//
class PDARAPTriangleEnergyParallel : public PDTriangleStretchingEnergy
{

public:
    PDARAPTriangleEnergyParallel(): PDTriangleStretchingEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: ARAP Triangle Energy (Parallel Ver.)"<<RESET<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), ARAPTriangleProjectionInRange(_proj, pos, *_geometry->getTriangles(), coeff, this->_weight, _area, _restMatrix));

        for(unsigned int  i = 0 ; i<getElementDimension(); i++)
        {
            const TriType &tri = _geometry->getTriangles()->row(i);

            rhs.row(tri[0]) += - _proj[i].row(0) -  _proj[i].row(1);
            rhs.row(tri[1]) += _proj[i].row(0);
            rhs.row(tri[2]) += _proj[i].row(1);
        }
    }

    real getCurrentEnergy(const MatX3R &pos) override
    {
        real E = 0.0;

        Mat3x2R F;
        real wi;

        for(unsigned int  i = 0 ; i<getElementDimension(); i++) {

            wi = this->_weight * _area[i];

            computeDeformationGradient(F, pos, i);

            Eigen::JacobiSVD<Mat3x2R > svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

            const Vec2R& s= svd.singularValues();

            E += 0.5 * wi * (s - Vec2R::Identity()).squaredNorm();
        }

        return E;
    }

};


}
