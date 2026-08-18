#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/PDTetrahedronEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

struct ARAPTetrahedronProjectionInRange {
    std::vector<Mat3R> &res;
    const MatX3R &pos;
    const MatX4I &tetrahedra;
    real coeff;
    real weight;
    const std::vector<real> & vol;
    const std::vector<Mat3R> & DmInv;
public:
    ARAPTetrahedronProjectionInRange(std::vector<Mat3R> &_res,
                                     const MatX3R &_pos,
                                     const MatX4I &_tetrahedra,
                                     real _coeff,
                                     real _weight,
                                     const std::vector<real> & _vol,
                                     const std::vector<Mat3R> & _DmInv)
            :res(_res),
             pos(_pos),
             tetrahedra(_tetrahedra),
             coeff(_coeff),
             weight(_weight),
             vol(_vol),
             DmInv(_DmInv){}

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

            Eigen::JacobiSVD<Mat3R> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Mat3R U = svd.matrixU();
            Mat3R V = svd.matrixV();

            Mat3R J = Mat3R::Identity();
            J(2, 2) = -1.0;

            if(U.determinant() < 0.0)
            {
                U = U * J;
            }

            if(V.determinant() < 0.0)
            {
                V = (J * V.transpose()).transpose();
            }

            Mat3R P = U * V.transpose();

            res[i] = wi * (DmInv[i] * P.transpose()); // tmp = w_i * D^T * R^T
        }
    }
};


// Implementation of tetrahedron energies with Neo-hookean elasticity
//
// Original Author: Ziqiu ZENG
//
class PDARAPTetrahedronEnergyParallel : public PDTetrahedronEnergy
{

public:
    PDARAPTetrahedronEnergyParallel(): PDTetrahedronEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: ARAP Tetrahedron Energy (Parallel Ver.)"<<RESET<<std::endl;
    }

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0,getElementDimension()), ARAPTetrahedronProjectionInRange(_proj, pos, *_geometry->getTetrahedra(), coeff, this->_weight, _vol, _DmInv));

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

            E += 0.5 * wi * (s - Vec3R::Identity()).squaredNorm();
        }

        return E;
    }


};


}
