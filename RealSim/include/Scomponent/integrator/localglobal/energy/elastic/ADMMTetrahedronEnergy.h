#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of Tetrahedron energies in ADMM-PD (Overby et al. 2017)
//
// Original Author: Ziqiu ZENG
//
class ADMMTetrahedronEnergy : public ElasticEnergy
{

public:
    ADMMTetrahedronEnergy(): ElasticEnergy(){}

    void setGeometry(geometry::SBaseGeometry * geometry) override;

    void init() override;

    unsigned int getVertexDimension() override {return _geometry->getPositions()->rows();}

    unsigned int getElementDimension() override {return _geometry->getTetrahedra()->rows();}

    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override { return getElementDimension();}

    void reset() override
    {
        _u.clear();
        _u.resize(getElementDimension(), Mat3R::Zero());
    }

protected:
    std::vector<Mat3R> _p, _z, _u;
    std::vector<real> _vol; //volume
    std::vector<Mat3x4R> _D; //volume

    std::shared_ptr<geometry::STetrahedraMesh> _geometry;

    void computeReduction();

    virtual Mat3R ADMMProjection(const Mat3R& z)
    {
        std::cout<<"ADMMTetrahedronEnergy::ADMMProjection not implemented"<<std::endl;
        return Mat3R::Zero();
    }


};


}
