#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Base implementation of Tetrahedron energies in PD
//
// Original Author: Ziqiu ZENG
//
class PDTetrahedronEnergy : public ElasticEnergy
{

public:
    PDTetrahedronEnergy(): ElasticEnergy(){}

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

protected:
    std::vector<real> _vol; //volume
    std::vector<Mat3R> _DmInv; //Dm^(-1)
    std::vector<Mat3R> _Q; // Q = Dm^(-1) * Dm^(-T)
    std::vector<Mat3R> _proj;

    std::shared_ptr<geometry::STetrahedraMesh> _geometry;

    virtual Mat3R PDProjection(const Mat3R& F)
    {
        std::cout<<"PDTetrahedronEnergy::PDProjection not implemented"<<std::endl;
        return Mat3R::Zero();
    }

    void computeDmInv();

    void computeDeformationGradient(Mat3R & F, const MatX3R & pos, Index tetraId);

};


}
