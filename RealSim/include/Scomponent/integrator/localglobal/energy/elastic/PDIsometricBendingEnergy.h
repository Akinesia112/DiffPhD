#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of isometric bending energies for (triangle) cloth simulations
//
// Original Author: Ziqiu ZENG
//
class PDIsometricBendingEnergy : public ElasticEnergy
{

public:
    PDIsometricBendingEnergy(): ElasticEnergy()
    {
//        std::cout<<GREEN<<"Bending Energy Model: Isometric Bending"<<RESET<<std::endl;
    }

    void setGeometry(geometry::SBaseGeometry * geometry) override;

    void setBendingParameter(real bending) {this->_weight = bending;}

    void init() override;

    unsigned int getVertexDimension() override {return _geometry->getPositions()->rows();}

    unsigned int getElementDimension() override {return _geometry->getTriangles()->rows();}

    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override { return _indices.size();}

protected:
    std::shared_ptr<geometry::STriangleMesh> _geometry;

    std::vector<std::vector<Index>> _indices;
    std::vector<real> _A0;
    std::vector<real> _A1;
    std::vector<std::vector<real>> _weightVertex;
    std::vector<real> _norm;

    std::vector<Vec3R> _e;

    void computeRestBendingEnergy(Index eid);
    Index findThirdIndex(Index tid, Index pid0, Index pid1);

};


}
