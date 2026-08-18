#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/STetrahedraMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of edge spring energies in PD
//
// Original Author: Ziqiu ZENG
//
class PDEdgeSpringEnergy : public ElasticEnergy
{

public:
    PDEdgeSpringEnergy(): ElasticEnergy()
    {
        std::cout<<GREEN<<"Elastic Energy Model: Spring Edge Energy"<<RESET<<std::endl;
    }

    void setGeometry(geometry::SBaseGeometry * geometry) override;
    
    void init() override;
    
    unsigned int getVertexDimension() override {return _geometry->getPositions()->rows();}

    unsigned int getElementDimension() override {return _geometry->getEdges()->rows();}

    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override { return getElementDimension();}

protected:
    std::vector<real> _length; // rest length
    std::vector<Vec3R> _normal; // rest length
    std::shared_ptr<geometry::SEdgeMesh> _geometry;

    void computeRestLength();
};


}
