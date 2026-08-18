#pragma once
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/STriangleMesh.h"
#include "tbb/tbb.h"

namespace RealSim::integrator::projectivedynamics::energy
{

// Base implementation of triangle stretching energies in PD
//
// Original Author: Ziqiu ZENG
//
class PDTriangleStretchingEnergy : public ElasticEnergy
{

public:
    PDTriangleStretchingEnergy(): ElasticEnergy(){}

    void setGeometry(geometry::SBaseGeometry * geometry) override;

    void init() override;

    unsigned int getVertexDimension() override {return _geometry->getPositions()->rows();}

    unsigned int getElementDimension() override {return _geometry->getTriangles()->rows();}

    void accumulateMatrix(MatXR &matrix) override;

    void accumulateMatrix(std::vector<Eigen::Triplet<real>> &triplets, real coeff) override;

    // 1. project the local energy
    // 2. build right-hand-sides with the auxiliary variables
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    unsigned int getSize() override { return getElementDimension();}

protected:
    std::vector<real> _area; // triangle area
    std::vector<Mat2R> _restMatrix; // Rest pose matrix
    std::vector<Mat2x3R> _proj;

    std::shared_ptr<geometry::STriangleMesh> _geometry;

    virtual Mat3x2R PDProjection(const Mat3x2R & F)
    {
        std::cout<<"PDTriangleStretchingEnergy::PDProjection not implemented"<<std::endl;
        return Mat3x2R::Zero();
    }

    void computeDmInv();

    void computeDeformationGradient(Mat3x2R & F, const MatX3R & pos, Index tetraId);

};


}
