#pragma once
#include <memory>
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Scomponent/collisiondetection/BaseCollision.h"

namespace RealSim::lagrange
{

// Implementation of general lagrange constraints
//
// Original Author: Ziqiu ZENG
//
class LagrangeConstraint
{
public:
    unsigned int getConstraintsInfo(const std::vector<RealSim::contact::ContactPair> & contactPairSet)
    {
        num_pairs = contactPairSet.size();
        ctype.clear();
        csize.clear();
        num_constraints = 0;
        for(const auto & pair : contactPairSet){
            ctype.push_back(pair.type());
            csize.push_back(pair.size());
            mu.emplace_back(pair.mu);
            num_constraints += pair.size();
        }
        return num_constraints;
    }

    static void buildJacobian(std::vector<Eigen::Triplet<real>> &triplets, const std::vector<RealSim::contact::ContactPair> & contactPairSet)
    {
        unsigned int cstId = 0;
        for(const auto & pair : contactPairSet){
            for (int j = 0; j < pair.size(); j++) {
                const RealSim::contact::LocalBasis & cinfo = pair[j];
                const int * dof = cinfo._index.data();
                const real * coeff = cinfo._coeff.data();
                const Vec3R & dir = cinfo._dir;
                for(int k=0; k<cinfo._n; k++)
                {
                    triplets.emplace_back(cstId, dof[k]*3, coeff[k] * dir[0]);
                    triplets.emplace_back(cstId, dof[k]*3+1, coeff[k] * dir[1]);
                    triplets.emplace_back(cstId, dof[k]*3+2, coeff[k] * dir[2]);
                }
                cstId++;
            }
        }
    }

    //pene_0 = d
    static void initPenetration(VecXR & pene0, const std::vector<RealSim::contact::ContactPair> & contactPairSet)
    {
        unsigned int cstId = 0;
        for(const auto & pair : contactPairSet) {
            for (int j = 0; j < pair.size(); j++) {
                const RealSim::contact::LocalBasis & cinfo = pair[j];
                pene0[cstId] = cinfo._dir.dot(cinfo._point); //pay attention to the sign
                cstId++;
            }
        }
    }

public:
    int num_constraints;
    int num_pairs;

    std::vector<RealSim::contact::ContactPair::TypePair> ctype;
    std::vector<int> csize;
    std::vector<real> mu;
};


}
