#pragma once
#include <utility>

#include "Scomponent/lagrange/BaseController.h"
#include "Scomponent/tools/barycentricmapper/TetrahedronBarycentricMapper.h"

namespace RealSim::lagrange
{

// Implementation of cable constraint controller
//
// Original Author: Ziqiu ZENG
// inspired by SOFA SoftRobots - CableConstraint
class CableController : public BaseController
{
public:
    CableController(){}

    CableController(MatX3R pos, real invStiff = 0.0)
    :_initpos(std::move(pos)), _invStiff(invStiff)
    {
        step = 0;
    }

    real objectiveTotalLength;
    unsigned int step;

    void init() override
    {
        _initTotalLength = 0.0;
        std::vector<real> localLength;
        real diff;
        for(Index p=1; p<_initpos.rows(); p+=2)
        {
            Vec3R v01 = _initpos.row(p) - _initpos.row(p-1);

            real length = v01.norm();
            localLength.push_back(length);
            _initTotalLength += length;
        }

        _ratio.clear();
        for(auto length : localLength)
        {
            _ratio.push_back(length/_initTotalLength);
        }

        objectiveTotalLength = _initTotalLength;
        _lastLength = _initTotalLength;

        _maxLength = _initTotalLength*1.0; // no push!
        _minLength = _initTotalLength*0.85;

        begin.clear();          end.clear();            increment.clear();
        begin.push_back(0);     end.push_back(1600);    increment.push_back(-0.01);
        begin.push_back(1450);  end.push_back(2200);    increment.push_back(0.01);

    }

    void mappingToTetrahedraMesh(const MatX3R& tetpos, const MatX4I& tetra)
    {
        _mapping.clear();
        _mapping.computeBarycentricCoordinate(tetra, tetpos, _initpos);
    }

    // set control info
    void set(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos) override
    {
//        std::cout<<"CableController::set"<<std::endl;

        real length = _lastLength;

        for(unsigned k=0; k<begin.size(); k++)
        {
            if(step >= begin[k] && step < end[k])
            {
                length += increment[k];
            }
        }

        if(length>_maxLength) length = _maxLength;
        if(length<_minLength) length = _minLength;

        objectiveTotalLength = length;

//        std::cout<<"length = "<<length<<std::endl;

        _lastLength = 0.0;

        auto barycoord = _mapping.getBarycentricCoordinates();
        auto indices = _mapping.getVertexIndices();

        for(Index p=1; p<_initpos.rows(); p+=2)
        {
            real objectLength = _ratio[(p-1)/2] * objectiveTotalLength;

            const TetType & indicesE0 = indices.row(p-1);
            const TetType & indicesE1 = indices.row(p);

            const Vec4R & barycoordE0 = barycoord.row(p-1);
            const Vec4R & barycoordE1 = barycoord.row(p);

            const Vec3R p0 = barycoordE0[0]*pos.row(indicesE0[0]) + barycoordE0[1]*pos.row(indicesE0[1]) + barycoordE0[2]*pos.row(indicesE0[2]) + barycoordE0[3]*pos.row(indicesE0[3]);
            const Vec3R p1 = barycoordE1[0]*pos.row(indicesE1[0]) + barycoordE1[1]*pos.row(indicesE1[1]) + barycoordE1[2]*pos.row(indicesE1[2]) + barycoordE1[3]*pos.row(indicesE1[3]);

            Vec3R v01 = p1 - p0;
            real length = v01.norm();
            Vec3R direction = v01 / length;

            VecXI index(8);
            VecXR coeff(8);
            index<<indicesE0[0], indicesE0[1], indicesE0[2], indicesE0[3], indicesE1[0], indicesE1[1], indicesE1[2], indicesE1[3];
            coeff<<-barycoordE0[0], -barycoordE0[1], -barycoordE0[2], -barycoordE0[3], barycoordE1[0], barycoordE1[1], barycoordE1[2], barycoordE1[3];

            RealSim::contact::ContactPair pair(RealSim::contact::ContactPair::bilateral, _invStiff);
            pair.addBasis(8, index, coeff, direction, direction*objectLength);
            output.push_back(pair);

            _lastLength += length;
        }

        step++;
    }

protected:
    MatX3R _initpos;

    real _invStiff; // making the constraint as a soft one, set _invStiff = 0 to have a hard constraint

    std::vector<real> _ratio;
    real _initTotalLength;

    real _maxLength;
    real _minLength;

    real _lastLength;

    RealSim::tools::TetrahedronBarycentricMapping _mapping;

    std::vector<unsigned int> begin, end;
    std::vector<real> increment;

};

}
