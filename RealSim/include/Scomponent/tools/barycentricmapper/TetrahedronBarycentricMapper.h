#pragma once
#include "types.h"
#include "Sgeometry/STetrahedraMesh.h"

namespace RealSim::tools
{

// Implementation of Tetrahedron Barycentric mapping component
//
// Original Author: Ziqiu ZENG
//
class TetrahedronBarycentricMapping
{

public:
    void clear()
    {
        _invbase.clear();
        _center.clear();
        _tetraIndices.clear();
    }

    void computeBarycentricCoordinate(const MatX4I & inputTopology, const MatX3R &inPos, const MatX3R &outPos)
    {
        _barycoord.resize(outPos.rows(), 4);
        _indices.resize(outPos.rows(), 4);
        _tetraIndices.clear();

        for(Index i=0; i<outPos.rows(); i++)
        {
            const Vec3R out = outPos.row(i);

            Vec4R coord;
            Index tetId = findClosestTetrahedronImpl2(coord, inputTopology, inPos,out);
            _tetraIndices.push_back(tetId);
            _indices.row(i) = inputTopology.row(tetId);
            _barycoord.row(i) = coord;
        }
    }

    const std::vector<Index>& getTetrahedronIndices()
    {
        return _tetraIndices;
    }

    const MatX4I& getVertexIndices()
    {
        return _indices;
    }

    const MatX4R& getBarycentricCoordinates()
    {
        return _barycoord;
    }

protected:
    MatX4R _barycoord;
    std::vector<Index> _tetraIndices;
    MatX4I _indices;

    std::vector<Mat3R> _invbase;
    std::vector<Vec3R> _center;

    static real tetrahedronVolume(const Vec3R& p0, const Vec3R& p1, const Vec3R& p2, const Vec3R& p3)
    {
        real vol = (p3-p0).dot((p2-p0).cross(p1-p0));
        return vol/6.0;
    }

    static real distancePointToPlane(const Vec3R& p, const Vec3R& p0, const Vec3R& p1, const Vec3R& p2) {
        Vec3R normal = (p1 - p0).cross(p2 - p0);

        real d = - p0.dot(normal);
        real distance = (p.dot(normal) + d) / sqrt(normal.dot(normal));

        return fabs(distance);
    }

    static Index findClosestTetrahedronImpl1(Vec4R & coord, const MatX4I & inputTopology, const MatX3R &inPos, const Vec3R &p)
    {
        //iterate in tetrahedra set
        real minDist = std::numeric_limits<real>::max();
        coord = Vec4R(0.0, 0.0, 0.0, 0.0);
        Index tetraId = InvalidID;
        for(Index i=0; i<inputTopology.rows(); i++)
        {
            bool inside = true;

            const TetType tet = inputTopology.row(i);

            const Vec3R p0 = inPos.row(tet[0]);
            const Vec3R p1 = inPos.row(tet[1]);
            const Vec3R p2 = inPos.row(tet[2]);
            const Vec3R p3 = inPos.row(tet[3]);

            real tetrahedronVol = tetrahedronVolume(p0, p1, p2, p3);
            real vol1 = tetrahedronVolume(p, p1, p2, p3);
            real vol2 = tetrahedronVolume(p0, p, p2, p3);
            real vol3 = tetrahedronVolume(p0, p1, p, p3);
            real vol4 = tetrahedronVol - vol1 - vol2 - vol3;
            if(tetrahedronVol < 0.0)
            {
                tetrahedronVol = -tetrahedronVol;
                vol1 = -vol1;
                vol2 = -vol2;
                vol3 = -vol3;
                vol4 = -vol4;
            }

            Vec4R newCoord = Vec4R(vol1/tetrahedronVol, vol2/tetrahedronVol, vol3/tetrahedronVol, vol4/tetrahedronVol);

            if(vol1*tetrahedronVol < 0.0)
            {
                real dist = distancePointToPlane(p, p1, p2, p3);
                if(minDist<dist)
                {
                    minDist = dist;
                    tetraId = i;
                    coord = newCoord;
                    inside = false;
                }
            }

            if(vol2*tetrahedronVol < 0.0)
            {
                real dist = distancePointToPlane(p0, p, p2, p3);
                if(minDist<dist)
                {
                    minDist = dist;
                    tetraId = i;
                    coord = newCoord;
                    inside = false;
                }
            }

            if(vol3*tetrahedronVol < 0.0)
            {
                real dist = distancePointToPlane(p0, p1, p, p3);
                if(minDist<dist)
                {
                    minDist = dist;
                    tetraId = i;
                    coord = newCoord;
                    inside = false;
                }
            }

            if(vol4*tetrahedronVol < 0.0)
            {
                real dist = distancePointToPlane(p0, p1, p2, p);
                if(minDist<dist)
                {
                    minDist = dist;
                    tetraId = i;
                    coord = newCoord;
                    inside = false;
                }
            }

            if(inside)
            {
                tetraId = i;
                coord = newCoord;
                return tetraId;
            }
        }

        return tetraId;
    }

    static void computeBaseAndCenter(Mat3R & invbase, Vec3R & center, const MatX3R & in, const TetType & element)
    {
        Mat3R base;

        base.row(0) = in.row(element[1]) - in.row(element[0]);
        base.row(1) = in.row(element[2]) - in.row(element[0]);
        base.row(2) = in.row(element[3]) - in.row(element[0]);

        invbase = (base.transpose()).inverse();
        center = ( in.row(element[0]) + in.row(element[1]) + in.row(element[2]) + in.row(element[3]) ) * 0.25;
    }

    //Faster version, imported from SOFA
    Index findClosestTetrahedronImpl2(Vec4R & coord, const MatX4I & inputTopology, const MatX3R &inPos, const Vec3R &p)
    {
        if(_invbase.size() != inputTopology.rows() || _center.size() != inputTopology.rows())
        {
            _invbase.clear();
            _invbase.resize(inputTopology.rows());
            _center.clear();
            _center.resize(inputTopology.rows());

            for(Index i=0; i<inputTopology.rows(); i++) {
                computeBaseAndCenter(_invbase[i], _center[i], inPos, inputTopology.row(i));
            }
        }

        Vec3R coefs;
        real minDist = std::numeric_limits<real>::max();
        Index tetraId = InvalidID;

        for(Index i=0; i<inputTopology.rows(); i++)
        {
            TetType tet = inputTopology.row(i);
            const Vec3R p0 = inPos.row(tet[0]);
            Vec3R v = _invbase[i] * (p - p0);
            real d = std::max(std::max(-v[0], -v[1]), std::max(-v[2], v[0] + v[1] + v[2] - 1));

            if (d>0) d = (p - _center[i]).squaredNorm();

            if (d<minDist)
            {
                coefs = v;
                minDist = d;
                tetraId = i;
            }
        }

        coord[0] = 1.0 - coefs[0] - coefs[1] - coefs[2];
        coord[1] = coefs[0];
        coord[2] = coefs[1];
        coord[3] = coefs[2];

        return tetraId;
    }

};

}