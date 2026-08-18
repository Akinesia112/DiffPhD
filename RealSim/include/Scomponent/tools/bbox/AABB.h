#pragma once
#include <Scomponent/tools/bbox/BoundingBox.h>
#include "Sgeometry/SBaseGeometry.h"

namespace RealSim::tools
{

class AxisAlignedBoundingBox : public BoundingBox
{

public:
    AxisAlignedBoundingBox(const std::vector<real> &box, geometry::SBaseGeometry *geometry)
    {
        setAignedBox(box);
        const MatX3R & pos = *geometry->getPositions();
        computeVerticesInside(pos);

        // TODO: compute topology structures inside
    }

    const VecXI& verticesInside()
    {
        return _indices;
    }

protected:
    Vec3R _maxBBox, _minBBox;
    VecXI _indices;

    void setAignedBox(const std::vector<real> &box)
    {
        if(box.size() != 6)
        {
            std::cerr<<"aligned box input error, dimension != 6 "<<std::endl;
        }

        const real max_real = std::numeric_limits<real>::max();
        const real min_real = std::numeric_limits<real>::lowest();
        _maxBBox = {min_real,min_real,min_real};
        _minBBox = {max_real,max_real,max_real};

        if (box[0] < _minBBox[0]) _minBBox[0] = box[0];
        if (box[1] < _minBBox[1]) _minBBox[1] = box[1];
        if (box[2] < _minBBox[2]) _minBBox[2] = box[2];
        if (box[3] > _maxBBox[0]) _maxBBox[0] = box[3];
        if (box[4] > _maxBBox[1]) _maxBBox[1] = box[4];
        if (box[5] > _maxBBox[2]) _maxBBox[2] = box[5];
    }

    void computeVerticesInside(const MatX3R & pos)
    {
//        std::cout<<"pos = "<<pos<<std::endl;

        std::vector<int> indices;
        for( unsigned i=0; i<pos.rows(); i++ )
        {
            if (isPointInside(pos.row(i)))
            {
                indices.push_back(i);
            }
        }

        _indices = Eigen::Map<VecXI, Eigen::Unaligned>(indices.data(), indices.size());
    }

    bool isPointInside(const Vec3R& p)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            if (p[i] < _minBBox[i] || p[i] > _maxBBox[i])
            {
                return false;
            }
        }
        return true;
    }
};

}