#pragma once
#include <memory>

namespace RealSim::contact
{

class LocalBasis
{
public:
    LocalBasis(int n, VecXI index, VecXR coeff, Vec3R dir, Vec3R point)
            :_n(n), _index(index), _coeff(coeff), _dir(dir), _point(point){}

    int _n; //number of impacted dofs
    VecXI _index; //impacted dofs
    VecXR _coeff; //coefficient

    Vec3R _dir; //contact normal
    Vec3R _point; //contact point
};

class ContactPair
{
public:
    enum TypePair {
        bilateral,
        unilateral,
        friction,
        none
    };

    ContactPair(TypePair t, real _mu):_type(t), mu(_mu){}

    unsigned int size() const {return _localBasis.size();}

    void clear() {_localBasis.clear();}

    TypePair type() const {return _type;}

    void addBasis(LocalBasis basis)
    {
        _localBasis.push_back(basis);
    }

    void addBasis(int n, VecXI index, VecXR coeff, Vec3R dir, Vec3R point)
    {
        _localBasis.push_back(LocalBasis(n, index, coeff, dir, point));
    }

    const LocalBasis & operator[](int index) const {return _localBasis[index];}

    real mu; // frictional coeff (for friction), or inverse stiff (for bilateral)

protected:
    std::vector<LocalBasis> _localBasis;

    TypePair _type;
};


}
