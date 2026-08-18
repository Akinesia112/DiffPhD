#pragma once
#include <MCL/Problem.hpp>
#include <MCL/LBFGS.hpp>
#include "types.h"

namespace RealSim::integrator::projectivedynamics::energy{

class ProjectionProblem3X3 : public mcl::optlib::Problem<real,9> {
public:
    typedef Eigen::Matrix<real,9,1> Vec9;
            
    bool converged(const Vec9 &x0, const Vec9 &x1, const Vec9 &grad) const override
    {
        return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
    }
};

class NHProjectionProblem3X3 : public ProjectionProblem3X3 {
public:
    void set_lame(real mu, real lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    real energy_density(const Vec9 &x) const
    {
        Mat3R FT = x.reshaped(3,3);

        real I_1 = FT.squaredNorm();
        real J = FT.determinant();

        real log_I3 = std::log( J * J );
        real t1 = 0.5 * _mu * ( I_1 - log_I3 - 3.0 );
        real t2 = 0.125 * _lambda * log_I3 * log_I3;
        real r = t1 + t2;
        return r;
    }

    real value(const Vec9 &x, const Vec9 &x0) const override
    {
        Mat3R FT = x.reshaped(3,3);

        real J = FT.determinant();
        if(J < 0.0)
        {
            return std::numeric_limits<real>::max();
        }

        real t1 = energy_density(x); // U(Dx)
        real t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    real gradient(const Vec9 &x, const Vec9 &x0, Vec9 &grad) const override
    {
        Mat3R FT = x.reshaped(3,3);
        Mat3R Finv = FT.inverse().transpose();
        real J = FT.determinant();

        if(J<0.0) J = -J;

        Mat3R P = _mu * (FT - Finv) + 0.5 * _lambda * std::log(J * J) * Finv;

        grad = P.reshaped(9,1);

        return value(x, x0);
    }

protected:
    real _mu, _lambda, _k;
};

class ProjectionProblem3X2 : public mcl::optlib::Problem<real,6> {
public:
    typedef Eigen::Matrix<real,6,1> Vec6;
    typedef Mat3x2R Mat3X2;

    bool converged(const Vec6 &x0, const Vec6 &x1, const Vec6 &grad) const override
    {
        return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
    }
};

class NHProjectionProblem3X2 : public ProjectionProblem3X2 {
public:
    void set_lame(real mu, real lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    real energy_density(const Vec6 &x) const
    {
        Mat3X2 F = x.reshaped(3,2);
        Mat2R FTF = F.transpose() * F;

        real I_1 = FTF.trace();
        real I_3 = FTF.determinant();

        real log_I3 = std::log( I_3 );
        real t1 = 0.5 * _mu * ( I_1 - log_I3 - 3.0 );
        real t2 = 0.125 * _lambda * log_I3 * log_I3;
        real r = t1 + t2;
        return r;
    }

    real value(const Vec6 &x, const Vec6 &x0) const override
    {
        real t1 = energy_density(x); // U(Dx)
        real t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    real gradient(const Vec6 &x, const Vec6 &x0, Vec6 &grad) const override
    {
        Mat3X2 F = x.reshaped(3,2);
        Mat2R FTF = F.transpose() * F;
        Mat3X2 FinvT = (FTF.inverse() * F.transpose()).transpose();
        real I_3 = FTF.determinant();

        Mat3X2 P = _mu * (F - FinvT) + 0.5 * _lambda * std::log(I_3) * FinvT;

        grad = P.reshaped(6,1);

        return value(x, x0);
    }

protected:
    real _mu, _lambda, _k;
};


}
