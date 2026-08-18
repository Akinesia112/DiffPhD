#pragma once
#include <MCL/Problem.hpp>
#include <MCL/LBFGS.hpp>

namespace RealSim::integrator::projectivedynamics::energy{

class ProjectionProblem3D : public mcl::optlib::Problem<double,3> {
public:
    bool converged(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const Eigen::Vector3d &grad) const override
    {
        return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
    }
};

class NHProjectionProblem3D : public ProjectionProblem3D {
public:
    void set_lame(double mu, double lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    double energy_density(const Eigen::Vector3d &x) const
    {
        double J = x[0]*x[1]*x[2];
        double I_1 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
        double I_3 = J*J;
        double log_I3 = std::log( I_3 );
        double t1 = 0.5 * _mu * ( I_1 - log_I3 - 3.0 );
        double t2 = 0.125 * _lambda * log_I3 * log_I3;
        double r = t1 + t2;
        return r;
    }

    double value(const Eigen::Vector3d &x, const Eigen::Vector3d &x0) const override
    {
        if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
            // No Mr. Linesearch, you have gone too far!
            return std::numeric_limits<float>::max();
        }
        double t1 = energy_density(x); // U(Dx)
        double t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    double gradient(const Eigen::Vector3d &x, const Eigen::Vector3d &x0, Eigen::Vector3d &grad) const override
    {
        double J = x[0]*x[1]*x[2];
        if( J <= 0.0 ){
            throw std::runtime_error("NHProjectionProblem::gradient Error: J <= 0");
        } else {
            Eigen::Vector3d x_inv(3);
            x_inv[0] = 1.0/x[0];
            x_inv[1] = 1.0/x[1];
            x_inv[2] = 1.0/x[2];
            grad = (_mu * (x - x_inv) + _lambda * std::log(J) * x_inv) + _k*(x-x0);
        }
        return value(x, x0);
    }

protected:
    double _mu, _lambda, _k;
};

class ProjectionProblem2D : public mcl::optlib::Problem<double,2> {
public:
    bool converged(const Eigen::Vector2d &x0, const Eigen::Vector2d &x1, const Eigen::Vector2d &grad) const override
    {
        return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
    }
};

class NHProjectionProblem2D : public ProjectionProblem2D {
public:
    void set_lame(double mu, double lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    double energy_density(const Eigen::Vector2d &x) const
    {
        double J = x[0]*x[1];
        double I_1 = x[0]*x[0]+x[1]*x[1];
        double I_3 = J*J;
        double log_I3 = std::log( I_3 );
        double t1 = 0.5 * _mu * ( I_1 - log_I3 - 3.0 );
        double t2 = 0.125 * _lambda * log_I3 * log_I3;
        double r = t1 + t2;
        return r;
    }

    double value(const Eigen::Vector2d &x, const Eigen::Vector2d &x0) const override
    {
        if( x[0]<0.0 || x[1]<0.0){
            // No Mr. Linesearch, you have gone too far!
            return std::numeric_limits<float>::max();
        }
        double t1 = energy_density(x); // U(Dx)
        double t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    double gradient(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, Eigen::Vector2d &grad) const override
    {
        double J = x[0]*x[1];
        if( J <= 0.0 ){
            throw std::runtime_error("NHProjectionProblem::gradient Error: J <= 0");
        } else {
            Eigen::Vector2d x_inv(2);
            x_inv[0] = 1.0/x[0];
            x_inv[1] = 1.0/x[1];
            grad = (_mu * (x - x_inv) + _lambda * std::log(J) * x_inv) + _k*(x-x0);
        }
        return value(x, x0);
    }

protected:
    double _mu, _lambda, _k;
};


class CorotProjectionProblem3D : public ProjectionProblem3D {
public:
    void set_lame(double mu, double lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    //E(Sig) = mu * ||Sig - I||F^2 + (lam / 2) * tr(Sig - I)^2
    double energy_density(const Eigen::Vector3d &x) const
    {
        Eigen::Vector3d I(3);
        I[0] = 1.0;
        I[1] = 1.0;
        I[2] = 1.0;

        double t1 = _mu * (x - I).squaredNorm();
        double trace = (x - I).trace();
        double t2 = 0.5 * _lambda * trace * trace;
        double r = t1 + t2;
        return r;
    }

    double value(const Eigen::Vector3d &x, const Eigen::Vector3d &x0) const override
    {
        if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
            // No Mr. Linesearch, you have gone too far!
            return std::numeric_limits<float>::max();
        }
        double t1 = energy_density(x); // U(Dx)
        double t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    // P(Sig) = 2 * mu * (Sig - I) + lam * tr(Sig - I) * I
    double gradient(const Eigen::Vector3d &x, const Eigen::Vector3d &x0, Eigen::Vector3d &grad) const override
    {
        Eigen::Vector3d I(3);
        I[0] = 1.0;
        I[1] = 1.0;
        I[2] = 1.0;
        grad = (2 * _mu * (x - I) + _lambda * (x - I).trace() * I) + _k*(x - x0);

        return value(x, x0);
    }

protected:
    double _mu, _lambda, _k;
};

class CorotProjectionProblem2D : public ProjectionProblem2D {
public:
    void set_lame(double mu, double lam){
        _mu = mu;
        _lambda = lam;
        _k = 2.0*_mu; //bulk modulus
    }

    //E(Sig) = mu * ||Sig - I||F^2 + (lam / 2) * tr(Sig - I)^2
    double energy_density(const Eigen::Vector2d &x) const
    {
        Eigen::Vector2d I(2);
        I[0] = 1.0;
        I[1] = 1.0;

        double t1 = _mu * (x - I).squaredNorm();
        double trace = (x - I).trace();
        double t2 = 0.5 * _lambda * trace * trace;
        double r = t1 + t2;
        return r;
    }

    double value(const Eigen::Vector2d &x, const Eigen::Vector2d &x0) const override
    {
        if( x[0]<0.0 || x[1]<0.0){
            // No Mr. Linesearch, you have gone too far!
            return std::numeric_limits<float>::max();
        }
        double t1 = energy_density(x); // U(Dx)
        double t2 = (_k*0.5) * (x-x0).squaredNorm(); // quad penalty
        return t1 + t2;
    }

    // P(Sig) = 2 * mu * (Sig - I) + lam * tr(Sig - I) * I
    double gradient(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, Eigen::Vector2d &grad) const override
    {
        Eigen::Vector2d I(2);
        I[0] = 1.0;
        I[1] = 1.0;
        grad = (2 * _mu * (x - I) + _lambda * (x -I).trace() * I) + _k*(x - x0);

        return value(x, x0);
    }

protected:
    double _mu, _lambda, _k;
};


}
