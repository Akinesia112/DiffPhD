#pragma once
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"
#include "Scomponent/linearsolver/iterative/BaseIterativeSolver.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::iterative
{

class DenseCGSolver: public BaseDenseLinearSolver, public BaseIterativeSolver {

public:
    DenseCGSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = false)
        : BaseIterativeSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: DenseCGSolver"<<RESET<<std::endl;
    }

    void init(MatXR * matrix) override
    {
        // do nothing
    }

    void set(MatXR * matrix) override
    {
        _matrix  = matrix;
        computePrecond(*matrix);
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        if(_matrix == nullptr) return;
        const MatXR & A = *_matrix;

        real rho, rho_old, alpha, beta;
        unsigned int max_iter = std::min<unsigned int>(_maxIter, b.size());
        VecXR r, d, q, s;
        r.resize(b.size());
        d.resize(b.size());
        q.resize(b.size());
        s.resize(b.size());

        _nb_iter = 1;

        const real dot_b = b.dot(b);
        if(dot_b != 0.0)
        {
            if(_warmStart)
            {
                RealSim::tools::linearalgebra::parallelDnMV(q, A, x);
                RealSim::tools::linearalgebra::parallelPeq(r, 1.0, b, -1.0, q);
            }
            else { r = b; x.setZero(); }

            if(_precond)
            {
                applyPrecond(d, r);
            }
            else
            {
                d = r;
            }

            rho = r.dot(d);

            real tol = _tol * _tol * rho;

            while((_nb_iter <= max_iter) && (rho > tol))
            {
                RealSim::tools::linearalgebra::parallelDnMV(q, A, d);

                const real den = d.dot(q);

                if(den == 0.0) break;
                alpha = rho/den;

                RealSim::tools::linearalgebra::parallelPeq(x, 1.0, x, alpha, d);
                RealSim::tools::linearalgebra::parallelPeq(r, 1.0, r, -alpha, q);

                rho_old = rho;

                if(_precond)
                {
                    applyPrecond(s, r);

                    rho = r.dot(s);
                    beta = rho/rho_old;

                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, s, beta, d);
                }
                else
                {
                    rho = r.dot(r);
                    beta = rho/rho_old;

                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, r, beta, d);
                }

                _nb_iter++;
            }

//            std::cout<<"     Conjugate Gradient iterations = "<<_nb_iter-1<<std::endl;
        }
        else
        {
//            std::cout<<"    Conjugate Residual iterations = "<<_nb_iter-1<<std::endl;
            x.setZero();
        }
    }

    int getIterations() override {return _nb_iter-1;}

protected:
    virtual void computePrecond(const MatXR &matrix)
    {
        _precond = false;
    }

    virtual void applyPrecond(VecXR &x, const VecXR &y)
    {
        x = y;
    }
};

}