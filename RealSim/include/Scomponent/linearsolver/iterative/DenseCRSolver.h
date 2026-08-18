#pragma once
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"
#include "Scomponent/linearsolver/iterative/BaseIterativeSolver.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::iterative
{

class DenseCRSolver: public BaseDenseLinearSolver, public BaseIterativeSolver  {

public:
    DenseCRSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = false)
            : BaseIterativeSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: DenseCRSolver"<<RESET<<std::endl;
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
        unsigned int nb_iter = 1;
        VecXR r, d, q, h, s;
        r.resize(b.size());
        d.resize(b.size());
        q.resize(b.size());
        h.resize(b.size());
        s.resize(b.size());

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

            RealSim::tools::linearalgebra::parallelDnMV(q, A, d);
            h = q;

            rho = r.dot(h);

            real tol = _tol * _tol * rho;

            while((nb_iter <= max_iter) && (rho > tol))
            {
                const real den = q.dot(q);

                if(den == 0.0) break;
                alpha = rho/den;

                RealSim::tools::linearalgebra::parallelPeq(x, 1.0, x, alpha, d);
                RealSim::tools::linearalgebra::parallelPeq(r, 1.0, r, -alpha, q);

                if(_precond)
                {
                    applyPrecond(s, r);

                    RealSim::tools::linearalgebra::parallelDnMV(h, A, s);

                    rho_old = rho;
                    rho = r.dot(h);
                    beta = rho / rho_old;

                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, s, beta, d);
                    RealSim::tools::linearalgebra::parallelPeq(q, 1.0, h, beta, q);
                }
                else
                {
                    RealSim::tools::linearalgebra::parallelDnMV(h, A, r);
                    rho_old = rho;
                    rho = r.dot(h);
                    beta = rho / rho_old;

                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, r, beta, d);
                    RealSim::tools::linearalgebra::parallelPeq(q, 1.0, h, beta, q);
                }

                nb_iter++;
            }

//            std::cout<<"    Conjugate Residual iterations = "<<nb_iter-1<<std::endl;
        }
        else
        {
//            std::cout<<"    Conjugate Residual iterations = "<<nb_iter-1<<std::endl;
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