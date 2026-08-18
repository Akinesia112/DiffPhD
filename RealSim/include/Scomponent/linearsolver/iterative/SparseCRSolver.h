#pragma once
#include "Scomponent/linearsolver/iterative/BaseIterativeSolver.h"
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::linearsolver::iterative
{

class SparseCRSolver: public BaseSparseLinearSolver, public BaseIterativeSolver {

public:
    SparseCRSolver(unsigned int maxIter = 100, real tol = 1e-5, bool warmStart = true)
        : BaseIterativeSolver(maxIter, tol, warmStart)
    {
        std::cout<<GREEN<<"Dense Linear Solver: SparseCRSolver"<<RESET<<std::endl;
    }

    void init(SpMatR * matrix) override
    {
        _matrix  = matrix;
        _csA = RealSim::tools::linearalgebra::CSMatrix(*matrix);
        computePrecond(*matrix);
    }

    void solve_vec(VecXR &x, const VecXR &b) override
    {
        solve_timer.start();

        real rho, rho_old, alpha, beta;
        unsigned int nb_iter;
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
                RealSim::tools::linearalgebra::parallelSpMV(q, _csA, x);
                r = b - q;
//                RealSim::tools::linearalgebra::parallelPeq(r, 1.0, b, -1.0, q);
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

            RealSim::tools::linearalgebra::parallelSpMV(q, _csA, d);
            h = q;

            rho = r.dot(h);
            real err = r.dot(r);

//            RealSim::tools::linearalgebra::parallelSpMV(q, _csA, b);
//            real tol = _tol * _tol * b.dot(s);
            real tol = _tol * _tol * dot_b;

            nb_iter = 1;
            while((nb_iter <= _maxIter) && (err > tol))
            {
                const real den = q.dot(q);
                if(den == 0.0) break;
                alpha = rho/den;


                x = x + alpha * d;
                r = r - alpha * q;
//                RealSim::tools::linearalgebra::parallelPeq(x, 1.0, x, alpha, d);
//                RealSim::tools::linearalgebra::parallelPeq(r, 1.0, r, -alpha, q);

                if(_precond)
                {
                    applyPrecond(s, r);
                    RealSim::tools::linearalgebra::parallelSpMV(h, _csA, s);
                    rho_old = rho;
                    rho = r.dot(h);
                    beta = rho / rho_old;
                    d = s - beta * d;
                    q = h + beta * q;
//                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, s, beta, d);
//                    RealSim::tools::linearalgebra::parallelPeq(q, 1.0, h, beta, q);
                }
                else
                {
                    RealSim::tools::linearalgebra::parallelSpMV(h, _csA, r);
                    rho_old = rho;
                    rho = r.dot(h);
                    beta = rho / rho_old;
                    d = r + beta * d;
                    q = h + beta * q;
//                    RealSim::tools::linearalgebra::parallelPeq(d, 1.0, r, beta, d);
//                    RealSim::tools::linearalgebra::parallelPeq(q, 1.0, h, beta, q);
                }

                err = r.dot(r);

                nb_iter++;
            }

//            std::cout<<"    Conjugate Residual iterations = "<<nb_iter<<std::endl;
        }
        else
        {
//            std::cerr<<YELLOW<<"Conjugate Residual: null norm of vector b"<<RESET<<std::endl;
            x.setZero();
        }

        solve_timer.stop();
    }

protected:
    RealSim::tools::linearalgebra::CSMatrix _csA;

    virtual void computePrecond(const SpMatR &matrix)
    {
        _precond = false;
    }

    virtual void applyPrecond(VecXR &x, const VecXR &y)
    {
        x = y;
    }

};

}