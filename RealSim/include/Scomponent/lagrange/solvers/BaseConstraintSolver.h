#pragma once
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"
#include "Scomponent/integrator/SBaseIntegrator.h"
#include "Scomponent/lagrange/LagrangeConstraint.h"
#include "Scomponent/collisiondetection/BaseCollision.h"
#include "Scomponent/tools/math/SparseMatrix.h"

namespace RealSim::lagrange::constraintsolver
{

enum TypeConstraintSolver {
    ProjectedGS,
    NonSmoothNewton,
    NonSmoothNewton_CUDA,
    Bilateralize
};

enum TypeNonsmoothFunction {
    MinimumMap,
    FischerBurmeister
};

std::string type2String(TypeConstraintSolver t);
TypeConstraintSolver string2CSolverType(std::string s);

std::string type2String(TypeNonsmoothFunction t);
TypeNonsmoothFunction string2NNFuncType(std::string s);

// Base object implementation for constraint solvers in complementarity problems
//
// Original Author: Ziqiu ZENG
//
class BaseConstraintSolver
{
public:
    BaseConstraintSolver()
    {
        _maxIter = 100;
        _tol = 1e-3;
        _maxforce = std::numeric_limits<real>::max();
        _function = FischerBurmeister;
        _num_constraint = 0;
    }

    virtual void init()
    {
        std::cout<<"BaseConstraintSolver::init() not implemented"<<std::endl;
    }

    void setSystemLinearSolver(linearsolver::BaseSparseLinearSolver * lSolver)
    {
        _systemlinearsolver = static_cast<const std::shared_ptr<linearsolver::BaseSparseLinearSolver>>(lSolver);
    }

    virtual void addConstraintLinearSolver(linearsolver::dense::TypeDenseLinearSolver type)
    {

    }

    void setMechanicalState(integrator::MechanicalState * state)
    {
        _mstate = static_cast<const std::shared_ptr<integrator::MechanicalState>>(state);
    }

    unsigned int getVertexDimension() const {return _mstate->_dim;} //number of vertex

    virtual void prepare(const std::vector<RealSim::contact::ContactPair> & contactPairSet)
    {
        unsigned int n = getVertexDimension();

        _num_constraint = _constraint.getConstraintsInfo(contactPairSet);
        if(_num_constraint == 0) return;

        prepare_timer.start();

        // get contact info
        std::vector<Eigen::Triplet<real>> triplets;
        _constraint.buildJacobian(triplets, contactPairSet);

        _jacobian.resize(_num_constraint, n*3);
        _jacobian.setZero();
        _jacobian.setFromTriplets(triplets.begin(), triplets.end());

        _csJ = RealSim::tools::linearalgebra::CSMatrix(_jacobian);
        _csJT = _csJ.switchOrder();

        // get initial penetration
        _pene0.resize(_num_constraint);
        _delasus.resize(_num_constraint, _num_constraint);
        _penetration.resize(_num_constraint);
        _lambda.resize(_num_constraint);

        _lambda.setZero();

        _systemlinearsolver->addHAinvHT(_delasus, _jacobian);
        _constraint.initPenetration(_pene0, contactPairSet);

        prepare_timer.stop();
    }

    virtual void build(const MatX3R &pos, const MatX3R &b)
    {
        std::cout<<"BaseConstraintSolver::build() not implemented"<<std::endl;
    }

    virtual void solve() = 0;

    virtual void applyConstraintCorrection(MatX3R &x, MatX3R &b) = 0;

    void setParameters(unsigned int maxIter = 100, real tol = 1e-3, real maxforce = std::numeric_limits<real>::max(), TypeNonsmoothFunction function= FischerBurmeister)
    {
        _maxIter = maxIter;
        _tol = tol;
        _maxforce = maxforce;
        _function = function;
    }

    void resetTimer()
    {
        prepare_timer.clear();
        build_timer.clear();
        solve_timer.clear();
        correction_timer.clear();
    }

    void printTimer()
    {
//        std::cout<<GREEN<<"\nNumber of constraints = "<<BLUE<<_num_constraint<<std::endl;
//        std::cout<<GREEN<<"Constraint solver::prepare, called by "<<BLUE<<prepare_timer.times()<<GREEN<< " times, total time cost = "<<BLUE<<prepare_timer.total_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
//        std::cout<<GREEN<<"Constraint solver::build, called by "<<BLUE<<build_timer.times()<<GREEN<< " times, total time cost = "<<BLUE<<build_timer.total_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
//        std::cout<<GREEN<<"Constraint solver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< " times, total time cost = "<<BLUE<<solve_timer.total_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
//        std::cout<<GREEN<<"Constraint solver::correction, called by "<<BLUE<<correction_timer.times()<<GREEN<< " times, total time cost = "<<BLUE<<correction_timer.total_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
//
        std::cout<<GREEN<<"\nNumber of constraints = "<<BLUE<<_num_constraint<<std::endl;
        std::cout<<GREEN<<"Constraint solver::prepare, called by "<<BLUE<<prepare_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<prepare_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
        std::cout<<GREEN<<"Constraint solver::build, called by "<<BLUE<<build_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<build_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
        std::cout<<GREEN<<"Constraint solver::solve, called by "<<BLUE<<solve_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<solve_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
        std::cout<<GREEN<<"Constraint solver::correction, called by "<<BLUE<<correction_timer.times()<<GREEN<< " times, mean time cost = "<<BLUE<<correction_timer.mean_elapsed_ms()<<GREEN<<" ms"<<RESET<<std::endl;
    }

    virtual void post()
    {
        //do nothing
    }

    const c_timer * getPrepareTimer() {return &prepare_timer;}
    const c_timer * getBuildTimer() {return &build_timer;}
    const c_timer * getSolveTimer() {return &solve_timer;}
    const c_timer * getCorrectionTimer() {return &correction_timer;}
    int getConstraintNumber() {return _num_constraint;}
    int getSolveIterations() {return _num_iteration;}

    unsigned int _maxIter;
    real _tol;

    real _maxforce;

protected:
    std::shared_ptr<integrator::MechanicalState> _mstate;
    std::shared_ptr<linearsolver::BaseSparseLinearSolver> _systemlinearsolver;

    LagrangeConstraint _constraint;

    int _num_constraint;
    int _num_iteration;

    SpMatR _jacobian;

    // sparse format of static Jacobian
    RealSim::tools::linearalgebra::CSMatrix _csJ;
    RealSim::tools::linearalgebra::CSMatrix _csJT;

    MatXR _delasus;

    VecXR _pene0;

    VecXR _penetration;
    VecXR _lambda;

    TypeNonsmoothFunction _function;

    c_timer prepare_timer, build_timer, solve_timer, correction_timer;
};

}