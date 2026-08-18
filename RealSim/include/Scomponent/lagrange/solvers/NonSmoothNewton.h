#pragma once
#include "Scomponent/lagrange/solvers/BaseConstraintSolver.h"
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"

#include "Scomponent/tools/math/SparseMatrix.h"
#include "Scomponent/tools/math/MatrixOperations.h"

namespace RealSim::lagrange::constraintsolver
{

// Implementation of Non-smooth Newton method for contact (LCP) and frictional contact (NCP) constraint resolution
//
// Original Author: Ziqiu ZENG
//
class NonSmoothNewtonSolver : public BaseConstraintSolver
{
public:
    typedef BaseConstraintSolver InheritSolver;

    void init() override;

    void addConstraintLinearSolver(linearsolver::dense::TypeDenseLinearSolver type) override;

    void prepare(const std::vector<RealSim::contact::ContactPair> & contactPairSet) override;

    void build(const MatX3R &pos, const MatX3R &b) override;

    void solve() override;

    void applyConstraintCorrection(MatX3R &x, MatX3R &b) override;

    void post() override;

protected:
    std::shared_ptr<linearsolver::BaseDenseLinearSolver> _constraintlinearsolver;

    MatXR _nonsmoothDelasus;

    VecXR _x;

    VecXR _dlam;
    VecXR _rhs;

private:
    VecXR _nonsmoothCompliance;
    VecXR _omega;
    VecXR _h;

    VecXR _precond;
    VecXR _precondLambda;

    VecXR _tmp_dof;
    VecXR _tmp_c1, _tmp_c2;

    void computePreconditioner(VecXR &precond);

    void computeNonsmoothFunction();

    void nonsmoothBilateralFunction(int cid, int size, real invStiff);
    void nonsmoothUnilateralFunction(int cid, int size);
    void nonsmoothFrictionalFunction(int cid, int size, real mu);

    void nonsmoothMinimumMapUnilateralFunction(int cid, int size);
    void nonsmoothMinimumMapFrictionalFunction(int cid, int size, real mu);
    void nonsmoothFischerBurmeisterUnilateralFunction(int cid, int size);
    void nonsmoothFischerBurmeisterFrictionalFunction(int cid, int size, real mu);

    void computeNonsmoothDelasus(MatXR & res, const MatXR & W, const VecXR & omega, const VecXR & compliance);

    void boundConstraintForces();

};

}