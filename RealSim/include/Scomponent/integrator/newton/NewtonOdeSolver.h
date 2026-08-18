#pragma once

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

namespace sofa::component::odesolver::backward
{

template<class TDataTypes>
class NewtonIterativeOdeSolver : public sofa::core::behavior::OdeSolver, public core::behavior::SingleStateAccessor<TDataTypes>
{
public:

    SOFA_CLASS2(SOFA_TEMPLATE(NewtonIterativeOdeSolver, TDataTypes), sofa::core::behavior::OdeSolver, SOFA_TEMPLATE(core::behavior::SingleStateAccessor, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real             Real;
    typedef typename DataTypes::Coord            Coord;
    typedef typename DataTypes::Deriv            Deriv;
    typedef typename DataTypes::VecCoord         VecCoord;
    typedef typename DataTypes::VecDeriv         VecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;

    Data<unsigned int> d_maxIter; ///< maximum number of iterations of the Newton solution
    Data<SReal> d_tolerance; ///< desired precision of the Newton solution

    Data<SReal> f_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> f_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0
    Data<SReal> f_velocityDamping; ///< Velocity decay coefficient (no decay if null)

    SingleLink<NewtonIterativeOdeSolver, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_GlobalLinearSolver;

public:
    NewtonIterativeOdeSolver();

    void init() override;

    // process Local-Global iterations
    void solve (const core::ExecParams* params, SReal dt, core::MultiVecCoordId xResult, core::MultiVecDerivId vResult) override;

protected:
    core::behavior::MultiVecDeriv x;

private:
    void convertType(core::MultiVecCoordId out, core::MultiVecDerivId in);
    void convertType(core::MultiVecDerivId out, core::MultiVecCoordId in);
};

} // namespace sofa::component::odesolver::backward
