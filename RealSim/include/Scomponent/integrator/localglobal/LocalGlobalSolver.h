#pragma once
#include <memory>
#include "Scomponent/integrator/SBaseIntegrator.h"
#include "Scomponent/integrator/localglobal/energy/Mass.h"
#include "Scomponent/integrator/localglobal/energy/BaseEnergy.h"
#include "Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h"
#include "Sgeometry/SBaseGeometry.h"
#include "Scomponent/linearsolver/BaseSparseLinearSolver.h"
#include "Scomponent/linearsolver/BaseDenseLinearSolver.h"

#include "Scomponent/lagrange/solvers/BaseConstraintSolver.h"
#include "Scomponent/collisiondetection/CollisionHandler.h"
#include "Scomponent/lagrange/ControlHandler.h"
#include "Scomponent/tools/action/Actions.h"


namespace RealSim::integrator::projectivedynamics
{


// Projective Dynamics implementation based on Bouaziz et al. 2014
//
// Original Author: Ziqiu ZENG
//
class LocalGlobalSolver: public SBaseIntegrator
{

public:
    LocalGlobalSolver(unsigned int maxIter)
        :_maxIter(maxIter) {
        _printEnergy = false;
        _activateRefSolver = false;
    }

    ~LocalGlobalSolver() {};

    void setIterations(int ite) { _maxIter = ite; }
    void setPrintEnergyAndError(bool print) { _printEnergy = print; }
    void setFrame(unsigned int frame) { _frame = frame; }
    void setTimerPrint(unsigned int timer) { _timer = timer; }

    void init(const MatX3R & initpos) override;
    void update(MatXR &pos) override;

    void setVertexDimension(unsigned int dim) {_state._dim = dim;} //number of vertex
    unsigned int getVertexDimension() const {return _state._dim;} //number of vertex

    void setDt(real dt) { _dt = dt; }

    void setMass();
    void addObjectMass(real totalmass, const VecXI & vertices);
    void addVertexMass(real vertexmass, const VecXI & vertices);
    void setGravity(Vec3R g) { _grav = g; }

    void addLinearSolver(linearsolver::sparse::TypeSparseLinearSolver type, bool ref = false);
    void setIterativeLinearSolver(unsigned int maxIter, real tol, bool warmStart);
    void addElasticEnergy(energy::ElasticModel type, geometry::SBaseGeometry *geometry, real youngModulus, real poissonRatio, real bendingParam, energy::ElasticModel bendtype);
    void addPinEnergy(real weight, const VecXI& vertices, const MatX3R& refpos);
    void addDynamicPinEnergy(real weight, const VecXI& vertices, const MatX3R& refpos, std::vector<std::shared_ptr<RealSim::tools::action::Action>> &actions);
    void addAxisPinEnergy(real weight, const MatX3R& refpos, const Vec3R& base, const Vec3R& axis, real radius, bool slide);
    void addADMMPinEnergy(real weight, const VecXI& vertices, const MatX3R& refpos);
    void addPlaneContactEnergy(real weight, const std::vector<int>& vertices, const Vec3R& origin, const Vec3R& normal, bool admm);

    void computeExternalForce(MatX3R &f);

    void setCollisionHandler(collisiondetection::CollisionHandler * chandler);
    void setControlHandler(lagrange::ControlHandler * chandler);
    void addConstraintSolver(lagrange::constraintsolver::TypeConstraintSolver type);
    void setConstraintSolver(linearsolver::dense::TypeDenseLinearSolver type, unsigned int maxIter, real tol, real maxforce, RealSim::lagrange::constraintsolver::TypeNonsmoothFunction function);

    real getTotalEnergy(const MatX3R &pos, const MatX3R &sn, real dt);

protected:
    // internal states
    MechanicalState _state;

    unsigned int _frame;
    unsigned int _timer;

    MatX3R _nextpos; //temporary vector to store positions in iterations
    MatX3R _b; //RHS

    MatX3R _sn;

    SpMatR _matrix;

    // parameters
    Vec3R _grav;
    real _dt;
    unsigned _maxIter;

    bool _printEnergy;
    bool _activateRefSolver;

    // components
    std::shared_ptr<energy::Mass> _mass;
    std::vector<std::shared_ptr<energy::BaseEnergy>> _energies;
    std::shared_ptr<linearsolver::BaseSparseLinearSolver> _linearsolver;

    std::shared_ptr<linearsolver::BaseSparseLinearSolver> _linearsolver_ref;

    // constraints
    std::vector<RealSim::contact::ContactPair> _contactpairset;
    std::shared_ptr<collisiondetection::CollisionHandler> _collisionhandler;
    std::shared_ptr<lagrange::ControlHandler> _controlhandler;
    std::shared_ptr<lagrange::constraintsolver::BaseConstraintSolver> _constraintsolver;

    // timers
    c_timer timer, local_timer, global_timer, cd_timer;
    int count_timer, count_timer_cst;
    std::vector<real> record_total, record_local, record_global, record_collision;
    std::vector<real> record_solve, record_schur;
    std::vector<int> record_cst_num, record_cstsolve_ite;
    std::vector<real> record_prepare, record_build, record_cstsolve, record_correction;


    void prepare(const MatX3R &pos0, const MatX3R &pos1, real dt);
    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff);
    void globalSolve(MatX3R &x, MatX3R &b);

    void resetTimer();
    void printTimer();

    void runRefSolver(std::vector<real> &energy_compare);

    void printEnergyAndError(const std::string &filename, const std::vector<real> &energy, real init_energy, real converged_energy) const;

};

}