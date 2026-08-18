#pragma once
#include <utility>

#include "Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.h"
#include "Scomponent/tools/action/Actions.h"


namespace RealSim::integrator::projectivedynamics::energy
{

// Implementation of dynamic pin energy in Projective Dynamics
//
// Original Author: Ziqiu ZENG
//
class DynamicPinEnergy : public PinEnergy
{

public:
    typedef PinEnergy InheritEnergy;

    DynamicPinEnergy(real weight, VecXI vertices, MatX3R refpos)
        :PinEnergy(weight, std::move(vertices), std::move(refpos))
        {}

    void init() override
    {
        InheritEnergy::init();
        _nextpos = _refpos;
    }

    void addAction(const std::shared_ptr<tools::action::Action> & action)
    {
        _actions.push_back(action);
    }

    void setActions(const std::vector<std::shared_ptr<tools::action::Action>> & actions)
    {
        _actions = actions;
    }

    void reset() override;

    void localProjection(MatX3R &rhs, const MatX3R &pos, real coeff) override;

    real getCurrentEnergy(const MatX3R &pos) override;

protected:
    std::vector<std::shared_ptr<tools::action::Action>> _actions;
    MatX3R _nextpos;
};

}
