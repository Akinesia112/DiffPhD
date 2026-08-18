#pragma once
#include <memory>
#include "Scomponent/lagrange/BaseController.h"
#include "Scomponent/lagrange/CableController.h"

namespace RealSim::lagrange
{

class ControlHandler
{
public:
    void clearControllerSet()
    {
        _controllerset.clear();
    }

    void init()
    {
        for(const auto & controller: _controllerset)
        {
            controller->init();
        }
    }

    void addController(BaseController * collision)
    {
        _controllerset.push_back(static_cast<const std::shared_ptr<BaseController>>(collision));
    }

    void addCableController(const MatX3R& pos, real invstiff)
    {
        auto c = std::make_shared<RealSim::lagrange::CableController>(pos, invstiff);
        _controllerset.push_back(c);
    }

    void setControl(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0)
    {
        for(const auto & controller: _controllerset)
        {
            controller->set(output, pos0);
        }
    }

    const std::vector<std::shared_ptr<BaseController>> & controllerSet()
    {
        return _controllerset;
    }

protected:
    std::vector<std::shared_ptr<BaseController>> _controllerset;
};

}
