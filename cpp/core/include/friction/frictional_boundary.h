#ifndef FRICTION_FRICTIONAL_BOUNDARY_H
#define FRICTION_FRICTIONAL_BOUNDARY_H

#include "common/config.h"
#include <vector>

template<int dim>
class FrictionalBoundary {
public:
    virtual ~FrictionalBoundary() {}

    virtual const Eigen::Matrix<real, dim, dim> GetLocalFrame(
        const Eigen::Matrix<real, dim, 1>& q) const = 0;

    virtual const Eigen::Matrix<real, dim, dim> GetLocalFrame(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
        return GetLocalFrame(q);
    }

    virtual const real GetDistance(
        const Eigen::Matrix<real, dim, 1>& q) const = 0;

    virtual const real GetDistance(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
        return GetDistance(q);
    }

    virtual const Eigen::Matrix<real, dim, 1> GetDistanceGradient(
        const Eigen::Matrix<real, dim, 1>& q) const = 0;

    virtual const Eigen::Matrix<real, dim, 1> GetDistanceGradient(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
        return GetDistanceGradient(q);
    }

    virtual const Eigen::Matrix<real, dim, 1> GetNormal(
        const Eigen::Matrix<real, dim, 1>& q) const = 0;

    virtual const Eigen::Matrix<real, dim, 1> GetNormal(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
        return GetNormal(q);
    }

    virtual const std::vector<Eigen::Matrix<real, dim, 1>> GetNormalDifferential(
        const Eigen::Matrix<real, dim, 1>& q) const = 0;

    virtual const std::vector<Eigen::Matrix<real, dim, 1>> GetNormalDifferential(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
        return GetNormalDifferential(q);
    }

    virtual const bool ForwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit) const = 0;

    virtual const bool ForwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit, const int node_idx) const {
        return ForwardIntersect(q, v, dt, t_hit);
    }

    virtual void BackwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit,
        const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
        Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv) const = 0;

    virtual void BackwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit,
        const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
        Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv,
        const int node_idx) const {
        BackwardIntersect(q, v, t_hit, dl_dq_hit, dl_dq, dl_dv);
    }
};

#endif