#ifndef FRICTION_MESH_FRICTIONAL_BOUNDARY_H
#define FRICTION_MESH_FRICTIONAL_BOUNDARY_H

#include "friction/frictional_boundary.h"
#include "common/common.h"

#include <array>
#include <vector>

// Dynamic mesh boundary for deformable-vs-deformable contact in AlgPhd.
// The boundary is represented by two surface triangle sets (left/right).
// For a query vertex, the distance/normal is computed against the opposite side.
template<int dim>
class MeshFrictionalBoundary : public FrictionalBoundary<dim> {
public:
    MeshFrictionalBoundary();

    void Initialize(const real contact_radius, const int split_index,
        const std::vector<int>& faces_left, const std::vector<int>& faces_right);

    void UpdateVertices(const VectorXr& q);

    real ContactRadius() const { return contact_radius_; }

    // Returns closest point on the opposite surface and its outward face normal.
    // Used by AlgPhd to anchor the contact constraint at frame start (avoids
    // the closest-face flip that occurs when a node penetrates deeply and a
    // back-side face becomes geometrically nearer than the entry face).
    void GetClosestPointAndNormal(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx,
        Eigen::Matrix<real, dim, 1>& closest,
        Eigen::Matrix<real, dim, 1>& normal) const;

    // Signed distance restricted to faces whose outward normal aligns with
    // ref_normal (n·ref_normal > 0). Avoids closest-face flip on deep
    // penetration: a back-side face (with opposing-direction normal) is
    // excluded, so the contact normal stays consistent with frame start.
    // Returns sign((p−closest)·ref_normal) · |p−closest| − contact_radius.
    real GetSignedDistanceWithReferenceNormal(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx,
        const Eigen::Matrix<real, dim, 1>& ref_normal) const;

    const Eigen::Matrix<real, dim, dim> GetLocalFrame(
        const Eigen::Matrix<real, dim, 1>& q) const override;

    const Eigen::Matrix<real, dim, dim> GetLocalFrame(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const override;

    const real GetDistance(
        const Eigen::Matrix<real, dim, 1>& q) const override;

    const real GetDistance(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const override;

    const Eigen::Matrix<real, dim, 1> GetDistanceGradient(
        const Eigen::Matrix<real, dim, 1>& q) const override;

    const Eigen::Matrix<real, dim, 1> GetDistanceGradient(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const override;

    const Eigen::Matrix<real, dim, 1> GetNormal(
        const Eigen::Matrix<real, dim, 1>& q) const override;

    const Eigen::Matrix<real, dim, 1> GetNormal(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const override;

    const std::vector<Eigen::Matrix<real, dim, 1>> GetNormalDifferential(
        const Eigen::Matrix<real, dim, 1>& q) const override;

    const std::vector<Eigen::Matrix<real, dim, 1>> GetNormalDifferential(
        const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const override;

    const bool ForwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit) const override;

    const bool ForwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit, const int node_idx) const override;

    void BackwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit,
        const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
        Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv) const override;

    void BackwardIntersect(
        const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit,
        const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
        Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv,
        const int node_idx) const override;

private:
    real contact_radius_ = 0;
    int split_index_ = 0;
    std::vector<std::array<int, 3>> faces_left_;
    std::vector<std::array<int, 3>> faces_right_;
    VectorXr q_;
    bool has_q_ = false;
};

#endif
