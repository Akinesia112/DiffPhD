#ifndef STATE_FORCE_MESH_CONTACT_STATE_FORCE_H
#define STATE_FORCE_MESH_CONTACT_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

#include <array>
#include <vector>

// Mesh-based contact for two deformable bodies. Faces are given as triangle indices
// in global vertex numbering. Forces are applied symmetrically to vertex and face vertices.
template<int vertex_dim>
class MeshContactStateForce : public StateForce<vertex_dim> {
public:
    void Initialize(const real contact_radius, const real kn_left, const real kn_right, const real kf, const real mu,
        const std::vector<int>& faces_a, const std::vector<int>& faces_b);

    const real contact_radius() const { return contact_radius_; }
    const real kn_left() const { return StateForce<vertex_dim>::parameters()(0); }
    const real kn_right() const { return StateForce<vertex_dim>::parameters()(1); }
    const real kf() const { return StateForce<vertex_dim>::parameters()(2); }
    const real mu() const { return StateForce<vertex_dim>::parameters()(3); }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    real contact_radius_ = 0;
    std::vector<std::array<int, 3>> faces_a_;
    std::vector<std::array<int, 3>> faces_b_;
    std::vector<int> vertices_a_;
    std::vector<int> vertices_b_;
};

#endif
