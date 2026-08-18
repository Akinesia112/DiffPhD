#include "state_force/mesh_contact_state_force.h"

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_set>

namespace {
struct Aabb {
    Vector3r min;
    Vector3r max;
};

struct ClosestPointResult {
    Vector3r closest;
    Vector3r bary;
    real dist2;
};

Aabb ComputeAabb(const std::vector<int>& vertices, const VectorXr& q) {
    Aabb box;
    const real inf = std::numeric_limits<real>::infinity();
    box.min = Vector3r(inf, inf, inf);
    box.max = Vector3r(-inf, -inf, -inf);
    for (const int idx : vertices) {
        const Vector3r p = q.segment(idx * 3, 3);
        box.min = box.min.cwiseMin(p);
        box.max = box.max.cwiseMax(p);
    }
    return box;
}

real AabbDistance2(const Aabb& a, const Aabb& b) {
    real d2 = 0;
    for (int k = 0; k < 3; ++k) {
        real d = 0;
        if (a.max(k) < b.min(k)) d = b.min(k) - a.max(k);
        else if (b.max(k) < a.min(k)) d = a.min(k) - b.max(k);
        d2 += d * d;
    }
    return d2;
}

ClosestPointResult ClosestPointTriangle(const Vector3r& p, const Vector3r& a, const Vector3r& b, const Vector3r& c) {
    const Vector3r ab = b - a;
    const Vector3r ac = c - a;
    const Vector3r ap = p - a;

    const real d1 = ab.dot(ap);
    const real d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) {
        return {a, Vector3r(1, 0, 0), (p - a).squaredNorm()};
    }

    const Vector3r bp = p - b;
    const real d3 = ab.dot(bp);
    const real d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) {
        return {b, Vector3r(0, 1, 0), (p - b).squaredNorm()};
    }

    const real vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const real v = d1 / (d1 - d3);
        const Vector3r point = a + v * ab;
        return {point, Vector3r(1 - v, v, 0), (p - point).squaredNorm()};
    }

    const Vector3r cp = p - c;
    const real d5 = ab.dot(cp);
    const real d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) {
        return {c, Vector3r(0, 0, 1), (p - c).squaredNorm()};
    }

    const real vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const real w = d2 / (d2 - d6);
        const Vector3r point = a + w * ac;
        return {point, Vector3r(1 - w, 0, w), (p - point).squaredNorm()};
    }

    const real va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vector3r point = b + w * (c - b);
        return {point, Vector3r(0, 1 - w, w), (p - point).squaredNorm()};
    }

    const real denom = 1 / (va + vb + vc);
    const real v = vb * denom;
    const real w = vc * denom;
    const Vector3r point = a + ab * v + ac * w;
    return {point, Vector3r(1 - v - w, v, w), (p - point).squaredNorm()};
}
}

template<int vertex_dim>
void MeshContactStateForce<vertex_dim>::Initialize(const real contact_radius, const real kn_left, const real kn_right,
    const real kf, const real mu,
    const std::vector<int>& faces_a, const std::vector<int>& faces_b) {
    contact_radius_ = contact_radius;
    VectorXr parameters(4);
    parameters(0) = kn_left;
    parameters(1) = kn_right;
    parameters(2) = kf;
    parameters(3) = mu;
    StateForce<vertex_dim>::set_parameters(parameters);

    CheckError(faces_a.size() % 3 == 0, "Invalid faces_a size.");
    CheckError(faces_b.size() % 3 == 0, "Invalid faces_b size.");

    const int face_num_a = static_cast<int>(faces_a.size() / 3);
    const int face_num_b = static_cast<int>(faces_b.size() / 3);
    faces_a_.resize(face_num_a);
    faces_b_.resize(face_num_b);

    for (int i = 0; i < face_num_a; ++i) {
        faces_a_[i] = {faces_a[3 * i], faces_a[3 * i + 1], faces_a[3 * i + 2]};
    }
    for (int i = 0; i < face_num_b; ++i) {
        faces_b_[i] = {faces_b[3 * i], faces_b[3 * i + 1], faces_b[3 * i + 2]};
    }

    std::unordered_set<int> vertex_set_a;
    std::unordered_set<int> vertex_set_b;
    for (const auto& face : faces_a_) {
        vertex_set_a.insert(face[0]);
        vertex_set_a.insert(face[1]);
        vertex_set_a.insert(face[2]);
    }
    for (const auto& face : faces_b_) {
        vertex_set_b.insert(face[0]);
        vertex_set_b.insert(face[1]);
        vertex_set_b.insert(face[2]);
    }

    vertices_a_.assign(vertex_set_a.begin(), vertex_set_a.end());
    vertices_b_.assign(vertex_set_b.begin(), vertex_set_b.end());
    std::sort(vertices_a_.begin(), vertices_a_.end());
    std::sort(vertices_b_.begin(), vertices_b_.end());
}

template<int vertex_dim>
const VectorXr MeshContactStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    CheckError(vertex_dim == 3, "MeshContactStateForce only supports 3D meshes.");
    CheckError(q.size() == v.size(), "Inconsistent q and v sizes.");
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");

    VectorXr f = VectorXr::Zero(dofs);
    if (faces_a_.empty() || faces_b_.empty()) return f;

    const real radius = contact_radius_;
    const real radius2 = radius * radius;
    const real eps = std::numeric_limits<real>::epsilon();

    const Aabb aabb_a = ComputeAabb(vertices_a_, q);
    const Aabb aabb_b = ComputeAabb(vertices_b_, q);
    if (AabbDistance2(aabb_a, aabb_b) > radius2) return f;

    auto accumulate_contacts = [&](const std::vector<int>& vertex_ids,
        const std::vector<std::array<int, 3>>& faces, const real kn_local) {
        for (const int vid : vertex_ids) {
            const Vector3r p = q.segment(vid * 3, 3);
            const Vector3r vp = v.segment(vid * 3, 3);
            for (const auto& face : faces) {
                const Vector3r a = q.segment(face[0] * 3, 3);
                const Vector3r b = q.segment(face[1] * 3, 3);
                const Vector3r c = q.segment(face[2] * 3, 3);
                const ClosestPointResult cp = ClosestPointTriangle(p, a, b, c);
                if (cp.dist2 >= radius2) continue;

                const real dist = std::sqrt(cp.dist2);
                Vector3r n;
                if (dist > eps) {
                    n = (p - cp.closest) / dist;
                } else {
                    n = (b - a).cross(c - a);
                    const real n_norm = n.norm();
                    if (n_norm <= eps) continue;
                    n /= n_norm;
                }
                if (n.dot(p - cp.closest) < 0) n = -n;

                const real penetration = radius - dist;
                const real fn_mag = kn_local * penetration;
                const Vector3r fn = fn_mag * n;

                const Vector3r va = v.segment(face[0] * 3, 3);
                const Vector3r vb = v.segment(face[1] * 3, 3);
                const Vector3r vc = v.segment(face[2] * 3, 3);
                const Vector3r vtri = cp.bary(0) * va + cp.bary(1) * vb + cp.bary(2) * vc;
                const Vector3r vrel = vp - vtri;
                const Vector3r vrel_t = vrel - n * vrel.dot(n);
                const real vrel_t_mag = vrel_t.norm();
                Vector3r ff = Vector3r::Zero();
                if (vrel_t_mag > eps) {
                    const real ff_mag = std::min(kf(), mu() * fn_mag / (vrel_t_mag + eps));
                    ff = -ff_mag * vrel_t;
                }

                const Vector3r total = fn + ff;
                f.segment(vid * 3, 3) += total;
                f.segment(face[0] * 3, 3) -= total * cp.bary(0);
                f.segment(face[1] * 3, 3) -= total * cp.bary(1);
                f.segment(face[2] * 3, 3) -= total * cp.bary(2);
            }
        }
    };

    accumulate_contacts(vertices_a_, faces_b_, kn_left());
    accumulate_contacts(vertices_b_, faces_a_, kn_right());

    return f;
}

template<int vertex_dim>
void MeshContactStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(StateForce<vertex_dim>::NumOfParameters());
}

template class MeshContactStateForce<2>;
template class MeshContactStateForce<3>;
