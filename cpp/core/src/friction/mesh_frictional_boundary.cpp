#include "friction/mesh_frictional_boundary.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {
Vector3r OrthonormalAxis(const Vector3r& n) {
    Vector3r unit_x = Vector3r::Zero();
    for (int i = 0; i < 3; ++i) {
        const Vector3r x = n.cross(Vector3r::Unit(i));
        if (x.squaredNorm() > unit_x.squaredNorm()) unit_x = x;
    }
    const real norm = unit_x.norm();
    if (norm > std::numeric_limits<real>::epsilon()) unit_x /= norm;
    else unit_x = Vector3r::UnitX();
    return unit_x;
}

Vector3r ClosestPointTriangle(const Vector3r& p, const Vector3r& a, const Vector3r& b, const Vector3r& c) {
    const Vector3r ab = b - a;
    const Vector3r ac = c - a;
    const Vector3r ap = p - a;

    const real d1 = ab.dot(ap);
    const real d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return a;

    const Vector3r bp = p - b;
    const real d3 = ab.dot(bp);
    const real d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return b;

    const real vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const real v = d1 / (d1 - d3);
        return a + v * ab;
    }

    const Vector3r cp = p - c;
    const real d5 = ab.dot(cp);
    const real d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return c;

    const real vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const real w = d2 / (d2 - d6);
        return a + w * ac;
    }

    const real va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    const real denom = 1 / (va + vb + vc);
    const real v = vb * denom;
    const real w = vc * denom;
    return a + ab * v + ac * w;
}
}

template<int dim>
MeshFrictionalBoundary<dim>::MeshFrictionalBoundary() = default;

template<int dim>
void MeshFrictionalBoundary<dim>::Initialize(const real contact_radius, const int split_index,
    const std::vector<int>& faces_left, const std::vector<int>& faces_right) {
    CheckError(dim == 3, "MeshFrictionalBoundary only supports 3D.");
    contact_radius_ = contact_radius;
    split_index_ = split_index;

    CheckError(faces_left.size() % 3 == 0, "Invalid faces_left size.");
    CheckError(faces_right.size() % 3 == 0, "Invalid faces_right size.");

    const int face_num_left = static_cast<int>(faces_left.size() / 3);
    const int face_num_right = static_cast<int>(faces_right.size() / 3);
    faces_left_.resize(face_num_left);
    faces_right_.resize(face_num_right);

    for (int i = 0; i < face_num_left; ++i) {
        faces_left_[i] = {faces_left[3 * i], faces_left[3 * i + 1], faces_left[3 * i + 2]};
    }
    for (int i = 0; i < face_num_right; ++i) {
        faces_right_[i] = {faces_right[3 * i], faces_right[3 * i + 1], faces_right[3 * i + 2]};
    }
}

template<int dim>
void MeshFrictionalBoundary<dim>::UpdateVertices(const VectorXr& q) {
    q_ = q;
    has_q_ = true;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> MeshFrictionalBoundary<dim>::GetLocalFrame(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return GetLocalFrame(q, -1);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> MeshFrictionalBoundary<dim>::GetLocalFrame(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
    const Vector3r n = GetNormal(q, node_idx);
    Matrix3r local;
    local.col(2) = n;
    const Vector3r unit_x = OrthonormalAxis(n);
    local.col(0) = unit_x;
    Vector3r unit_y = n.cross(unit_x);
    const real norm = unit_y.norm();
    if (norm > std::numeric_limits<real>::epsilon()) unit_y /= norm;
    else unit_y = Vector3r::UnitY();
    local.col(1) = unit_y;
    return local;
}

template<int dim>
const real MeshFrictionalBoundary<dim>::GetDistance(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return GetDistance(q, -1);
}

template<int dim>
const real MeshFrictionalBoundary<dim>::GetDistance(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
    const Vector3r p = q;
    CheckError(has_q_, "MeshFrictionalBoundary requires UpdateVertices before query.");
    const std::vector<std::array<int, 3>>* faces = &faces_left_;
    if (node_idx >= 0 && node_idx < split_index_) {
        faces = &faces_right_;
    } else if (node_idx >= split_index_) {
        faces = &faces_left_;
    }

    if (faces->empty()) return std::numeric_limits<real>::infinity();

    const real eps = std::numeric_limits<real>::epsilon();
    real best_dist2 = std::numeric_limits<real>::infinity();
    Vector3r best_closest = Vector3r::Zero();
    Vector3r best_normal = Vector3r::UnitZ();
    for (const auto& face : *faces) {
        const Vector3r a = q_.segment(face[0] * 3, 3);
        const Vector3r b = q_.segment(face[1] * 3, 3);
        const Vector3r c = q_.segment(face[2] * 3, 3);
        const Vector3r closest = ClosestPointTriangle(p, a, b, c);
        const real dist2 = (p - closest).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_closest = closest;
            Vector3r n = (b - a).cross(c - a);
            const real n_norm = n.norm();
            if (n_norm > eps) n /= n_norm;
            else n = Vector3r::UnitZ();
            best_normal = n;
        }
    }
    // Unsigned distance to the opposing surface minus contact_radius.
    // Sign-based approaches using face normals are unreliable for non-convex
    // meshes (gatorman has many concavities). Signed distance for the NCP
    // solver is computed separately via GetSignedDistanceWithReferenceNormal
    // which uses the robust (p−closest)/|p−closest| direction.
    return std::sqrt(best_dist2) - contact_radius_;
}

template<int dim>
const Eigen::Matrix<real, dim, 1> MeshFrictionalBoundary<dim>::GetDistanceGradient(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return GetDistanceGradient(q, -1);
}

template<int dim>
const Eigen::Matrix<real, dim, 1> MeshFrictionalBoundary<dim>::GetDistanceGradient(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
    return GetNormal(q, node_idx);
}

template<int dim>
void MeshFrictionalBoundary<dim>::GetClosestPointAndNormal(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx,
    Eigen::Matrix<real, dim, 1>& closest_out,
    Eigen::Matrix<real, dim, 1>& normal_out) const {
    CheckError(has_q_, "MeshFrictionalBoundary requires UpdateVertices before query.");
    const Vector3r p = q;
    const std::vector<std::array<int, 3>>* faces = &faces_left_;
    if (node_idx >= 0 && node_idx < split_index_) faces = &faces_right_;
    else if (node_idx >= split_index_) faces = &faces_left_;

    const real eps = std::numeric_limits<real>::epsilon();
    real best_dist2 = std::numeric_limits<real>::infinity();
    Vector3r best_closest = Vector3r::Zero();
    Vector3r best_face_normal = Vector3r::UnitZ();
    for (const auto& face : *faces) {
        const Vector3r a = q_.segment(face[0] * 3, 3);
        const Vector3r b = q_.segment(face[1] * 3, 3);
        const Vector3r c = q_.segment(face[2] * 3, 3);
        const Vector3r closest = ClosestPointTriangle(p, a, b, c);
        const real dist2 = (p - closest).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_closest = closest;
            Vector3r n = (b - a).cross(c - a);
            const real n_norm = n.norm();
            if (n_norm > eps) n /= n_norm;
            else n = Vector3r::UnitZ();
            best_face_normal = n;
        }
    }
    // Contact normal: direction FROM the opposing surface TOWARD the query point.
    // This is the direction along which a repulsive contact force should push p.
    // Using (p − closest)/|p − closest| is robust for non-convex meshes (does
    // not depend on face winding) AND is naturally consistent with how the PD
    // iteration assembles J^T·λ — λ pushes p in the +ref_n direction.
    // Falls back to face normal when p sits exactly on the surface.
    closest_out = best_closest;
    const Vector3r diff = p - best_closest;
    const real diff_norm = diff.norm();
    if (diff_norm > real(1e-12)) normal_out = diff / diff_norm;
    else normal_out = best_face_normal;
}

template<int dim>
real MeshFrictionalBoundary<dim>::GetSignedDistanceWithReferenceNormal(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx,
    const Eigen::Matrix<real, dim, 1>& ref_normal) const {
    CheckError(has_q_, "MeshFrictionalBoundary requires UpdateVertices before query.");
    const Vector3r p = q;
    const Vector3r ref_n = ref_normal;
    const std::vector<std::array<int, 3>>* faces = &faces_left_;
    if (node_idx >= 0 && node_idx < split_index_) faces = &faces_right_;
    else if (node_idx >= split_index_) faces = &faces_left_;

    if (faces->empty()) return std::numeric_limits<real>::infinity();

    // Find globally closest point across ALL faces (no face-normal filtering).
    // Face-normal filtering caused collapse: as deformable bodies deform during
    // PD iterations, faces rotate so their outward normals no longer align with
    // the frame-start ref_n, causing valid front-facing contacts to be rejected
    // and delta_n to jump to +infinity → lambda collapses to 0 mid-iteration.
    real best_dist2 = std::numeric_limits<real>::infinity();
    Vector3r best_closest = Vector3r::Zero();
    for (const auto& face : *faces) {
        const Vector3r a = q_.segment(face[0] * 3, 3);
        const Vector3r b = q_.segment(face[1] * 3, 3);
        const Vector3r c = q_.segment(face[2] * 3, 3);
        const Vector3r closest = ClosestPointTriangle(p, a, b, c);
        const real dist2 = (p - closest).squaredNorm();
        if (dist2 < best_dist2) { best_dist2 = dist2; best_closest = closest; }
    }
    // Signed distance = projection of (p − closest) onto the fixed ref_n.
    // ref_n was computed at frame start as (p₀ − closest₀)/|...| pointing
    // FROM the opposing surface TOWARD the candidate node. This projection is:
    //   > 0  when p is on the same side as its frame-start position (outside)
    //   < 0  when p has crossed the surface (penetrating)
    // The projection is monotone in penetration depth and robust to face
    // rotation, since ref_n is fixed and we project against it directly.
    const real signed_proj = (p - best_closest).dot(ref_n);
    return signed_proj - contact_radius_;
}

template<int dim>
const Eigen::Matrix<real, dim, 1> MeshFrictionalBoundary<dim>::GetNormal(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return GetNormal(q, -1);
}

template<int dim>
const Eigen::Matrix<real, dim, 1> MeshFrictionalBoundary<dim>::GetNormal(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
    CheckError(has_q_, "MeshFrictionalBoundary requires UpdateVertices before query.");
    const Vector3r p = q;
    const std::vector<std::array<int, 3>>* faces = &faces_left_;
    if (node_idx >= 0 && node_idx < split_index_) {
        faces = &faces_right_;
    } else if (node_idx >= split_index_) {
        faces = &faces_left_;
    }

    const real eps = std::numeric_limits<real>::epsilon();
    real best_dist2 = std::numeric_limits<real>::infinity();
    Vector3r best_normal = Vector3r::UnitZ();

    for (const auto& face : *faces) {
        const Vector3r a = q_.segment(face[0] * 3, 3);
        const Vector3r b = q_.segment(face[1] * 3, 3);
        const Vector3r c = q_.segment(face[2] * 3, 3);
        const Vector3r closest = ClosestPointTriangle(p, a, b, c);
        const real dist2 = (p - closest).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            Vector3r n = (b - a).cross(c - a);
            const real n_norm = n.norm();
            if (n_norm > eps) n /= n_norm;
            else n = Vector3r::UnitZ();
            // Do NOT flip by (p-closest): when p penetrates the boundary body,
            // (p-closest) points inward, which would flip n to also point inward
            // and drive the contact force to push p deeper into penetration.
            // Face winding is guaranteed consistently outward by
            // _extract_boundary_faces (centroid-orientation pass in Python).
            best_normal = n;
        }
    }

    return best_normal;
}

template<int dim>
const std::vector<Eigen::Matrix<real, dim, 1>> MeshFrictionalBoundary<dim>::GetNormalDifferential(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return GetNormalDifferential(q, -1);
}

template<int dim>
const std::vector<Eigen::Matrix<real, dim, 1>> MeshFrictionalBoundary<dim>::GetNormalDifferential(
    const Eigen::Matrix<real, dim, 1>& q, const int node_idx) const {
    std::vector<Eigen::Matrix<real, dim, 1>> ret(dim);
    for (int k = 0; k < dim; ++k) ret[k] = Eigen::Matrix<real, dim, 1>::Zero();
    return ret;
}

template<int dim>
const bool MeshFrictionalBoundary<dim>::ForwardIntersect(
    const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v,
    const real dt, real& t_hit) const {
    return ForwardIntersect(q, v, dt, t_hit, -1);
}

template<int dim>
const bool MeshFrictionalBoundary<dim>::ForwardIntersect(
    const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v,
    const real dt, real& t_hit, const int node_idx) const {
    const Vector3r q_next = q + dt * v;
    const real dist_next = GetDistance(q_next, node_idx);
    if (dist_next > 0) return false;
    t_hit = 0;
    return true;
}

template<int dim>
void MeshFrictionalBoundary<dim>::BackwardIntersect(
    const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v,
    const real t_hit,
    const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
    Eigen::Matrix<real, dim, 1>& dl_dq,
    Eigen::Matrix<real, dim, 1>& dl_dv) const {
    BackwardIntersect(q, v, t_hit, dl_dq_hit, dl_dq, dl_dv, -1);
}

template<int dim>
void MeshFrictionalBoundary<dim>::BackwardIntersect(
    const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v,
    const real t_hit,
    const Eigen::Matrix<real, dim, 1>& dl_dq_hit,
    Eigen::Matrix<real, dim, 1>& dl_dq,
    Eigen::Matrix<real, dim, 1>& dl_dv,
    const int node_idx) const {
    dl_dq = dl_dq_hit;
    dl_dv = Eigen::Matrix<real, dim, 1>::Zero();
}

template class MeshFrictionalBoundary<3>;

// dim=2 explicit specializations: stubs required for linking.
// These are never called at runtime because SetFrictionalBoundary and Initialize
// both CheckError(dim == 3 / vertex_dim == 3) before any of these can execute.
template<>
const real MeshFrictionalBoundary<2>::GetDistance(
    const Eigen::Matrix<real, 2, 1>&, const int) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
    return 0;
}

template<>
const Eigen::Matrix<real, 2, 1> MeshFrictionalBoundary<2>::GetNormal(
    const Eigen::Matrix<real, 2, 1>&, const int) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
    return Eigen::Matrix<real, 2, 1>::Zero();
}

template<>
const Eigen::Matrix<real, 2, 2> MeshFrictionalBoundary<2>::GetLocalFrame(
    const Eigen::Matrix<real, 2, 1>&, const int) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
    return Eigen::Matrix<real, 2, 2>::Identity();
}

template<>
const bool MeshFrictionalBoundary<2>::ForwardIntersect(
    const Eigen::Matrix<real, 2, 1>&,
    const Eigen::Matrix<real, 2, 1>&,
    const real, real&, const int) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
    return false;
}

template<>
void MeshFrictionalBoundary<2>::GetClosestPointAndNormal(
    const Eigen::Matrix<real, 2, 1>&, const int,
    Eigen::Matrix<real, 2, 1>&, Eigen::Matrix<real, 2, 1>&) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
}

template<>
real MeshFrictionalBoundary<2>::GetSignedDistanceWithReferenceNormal(
    const Eigen::Matrix<real, 2, 1>&, const int,
    const Eigen::Matrix<real, 2, 1>&) const {
    CheckError(false, "MeshFrictionalBoundary only supports 3D.");
    return 0;
}

template class MeshFrictionalBoundary<2>;
