#ifndef SOLVER_MAS_PRECONDITIONER_H
#define SOLVER_MAS_PRECONDITIONER_H

#include "common/config.h"
#include "SeSchwarzPreconditioner.h"
#include "SeVectorSimd.h"

// Global pointer to the MAS preconditioner instance, set before CG solve.
// Follows the same pattern as global_deformable for DeformablePreconditioner.
extern SE::SeSchwarzPreconditioner* global_mas_preconditioner;
extern int global_mas_num_verts;

namespace Eigen {

// Eigen CG preconditioner that delegates to MAS.
// Plugs into: ConjugateGradient<SparseMatrix, Lower|Upper, MasPreconditioner<real>>
template <typename _Scalar>
class MasPreconditioner {
    typedef _Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, 1> Vector;
public:
    typedef typename Vector::StorageIndex StorageIndex;
    enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
    };

    MasPreconditioner() : rows_(0), cols_(0) {}

    template<typename MatType>
    explicit MasPreconditioner(const MatType& mat)
        : rows_(mat.rows()), cols_(mat.cols()) {
        compute(mat);
    }

    Index rows() const { return rows_; }
    Index cols() const { return cols_; }

    // analyzePattern / factorize / compute are no-ops.
    // All MAS setup is done externally before CG solve.
    template<typename MatType>
    MasPreconditioner& analyzePattern(const MatType&) { return *this; }

    template<typename MatType>
    MasPreconditioner& factorize(const MatType& mat) { return *this; }

    template<typename MatType>
    MasPreconditioner& compute(const MatType& mat) {
        rows_ = mat.rows();
        cols_ = mat.cols();
        return *this;
    }

    // Core: z = M_MAS^{-1} * b
    // Converts double <-> float for MAS (which operates in float).
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
        const int N = global_mas_num_verts;
        const int dofs = N * 3;

        // Convert residual: double -> SeVec3fSimd (float, node-interleaved)
        std::vector<SE::SeVec3fSimd> r(N);
        for (int i = 0; i < N; ++i) {
            r[i] = SE::SeVec3fSimd(
                static_cast<float>(b(3 * i + 0)),
                static_cast<float>(b(3 * i + 1)),
                static_cast<float>(b(3 * i + 2))
            );
        }

        // Apply MAS preconditioning: z = M^{-1} * r
        std::vector<SE::SeVec3fSimd> z(N);
        global_mas_preconditioner->Preconditioning(z.data(), r.data(), 3);

        // Convert back: float -> double
        for (int i = 0; i < N; ++i) {
            x(3 * i + 0) = static_cast<Scalar>(z[i].x);
            x(3 * i + 1) = static_cast<Scalar>(z[i].y);
            x(3 * i + 2) = static_cast<Scalar>(z[i].z);
        }
    }

    template<typename Rhs> inline const Solve<MasPreconditioner, Rhs>
    solve(const MatrixBase<Rhs>& b) const {
        eigen_assert(rows_ == b.rows() && rows_ == cols_);
        return Solve<MasPreconditioner, Rhs>(*this, b.derived());
    }

    ComputationInfo info() { return Success; }

protected:
    Index rows_, cols_;
};

}  // namespace Eigen

#endif
