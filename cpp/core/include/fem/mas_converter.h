#pragma once

// Converts Eigen 3N×3N scalar SparseMatrix into node-level block-CSR format
// expected by the MAS preconditioner (SeSchwarzPreconditioner).
//
// MAS expects:
//   - diagonal[i]:                  3×3 float block for node i's self-coupling
//   - csrOffDiagonals[starts[i]+k]: 3×3 float block for node i's k-th neighbour
//   - csrRanges (= starts):         row pointer array, same order as neighbours
//   - neighbours (SeCsr<int>):      node-level adjacency (excluding self)
//   - positions:                    SeVec3fSimd per node (float, SIMD-aligned)

#include "common/config.h"
#include "SeSchwarzPreconditioner.h"
#include "SeCsr.h"
#include "SeMatrix.h"
#include "SeVectorSimd.h"

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <unordered_set>

struct MasBlockCsr {
    int numVerts = 0;
    std::vector<SE::SeMatrix3f>  diagonal;         // [N]
    std::vector<SE::SeMatrix3f>  csrOffDiagonals;  // [total off-diag blocks]
    SE::SeCsr<int>               neighbours;        // node-level CSR (excludes self)
    std::vector<SE::SeVec3fSimd> positions;         // [N]

    // Build all data from an Eigen 3N×3N SparseMatrix and position vector q.
    // mat: Newton Hessian (3N×3N, column-major Eigen::SparseMatrix<double>)
    // q:   vertex positions (3N, interleaved x0 y0 z0 x1 y1 z1 ...)
    void BuildFromEigen(const SparseMatrix& mat, const VectorXr& q) {
        assert(mat.rows() == mat.cols());
        const int dofs = static_cast<int>(mat.rows());
        assert(dofs % 3 == 0);
        numVerts = dofs / 3;

        // ================================================================
        // Step 1: Discover node-level adjacency from sparsity pattern.
        //
        // Eigen SparseMatrix<double> is column-major by default.
        // We scan all nonzeros and collect (row_node, col_node) pairs.
        // ================================================================
        std::vector<std::unordered_set<int>> adj_set(numVerts);

        for (int col = 0; col < dofs; ++col) {
            const int col_node = col / 3;
            for (SparseMatrix::InnerIterator it(mat, col); it; ++it) {
                const int row_node = static_cast<int>(it.row()) / 3;
                if (row_node != col_node) {
                    adj_set[row_node].insert(col_node);
                }
            }
        }

        // Convert to sorted adjacency lists
        std::vector<std::vector<int>> adj(numVerts);
        for (int i = 0; i < numVerts; ++i) {
            adj[i].assign(adj_set[i].begin(), adj_set[i].end());
            std::sort(adj[i].begin(), adj[i].end());
        }
        adj_set.clear();

        // ================================================================
        // Step 2: Build SeCsr<int> neighbours
        // ================================================================
        std::vector<int> starts(numVerts + 1, 0);
        for (int i = 0; i < numVerts; ++i) {
            starts[i + 1] = starts[i] + static_cast<int>(adj[i].size());
        }
        std::vector<int> idxs(starts[numVerts]);
        for (int i = 0; i < numVerts; ++i) {
            std::memcpy(idxs.data() + starts[i], adj[i].data(),
                        sizeof(int) * adj[i].size());
        }
        std::vector<int> dummy_values;
        neighbours = SE::SeCsr<int>(starts, idxs, dummy_values);

        // ================================================================
        // Step 3: Build lookup for fast block extraction.
        //
        // For each node i and neighbour j, we need to know the index k
        // such that adj[i][k] == j, so we can write to the correct slot
        // in csrOffDiagonals.
        //
        // We also pre-allocate diagonal and off-diagonal with zeros.
        // ================================================================
        diagonal.assign(numVerts, SE::SeMatrix3f(0.0f));
        csrOffDiagonals.assign(starts[numVerts], SE::SeMatrix3f(0.0f));

        // Build neighbour-to-slot lookup per node
        // adj_slot[i][j] = k  means adj[i][k] == j
        std::vector<std::unordered_map<int,int>> adj_slot(numVerts);
        for (int i = 0; i < numVerts; ++i) {
            for (int k = 0; k < static_cast<int>(adj[i].size()); ++k) {
                adj_slot[i][adj[i][k]] = k;
            }
        }

        // ================================================================
        // Step 4: Single scan over all nonzeros — fill diagonal & off-diag.
        //
        // For each scalar entry mat(row, col) with value v:
        //   row_node = row / 3,  ri = row % 3
        //   col_node = col / 3,  ci = col % 3
        //
        //   if row_node == col_node:
        //     diagonal[row_node](ri, ci) += v
        //   else:
        //     csrOffDiagonals[starts[row_node] + adj_slot[row_node][col_node]](ri, ci) += v
        //
        // SeMatrix3f is column-major: (ri, ci) maps to data[ci * 3 + ri].
        // ================================================================
        for (int col = 0; col < dofs; ++col) {
            const int col_node = col / 3;
            const int ci = col % 3;
            for (SparseMatrix::InnerIterator it(mat, col); it; ++it) {
                const int row = static_cast<int>(it.row());
                const int row_node = row / 3;
                const int ri = row % 3;
                const float val = static_cast<float>(it.value());

                if (row_node == col_node) {
                    diagonal[row_node](ri, ci) = val;
                } else {
                    auto slot_it = adj_slot[row_node].find(col_node);
                    if (slot_it != adj_slot[row_node].end()) {
                        csrOffDiagonals[starts[row_node] + slot_it->second](ri, ci) = val;
                    }
                }
            }
        }

        // ================================================================
        // Step 5: Convert positions (double → float SIMD)
        // ================================================================
        positions.resize(numVerts);
        for (int i = 0; i < numVerts; ++i) {
            positions[i] = SE::SeVec3fSimd(
                static_cast<float>(q(3 * i + 0)),
                static_cast<float>(q(3 * i + 1)),
                static_cast<float>(q(3 * i + 2))
            );
        }
    }

    // Convenience: get csrRanges pointer for PreparePreconditioner().
    // This returns the internal starts array of the SeCsr.
    const int* CsrRanges() const {
        return neighbours.StartPtr(0);
    }
};
