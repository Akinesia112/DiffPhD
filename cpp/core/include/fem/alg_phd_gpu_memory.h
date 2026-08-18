#pragma once

#include <vector>

// Total bytes held by the persistent AlgPhd GPU S / S^T CSR buffers across all
// 3 axes. Returns 0 when no AlgPhd forward has been run yet or when the binary
// was built without CUDA. Counts d_row + d_col + d_val per GpuCSR:
//   per-CSR bytes = 4*(N+1) + 4*nnz + 8*nnz
long long AlgPhdSMemoryBytes();

// Total non-zeros of one S factor (axis 0). Together with the vertex count
// this gives the sparsity density used in the contrast-sweep table.
long long AlgPhdSNnz();

// Vertex count N cached by the last AlgPhd forward (per-axis block size of A).
int AlgPhdCachedN();

// ── Per-frame PD iteration statistics ────────────────────────────────────────
// The AlgPhd forward appends one entry per frame. Python reads these after a
// rollout to report mean/max PD iterations and the non-convergence count,
// which is the machine-independent measure of how material heterogeneity
// affects solver conditioning.
//
// Call AlgPhdResetFrameStats() before a rollout, then read the two vectors after.
void AlgPhdResetFrameStats();
std::vector<int> AlgPhdFrameIters();       // PD iterations used by each frame
std::vector<int> AlgPhdFrameConverged();   // 1 = converged, 0 = hit max_pd_iter
