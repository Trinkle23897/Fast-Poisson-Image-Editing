#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <tuple>
#include "solver.h"

// FIX 1: Removed the duplicate OpenMPBlockRBSolver(tile_size, n_cpu) call.
//         C++ only allows each base class to appear once in the initializer
//         list. The single call below is enough — it satisfies the inherited
//         constructor and sets tile_size / n_cpu correctly.
OpenMPMultiSweepsRedBlackSolver::OpenMPMultiSweepsRedBlackSolver(
    int tile_size, int n_cpu, int a_max, float conv_threshold)
    : OpenMPBlockRBSolver(tile_size, n_cpu),
      tile_residuals(NULL),
      a_max(a_max),
      conv_threshold(conv_threshold) {}

OpenMPMultiSweepsRedBlackSolver::~OpenMPMultiSweepsRedBlackSolver() {
  if (tile_residuals != NULL) {
    delete[] tile_residuals;
  }
}

void OpenMPMultiSweepsRedBlackSolver::post_reset() {
  // Let the parent allocate tmp[], imgbuf[], etc. first.
  OpenMPBlockRBSolver::post_reset();

  int T_x = (N + tile_size - 1) / tile_size;
  int T_y = (M + tile_size - 1) / tile_size;
  if (tile_residuals != NULL) {
    delete[] tile_residuals;
  }
  tile_residuals = new float[T_x * T_y];
  memset(tile_residuals, 0, sizeof(float) * T_x * T_y);
}

// FIX 2: Removed the duplicate update_tile() and calc_error() definitions.
//         OpenMPMultiSweepsRedBlackSolver inherits them from
//         OpenMPBlockRBSolver — they are identical, so redefining them here
//         was dead code that also forced the compiler to link an ambiguous
//         override, preventing the vtable from being set up correctly.

// After calc_error() has filled tmp[] with per-pixel absolute residuals,
// sum them per tile so we know which tiles still need work.
void OpenMPMultiSweepsRedBlackSolver::calc_tile_residuals() {
  int T_x = (N + tile_size - 1) / tile_size;
  int T_y = (M + tile_size - 1) / tile_size;

#pragma omp parallel for collapse(2) schedule(static)
  for (int ti = 0; ti < T_x; ti++) {
    for (int tj = 0; tj < T_y; tj++) {
      int tile_idx = ti * T_y + tj;
      float tile_res = 0.0f;

      int r0 = ti * tile_size;
      int r1 = r0 + tile_size < N ? r0 + tile_size : N;
      int c0 = tj * tile_size;
      int c1 = c0 + tile_size < M ? c0 + tile_size : M;

      for (int x = r0; x < r1; x++) {
        for (int y = c0, id = x * M + c0; y < c1; y++, id++) {
          if (mask[id]) {
            int off3 = id * 3;
            // tmp[] holds absolute residual per channel (set by calc_error)
            tile_res += tmp[off3 + 0] + tmp[off3 + 1] + tmp[off3 + 2];
          }
        }
      }
      tile_residuals[tile_idx] = tile_res;
    }
  }
}

// Residual Proportional Scaling:
//   a_{k+1} = max(1, round(a_max * (1 - p_k)))
//   where p_k = current_residual / initial_residual  (clamped to [0,1])
//
// Interpretation:
//   - p_k close to 1  → tile barely converged → many sweeps
//   - p_k close to 0  → tile nearly done      → few sweeps (minimum 1)
int OpenMPMultiSweepsRedBlackSolver::compute_adaptive_sweeps(
    float initial_residual, float current_residual) {
  if (initial_residual < 1e-8f) {
    return 1;  // tile was already converged at the start; no-op sweep
  }
  float p_k = current_residual / initial_residual;
  if (p_k > 1.0f) p_k = 1.0f;
  return std::max(1, (int)std::round(a_max * (1.0f - p_k)));
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenMPMultiSweepsRedBlackSolver::step(int iteration) {
  int T_x = (N + tile_size - 1) / tile_size;
  int T_y = (M + tile_size - 1) / tile_size;

  // Snapshot per-tile residuals before the first iteration so that
  // compute_adaptive_sweeps() always has a stable denominator.
  calc_error();
  calc_tile_residuals();

  float* initial_residuals = new float[T_x * T_y];
  memcpy(initial_residuals, tile_residuals, sizeof(float) * T_x * T_y);

  for (int i = 0; i < iteration; ++i) {

    // ── RED phase ────────────────────────────────────────────────────────────
    // Tiles where (ti+tj) is even. Red tiles only border black tiles, so
    // concurrent red tiles never touch the same pixel → no data race.
#pragma omp parallel for collapse(2) schedule(static)
    for (int ti = 0; ti < T_x; ti++) {
      for (int tj = 0; tj < T_y; tj++) {
        if ((ti + tj) % 2 != 0) continue;
        int tile_idx = ti * T_y + tj;
        int r0 = ti * tile_size;
        int r1 = r0 + tile_size < N ? r0 + tile_size : N;
        int c0 = tj * tile_size;
        int c1 = c0 + tile_size < M ? c0 + tile_size : M;

        int a_k = compute_adaptive_sweeps(initial_residuals[tile_idx],
                                          tile_residuals[tile_idx]);
        for (int sweep = 0; sweep < a_k; ++sweep) {
          update_tile(r0, r1, c0, c1);
        }
      }
    }
    // Implicit OMP barrier: all red writes visible before black reads them.

    // ── BLACK phase ──────────────────────────────────────────────────────────
    // Tiles where (ti+tj) is odd. Reads the fresh red-tile boundary values.
#pragma omp parallel for collapse(2) schedule(static)
    for (int ti = 0; ti < T_x; ti++) {
      for (int tj = 0; tj < T_y; tj++) {
        if ((ti + tj) % 2 != 1) continue;
        int tile_idx = ti * T_y + tj;
        int r0 = ti * tile_size;
        int r1 = r0 + tile_size < N ? r0 + tile_size : N;
        int c0 = tj * tile_size;
        int c1 = c0 + tile_size < M ? c0 + tile_size : M;

        int a_k = compute_adaptive_sweeps(initial_residuals[tile_idx],
                                          tile_residuals[tile_idx]);
        for (int sweep = 0; sweep < a_k; ++sweep) {
          update_tile(r0, r1, c0, c1);
        }
      }
    }
    // Implicit OMP barrier.

    // Refresh per-tile residuals for the next iteration's sweep-count formula.
    if (i < iteration - 1) {
      calc_error();
      calc_tile_residuals();
    }
  }

  // Final error for the return value.
  calc_error();

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) {
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : (unsigned char)tgt[i];
  }

  delete[] initial_residuals;

  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}