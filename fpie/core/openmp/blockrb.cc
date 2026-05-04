#include <cmath>
#include <cstring>
#include <omp.h>
#include <tuple>
#include "solver.h"

OpenMPBlockRBSolver::OpenMPBlockRBSolver(int tile_size, int n_cpu)
    : imgbuf(NULL), tmp(NULL), tile_size(tile_size),
      GridSolver(tile_size, tile_size) {
    // GridSolver(tile_size, tile_size) satisfies the base constructor signature
    // (it expects grid_x, grid_y). The values are unused by us -- our step()
    // does its own tiling. We never call OpenMPGridSolver so there is no
    // dependency on its parallel-for structure.
    omp_set_num_threads(n_cpu);
}

OpenMPBlockRBSolver::~OpenMPBlockRBSolver() {
    if (tmp != NULL) {
        delete[] tmp;
        delete[] imgbuf;
    }
}

void OpenMPBlockRBSolver::post_reset() {
    if (tmp != NULL) {
        delete[] tmp;
        delete[] imgbuf;
    }
    // tmp is only used by calc_error() -- we do NOT use it as a write buffer
    // in step(), unlike GridSolver. All step() updates go directly into tgt[].
    tmp = new float[N * m3];
    imgbuf = new unsigned char[N * m3];
    memset(tmp, 0, sizeof(float) * N * m3);
}

// Update all active pixels in one tile, row-major Gauss-Seidel order.
// Writes directly into tgt[] -- no double buffer.
// Because this runs single-threaded per tile and the tile fits in L1 cache,
// every neighbour read after the first row/col is a cache hit.
inline void OpenMPBlockRBSolver::update_tile(int r0, int r1, int c0, int c1) {
    for (int x = r0; x < r1; x++) {
        for (int y = c0, id = x * M + c0; y < c1; y++, id++) {
        if (!mask[id]) continue;
        int off3 = id * 3;
        int id0  = off3 - m3;  // (x-1, y)
        int id1  = off3 - 3;   // (x,   y-1)
        int id2  = off3 + 3;   // (x,   y+1)
        int id3  = off3 + m3;  // (x+1, y)
        tgt[off3 + 0] = (grad[off3 + 0] + tgt[id0 + 0] + tgt[id1 + 0] +
                        tgt[id2 + 0]   + tgt[id3 + 0]) / 4.0f;
        tgt[off3 + 1] = (grad[off3 + 1] + tgt[id0 + 1] + tgt[id1 + 1] +
                        tgt[id2 + 1]   + tgt[id3 + 1]) / 4.0f;
        tgt[off3 + 2] = (grad[off3 + 2] + tgt[id0 + 2] + tgt[id1 + 2] +
                        tgt[id2 + 2]   + tgt[id3 + 2]) / 4.0f;
        }
    }
}

void OpenMPBlockRBSolver::calc_error() {
    // Identical to OpenMPGridSolver::calc_error() -- reads from tgt[], writes
    // per-pixel residuals into tmp[], then sums into err[].
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < N * M; ++id) {
        if (mask[id]) {
        int off3 = id * 3;
        int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
        tmp[off3+0] = std::abs(grad[off3+0] + tgt[id0+0] + tgt[id1+0] +
                                tgt[id2+0]   + tgt[id3+0] - tgt[off3+0] * 4.0f);
        tmp[off3+1] = std::abs(grad[off3+1] + tgt[id0+1] + tgt[id1+1] +
                                tgt[id2+1]   + tgt[id3+1] - tgt[off3+1] * 4.0f);
        tmp[off3+2] = std::abs(grad[off3+2] + tgt[id0+2] + tgt[id1+2] +
                                tgt[id2+2]   + tgt[id3+2] - tgt[off3+2] * 4.0f);
        }
    }
    memset(err, 0, sizeof(err));
    for (int id = 0; id < N * M; ++id) {
        int off3 = id * 3;
        err[0] += tmp[off3+0];
        err[1] += tmp[off3+1];
        err[2] += tmp[off3+2];
    }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenMPBlockRBSolver::step(int iteration) {
    int T_x = (N + tile_size - 1) / tile_size;
    int T_y = (M + tile_size - 1) / tile_size;

    for (int i = 0; i < iteration; ++i) {

    // RED phase -- tiles where (ti+tj) is even.
    // Red tiles only border black tiles, so concurrent red tiles never
    // write adjacent pixels. No data race within this parallel-for.
    #pragma omp parallel for collapse(2) schedule(static)
        for (int ti = 0; ti < T_x; ti++) {
        for (int tj = 0; tj < T_y; tj++) {
            if ((ti + tj) % 2 != 0) continue;
            int r0 = ti * tile_size, r1 = r0 + tile_size < N ? r0 + tile_size : N;
            int c0 = tj * tile_size, c1 = c0 + tile_size < M ? c0 + tile_size : M;
            update_tile(r0, r1, c0, c1);
        }
        }
    // Implicit OMP barrier: all red writes committed before black reads them.

    // BLACK phase -- tiles where (ti+tj) is odd.
    // Black tiles read fresh red-tile boundary values from above.
    #pragma omp parallel for collapse(2) schedule(static)
        for (int ti = 0; ti < T_x; ti++) {
        for (int tj = 0; tj < T_y; tj++) {
            if ((ti + tj) % 2 != 1) continue;
            int r0 = ti * tile_size, r1 = r0 + tile_size < N ? r0 + tile_size : N;
            int c0 = tj * tile_size, c1 = c0 + tile_size < M ? c0 + tile_size : M;
            update_tile(r0, r1, c0, c1);
        }
        }
        // Implicit OMP barrier.
    }

    calc_error();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * M * 3; ++i) {
        imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
    }
    return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}