# Backend

To specify backend, simply typing `-b cuda` or `--backend openmp`, together with other parameters described below.

Feel free to play `fpie` with other arguments!

## GridSolver

GridSolver keeps most of the 2D structure of the image, instead of relabeling pixels as EquSolver. To use GridSolver in some of the following backends, you need to specify `--grid-x` and `--grid-y` to determine the access pattern of the large 2D array.

Here is a Python pseudocode to show how it works:

```python
arr = np.random.random(size=[N, M])
# here is a sequential scan:
for i in range(N):
    for j in range(M):
        func(arr[i, j])
# however, we can use block-level access pattern to improve the cache hit rate:
for i in range(N // grid_x):
    for j in range(M // grid_y):
        # the grid size is (grid_x, grid_y)
        for x in range(grid_x):
            for y in range(grid_y):
                func(arr[i * grid_x + x, j * grid_y + y])
```

## NumPy

This backend uses NumPy vectorized operation for parallel computation.

There's no extra parameter for NumPy EquSolver:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b numpy --method equ
Successfully initialize PIE equ solver with numpy backend
# of vars: 12559
Iter 5000, abs error [450.09415 445.24747 636.1397 ]
Time elapsed: 3.26s
Successfully write image to result.jpg
```

There's no extra parameter for NumPy GridSolver:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b numpy --method grid
Successfully initialize PIE grid solver with numpy backend
# of vars: 17227
Iter 5000, abs error [450.07922 445.27014 636.1374 ]
Time elapsed: 3.09s
Successfully write image to result.jpg
```

For NumPy backend, GridSolver is a little bit faster than EquSolver. However, for other backends, GridSolver is always slower than EquSolver.

## GCC

This backend uses a single thread C++ program to perform computation.

There's no extra parameter for GCC EquSolver:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b gcc --method equ
Successfully initialize PIE equ solver with gcc backend
# of vars: 12559
Iter 5000, abs error [ 5.179281   6.6939087 11.006622 ]
Time elapsed: 0.29s
Successfully write image to result.jpg
```

For GCC GridSolver, you need to specify `--grid-x` and `--grid-y` described in the first section:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b gcc --method grid --grid-x 8 --grid-y 8
Successfully initialize PIE grid solver with gcc backend
# of vars: 17227
Iter 5000, abs error [ 5.1776047  6.69458   11.001862 ]
Time elapsed: 0.36s
Successfully write image to result.jpg
```

## Taichi

[Taichi](https://github.com/taichi-dev/taichi) is an open-source, imperative, parallel programming language for high-performance numerical computation. We provide 3 choices: `taichi-cpu` for CPU-level parallelization, `taichi-gpu` and `taichi-cuda` for GPU-level parallelization. You can install taichi via `pip install taichi`.

- For `taichi-cpu`: use `-c` or `--cpu` to determine how many CPUs it will use;
- For `taichi-gpu` or `taichi-cuda`: use `-z` or `--block-size` to determine the number of threads used in a block.

The parallelization strategy for Taichi backend is written by Taichi itself.

There's no other parameters for Taichi EquSolver:

```bash
# taichi-cpu
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b taichi-cpu --method equ -c 6 
[Taichi] version 0.9.2, llvm 10.0.0, commit 7a4d73cd, linux, python 3.8.10
[Taichi] Starting on arch=x64
Successfully initialize PIE equ solver with taichi-cpu backend
# of vars: 12559
Iter 5000, abs error [ 5.1899223  6.708023  11.034821 ]
Time elapsed: 0.57s
Successfully write image to result.jpg
```

```bash
# taichi-gpu
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b taichi-gpu --method equ -z 1024
[Taichi] version 0.9.2, llvm 10.0.0, commit 7a4d73cd, linux, python 3.8.10
[Taichi] Starting on arch=cuda
Successfully initialize PIE equ solver with taichi-gpu backend
# of vars: 12559
Iter 5000, abs error [37.35366  46.433205 76.09506 ]
Time elapsed: 0.60s
Successfully write image to result.jpg
```

For Taichi GridSolver, you also need to specify `--grid-x` and `--grid-y` described in the first section:

```bash
# taichi-cpu
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b taichi-cpu --method grid --grid-x 16 --grid-y 16 -c 12
[Taichi] version 0.9.2, llvm 10.0.0, commit 7a4d73cd, linux, python 3.8.10
[Taichi] Starting on arch=x64
Successfully initialize PIE grid solver with taichi-cpu backend
# of vars: 17227
Iter 5000, abs error [ 5.310623   6.8661118 11.2751465]
Time elapsed: 0.73s
Successfully write image to result.jpg
```

```bash
# taichi-gpu
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b taichi-gpu --method grid --grid-x 8 --grid-y 8 -z 64 
[Taichi] version 0.9.2, llvm 10.0.0, commit 7a4d73cd, linux, python 3.8.10
[Taichi] Starting on arch=cuda
Successfully initialize PIE grid solver with taichi-gpu backend
# of vars: 17227
Iter 5000, abs error [37.74704  46.853233 74.741455]
Time elapsed: 0.63s
Successfully write image to result.jpg
```

## OpenMP

OpenMP backend needs to specify the number of CPU cores it can use, with `-c` or `--cpu` option (default choice is to use all CPU cores).

There's no other parameters for OpenMP EquSolver:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b openmp --method equ -c 6
Successfully initialize PIE equ solver with openmp backend
# of vars: 12559
Iter 5000, abs error [ 5.2758713  6.768402  11.11969  ]
Time elapsed: 0.06s
Successfully write image to result.jpg
```

For OpenMP GridSolver, you also need to specify `--grid-x` and `--grid-y` described in the first section:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b openmp --method grid --grid-x 8 --grid-y 8 -c 6
Successfully initialize PIE grid solver with openmp backend
# of vars: 17227
Iter 5000, abs error [ 5.187172  6.701462 11.020264]
Time elapsed: 0.10s
Successfully write image to result.jpg
```

### Parallelization Strategy

For [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/equ.cc), it first groups the pixels into two folds by `(i+j)%2`, then parallelizes per-pixel iteration inside a group in each step. This strategy can utilize the thread-local accessment.

For [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/grid.cc), it parallelizes per-grid iteration in each step, where the grid size is `(grid_x, grid_y)`. It simply iterates all pixels in each grid.

## MPI

To run with MPI backend, you need to install both mpicc and mpi4py (`pip install mpi4py`).

Different from other methods, you need to use `mpiexec` or `mpirun` to launch MPI service instead of directly calling `fpie` program. `-np` option is to indicate the number of process it will launch.

Apart from that, you need to specify the synchronization interval for MPI backend with `--mpi-sync-interval`. If this number is too small, it will cause a large amount of overhead of synchronization; however, if it is too large, the quality of solution drops down dramatically.

MPI EquSolver and GridSolver don't have any other arguments because of the parallelization strategy we used, see the next section.

```bash
$ mpiexec -np 6 fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b mpi --method equ --mpi-sync-interval 100 
Successfully initialize PIE equ solver with mpi backend
# of vars: 12559
Iter 5000, abs error [264.6767  269.55304 368.4869 ]
Time elapsed: 0.10s
Successfully write image to result.jpg
```

```bash
$ mpiexec -np 6 fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b mpi --method grid --mpi-sync-interval 100
Successfully initialize PIE grid solver with mpi backend
# of vars: 17227
Iter 5000, abs error [204.41124 215.00548 296.4441 ]
Time elapsed: 0.13s
Successfully write image to result.jpg
```

### Parallelization Strategy

MPI cannot use share-memory program model, so that we need to reduce the amount of data for communication. Each process is only responsible for a part of computation, and synchronized with other process per `mpi_sync_interval` steps.

For [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/equ.cc), it's hard to say which part of the data should be exchanged to other process, since it relabels all pixels at the very beginning of this process. We use `MPI_Bcast` to force sync all data.

For [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/grid.cc), we use line partition: process `i` exchanges its first and last line data with process `i-1` and `i+1` separately. This strategy has a continuous memory layout to exchange, thus has less overhead comparing with block partition.

However, even if we don't use the synchronization in MPI (set `mpi_sync_interval` to be greater than the number of iteration), it is still slower than OpenMP and CUDA backends.

## CUDA

CUDA backend needs to specify the number of threads in one block it will use, with `-z` or `--block-size` option (default value is 1024).

There's no other parameters for CUDA EquSolver:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b cuda --method equ -z 256
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1060
   SMs:        10
   Global mem: 6078 MB
   CUDA Cap:   6.1
---------------------------------------------------------
Successfully initialize PIE equ solver with cuda backend
# of vars: 12559
Iter 5000, abs error [37.63664 48.39614 79.6199 ]
Time elapsed: 0.06s
Successfully write image to result.jpg
```

For CUDA GridSolver, you also need to specify `--grid-x` and `--grid-y` described in the first section:

```bash
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result.jpg -h1 130 -w1 130 -n 5000 -g src -b cuda --method grid --grid-x 4 --grid-y 128 -z 1024
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce GTX 1060
   SMs:        10
   Global mem: 6078 MB
   CUDA Cap:   6.1
---------------------------------------------------------
Successfully initialize PIE grid solver with cuda backend
# of vars: 17227
Iter 5000, abs error [37.50096  48.061874 79.06909 ]
Time elapsed: 0.07s
Successfully write image to result.jpg
```

### Parallelization Strategy

The strategy used in CUDA backend is quite similar to OpenMP.

For [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/equ.cu), it performs equation-level parallelization.

For [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/grid.cu), each grid with size `(grid_x, grid_y)` will be in the same block. A thread in a block performs iteration only for a single pixel.
