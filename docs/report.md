# Final Report

## Summary

We implemented a parallelized Poisson image editor with Jacobi method. It can compute results using seven extensions: NumPy, [Numba](https://github.com/numba/numba), [Taichi](https://github.com/taichi-dev/taichi), single-thread c++, OpenMP, MPI, and CUDA. In terms of performance, we have a detailed benchmarking result that the CUDA backend can achieve 31 to 42 times faster on GHC machines compared to the single-threaded c++ implementation. In terms of user-experience, we have a simple GUI to demonstrate the results interactively, released a standard [PyPI package](https://pypi.org/project/fpie/), and provide [a website](https://fpie.readthedocs.io/) for project documentation.

| Source image                                                 | Mask image                                                   | Target image                                                 | Result image                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_src.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_mask.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_target.png) | ![](/_static/images/result2.jpg) |

## Background

### Poisson Image Editing

[Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) is a technique that can blend two images together without artifacts. Given a source image and its corresponding mask, and a coordination on target image, this algorithm can always generate amazing result. The general idea is to keep most of gradient in source image, while matching the boundary of source image and target image pixels.

The gradient per pixel is computed by

$$\nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1)$$

After computing the gradient in source image, the algorithm tries to solve the following problem: given the gradient and the boundary value, calculate the approximate solution that meets the requirement, i.e., to keep target image's gradient as similar as the source image.

This process can be formulated as $(4-A)\vec{x}=\vec{b}$, where $A\in\mathbb{R}^{N\times N}$, $\vec{x}\in\mathbb{R}^N$, $\vec{b}\in\mathbb{R}^N$, $N$ is the number of pixels in the mask, $A$ is a giant sparse matrix because each line of A only contains at most 4 non-zero value (neighborhood), $\vec{b}$ is the gradient from source image, and $\vec{x}$ is the result value.

$N$ is always a large number, i.e., greater than 50k, so the Gauss-Jordan Elimination cannot be directly applied here because of the high time complexity $O(N^3)$. People use [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) to solve the problem. Thanks to the sparsity of matrix $A$, the overall time complexity is $O(MN)$ where $M$ is the number of iteration performed by Poisson image editing. The iterative equation is $\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4$.

This project parallelizes Jacobi method to speed up the computation. To our best knowledge, there's no public project on GitHub that implements Poisson image editing with either OpenMP, or MPI, or CUDA. All of them can only handle a small size image workload ([link](https://github.com/PPPW/poisson-image-editing/issues/1)).

### PIE Solver

We implemented two different solvers: EquSolver and GridSolver.

EquSolver directly constructs the equations $(4-A)\vec{x}=\vec{b}$ by re-labeling the pixel, and use Jacobi method to get the solution via $\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4$.

```python
""" EquSolver pseudocode."""

# pre-process
src, mask, tgt = read_images(src_name, mask_name, tgt_name)
A = build_A(mask)            # shape: (N, 4), dtype: int
X = build_X(tgt, mask)       # shape: (N, 3), dtype: float
B = build_B(src, mask, tgt)  # shape: (N, 3), dtype: float

# major computation, can be parallelized
for _ in range(n_iter):
    X = (X[A[:, 0]] + X[A[:, 1]] + X[A[:, 2]] + X[A[:, 3]] + B) / 4.0

# post-process
out = merge(tgt, X, mask)
write_image(out_name, out)
```

GridSolver uses the same Jacobi iteration, however, it keeps the 2D structure of the original image instead of re-labeling the pixel in the mask. It may take some advantage when the mask region covers all of the image, because in this case GridSolver can save 4 read instructions by directly calculating the neighborhood's coordinate. Meanwhile, it has a better locality of fetching required data per iteration if we properly setup the access pattern (will be discussed in Section [Access Pattern](#access-pattern)).

```python
""" GridSolver pseudocode."""

# pre-process
src, mask, tgt = read_images(src_name, mask_name, tgt_name)
# mask: shape: (N, M), dtype: uint8
grad = calc_grad(src, mask, tgt)  # shape: (N, M, 3), dtype: float
x, y = np.nonzero(mask)  # find element-wise pixel index of mask array

# major computation, can be parallelized
for _ in range(n_iter):
    tgt[x, y] = (tgt[x - 1, y] + tgt[x, y - 1] + tgt[x, y + 1] + tgt[x + 1, y] + grad[x, y]) / 4.0

# post-process
write_image(out_name, tgt)
```

The bottleneck for both solvers is the for-loop and can be easily parallelized. The implementation detail and parallelization strategies will be discussed in Section [Parallelization Strategy](#parallelization-strategy).


## Method

### Language and Hardware Setup

We start to build PIE with the help of [pybind11](https://github.com/pybind/pybind11) because our goal is to benchmark multiple parallelization approaches, including hand-written CUDA code and other 3rd-party libraries such as NumPy.

One of our project goal is to let the algorithm run on any \*nix machine and can have a real-time interactive result demonstration. For this reason, we don't choose super computing cluster as the hardware setup. Instead, we choose GHC machine to develop and measure the performance, which has 8x i7-9700 cores and an Nvidia RTX 2080Ti.

### Access Pattern

For EquSolver, we can re-organize the pixel order to achieve a better locality when performing parallel operations. Specifically, we can group all pixels into two folds by `(x + y) % 2`. Here is a small example:

```
# before
x1   x2   x3   x4   x5
x6   x7   x8   x9   x10
x11  x12  x13  x14  x15
...

# re-order
x1   x10  x2   x11  x3
x12  x4   x13  x5   x14
x6   x15  x7   x16  x8
...
```

By doing so, every pixel's 4 neighbors are closer with each other. The ideal access pattern is to separately iterate these two groups, i.e.,

```python
for _ in range(n_iter):
    parallel for i in range(1, p):
        # i < p, neighbor >= p
        x_[i] = calc(b[i], neighbor(x, i))

    parallel for i in range(p, N):
        # i >= p, neighbor < p
        x[i] = calc(b[i], neighbor(x_, i))
```

Unfortunately, we only observe a clear advantage with OpenMP EquSolver. For other backend, the sequential id assignment is much better than re-order. The related discussion is in Section [Parallelization Strategy - OpenMP](#openmp).

For GridSolver, since it keeps most of the 2D structure of the image, we can use block-level access pattern instead of a sequential one to improve cache hit rate. Here is a Python pseudocode to show how it works:

```python
N, M = tgt.shape[:2]
# here is a sequential scan:
parallel for i in range(N):
    parallel for j in range(M):
        if mask[i, j]:
	        tgt_[i, j] = calc(grad[i, j], neighbor(tgt, i, j))
# however, we can use block-level access pattern to improve the cache hit rate:
parallel for i in range(N // grid_x):
    parallel for j in range(M // grid_y):
        # the grid size is (grid_x, grid_y)
        for x in range(i * grid_x, (i + 1) * grid_x):
            for y in range(j * grid_y, (j + 1) * grid_y):
                if mask[x, y]:
	                tgt_[x, y] = calc(grad[x, y], neighbor(tgt, x, y))
```

### Synchronization vs Converge Speed

Since Jacobi Method is an iterative method to solve a matrix equation, there is a trade-off between the quality of solution and the frequency of synchronization.

#### Share-memory Programming Model

The naive approach is to create another matrix to store the solution. Once all pixels' calculation has been finished, the algorithm will refresh the original array with the new value:

```python
for _ in range(n_iter):
    tmp = np.zeros_like(x)
    parallel for i in range(1, N):
        tmp[i] = calc(b[i], neighbor(x, i))
    x = tmp
```

It's quite similar to the "gradient decent" method in machine learning by using all data samples to perform only one step optimization. Interestingly, "stochastic gradient decent"-style Jacobi Method works quite well:

```python
for _ in range(n_iter):
    parallel for i in range(1, N):
        x[i] = calc(b[i], neighbor(x, i))
```

It's because Jacobi Method guarantees its convergence, and w/o such a barrier, the error per pixel will always become smaller. Comparing with the original approach, it also has a faster converge speed.

#### Non Share-memory Programming Model

The above approach works with share-memory programming model such as OpenMP and CUDA. However, for non share-memory programming model such as MPI, the above approach cannot work well. The solution will be discussed in Section [Parallelization Strategy - MPI](mpi).

### Parallelization Strategy

This section will cover the implementation detail with three different backend (OpenMP/MPI/CUDA) and two different solvers (EquSolver/GridSolver).

#### OpenMP

As discussed before, OpenMP [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/equ.cc) first groups the pixels into two folds by `(x + y) % 2`, then parallelizes per-pixel iteration inside a group in each step.

This strategy can utilize the thread-local assessment because the position of four neighborhood become closer. However, it needs to go over the entire array twice because of the split of pixels. In some cases, such as CUDA, this approach introduces an overhead that exceeds the original computational cost. However, in OpenMP, it has a significant runtime improvement.

OpenMP [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/grid.cc) assigns equal amount of blocks for each threads, with size `(grid_x, grid_y)` per block. Each process simply iterates all pixels in each block independently.

We use static assignment for both solvers to minimize the runtime task-assignment overhead, since the workload per pixel/grid is even.

#### MPI

MPI cannot use share-memory program model. We need to reduce the amount of data communicated while maintaining the quality of the solution.

Each MPI process is only responsible for a part of computation, and synchronized with other process per `mpi_sync_interval` steps, denoted as $S$ in this section. When $S$ is too small, the synchronization overhead dominates the computation; when $S$ is too large, each process computes solution independently without global information, therefore the quality of the solution gradually deteriorates.

For MPI [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/equ.cc), it's hard to say which part of the data should be exchanged to other process, since it relabels all pixels at the very beginning of this process. We assign each process with equal amount of equations and use `MPI_Bcast` to force sync all data per $S$ iterations.

MPI [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/grid.cc) uses line partition: process `i` exchanges its first and last line data with process `i-1` and `i+1` separately per $S$ iterations. This strategy has a continuous memory layout to exchange, thus has less overhead comparing with block-level partition.

The workload per pixel is small and fixed. In fact, this type of workload is not suitable for MPI.


#### CUDA

The strategy used on the CUDA backend is quite similar to OpenMP.

CUDA [EquSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/equ.cu) performs equation-level parallelization. It has sequential labeling instead of grouping to two folds as OpenMP. Each block is assigned with equal amount of equations to perform Jacobi Method independently. A thread in a block performs iteration only for a single equation. We also tested the share-memory kernel, but it's much slower than non share-memory version kernel.

For [GridSolver](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/grid.cu), each grid with size `(grid_x, grid_y)` will be in the same block. A thread in a block performs iteration only for a single pixel.

There's no barrier in both solvers' iteration process. The reason has been discussed in Section [Share-memory Programming Model](#share-memory-programming-model).

## Experiments

- RESULTS: How successful were you at achieving your goals? We expect results sections to differ from project to project, but we expect your evaluation to be very thorough (your project evaluation is a great way to demonstrate you understood topics from this course). Here are a few ideas:
  - If your project was optimizing an algorithm, please define how you measured performance. Is it wall-clock time? Speedup? An application specific rate? (e.g., moves per second, images/sec)
  - Please also describe your experimental setup. What were the size of the inputs? How were requests generated?
  - Provide graphs of speedup or execute time. Please precisely define the configurations being compared. Is your baseline single-threaded CPU code? It is an optimized parallel implementation for a single CPU?
  - Recall the importance of problem size. Is it important to report results for different problem sizes for your project? Do different workloads exhibit different execution behavior?
  - **IMPORTANT:** What limited your speedup? Is it a lack of parallelism? (dependencies) Communication or synchronization overhead? Data transfer (memory-bound or bus transfer bound). Poor SIMD utilization due to divergence? As you try and answer these questions, we strongly prefer that you provide data and measurements to support your conclusions. If you are merely speculating, please state this explicitly. Performing a solid analysis of your implementation is a good way to pick up credit even if your optimization efforts did not yield the performance you were hoping for.
  - Deeper analysis: Can you break execution time of your algorithm into a number of distinct components. What percentage of time is spent in each region? Where is there room to improve?
  - Was your choice of machine target sound? (If you chose a GPU, would a CPU have been a better choice? Or vice versa.)

If the GridSolver's parameter `grid_x` and `grid_y` is carefully tuned, it can always perform better than EquSolver with different backend configuration.

![](/_static/images/benchmark.png)

### Benchmark Analysis for 3rd-party Backend

numpy vs numba: hard to say

numpy vs gcc: gcc is much faster

taichi: cpu: equal or better than gcc; gpu: good performance; both of them need a large amount of data to show its advantage

### Benchmark Analysis for Non 3rd-party Backend

OpenMP and MPI can achieve almost the same speed. MPI's converge speed is slower.

CUDA is super fast

### Case study: OpenMP

![](/_static/images/openmp.png)

### Case study: MPI

![](/_static/images/mpi.png)

### Case study: CUDA

![](/_static/images/cuda.png)



## REFERENCE

[1] Pérez, Patrick, Michel Gangnet, and Andrew Blake. "Poisson image editing." *ACM SIGGRAPH 2003 Papers*. 2003. 313-318.

[2] Harris, Charles R., et al. "Array programming with NumPy." *Nature* 585.7825 (2020): 357-362.

[3] Lam, Siu Kwan, Antoine Pitrou, and Stanley Seibert. "Numba: A llvm-based python jit compiler." *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*. 2015.

[4] Hu, Yuanming, et al. "Taichi: a language for high-performance computation on spatially sparse data structures." *ACM Transactions on Graphics (TOG)* 38.6 (2019): 1-16.
