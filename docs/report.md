# Final Report

## Summary

We implemented a parallelized Poisson image editor with Jacobi method. It can compute results using seven extensions: NumPy, [Numba](https://github.com/numba/numba), [Taichi](https://github.com/taichi-dev/taichi), single-thread c++, OpenMP, MPI, and CUDA. In terms of performance, we have a detailed benchmarking result that the CUDA backend can achieve 31 to 42 times faster on GHC machines compared to the single-threaded c++ backend. In terms of user-experience, we have a simple GUI to demonstrate the results interactively, released a standard [PyPI package](https://pypi.org/project/fpie/), and provide [a website](https://fpie.readthedocs.io/) for project documentation.

| Source image                                                 | Mask image                                                   | Target image                                                 | Result image                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_src.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_mask.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_target.png) | ![](/_static/images/result2.jpg) |

## Background

[Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) is a technique that can blend two images together without artifacts. Given a source image and its corresponding mask, and a coordination on target image, this algorithm can always generate amazing result. The general idea is to keep most of gradient in source image, while matching the boundary of source image and target image pixels.

The gradient per pixel is computed by

$$\nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1)$$

After computing the gradient in source image, the algorithm tries to solve the following problem: given the gradient and the boundary value, calculate the approximate solution that meets the requirement, i.e., to keep target image's gradient as similar as the source image.

This process can be formulated as $(4-A)\vec{x}=\vec{b}$, where $A\in\mathbb{R}^{N\times N}$, $\vec{x}\in\mathbb{R}^N$, $\vec{b}\in\mathbb{R}^N$, $N$ is the number of pixels in the mask, $A$ is a giant sparse matrix because each line of A only contains at most 4 non-zero value (neighborhood), $\vec{b}$ is the gradient from source image, and $\vec{x}$ is the result value.

$N$ is always a large number, i.e., greater than 50k, so the Gauss-Jordan Elimination cannot be directly applied here because of the high time complexity $O(N^3)$. People use [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) to solve the problem. Thanks to the sparsity of matrix $A$, the overall time complexity is $O(MN)$ where $M$ is the number of iteration performed by Poisson image editing. The iterative equation is $\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4$.

This project parallelizes Jacobi method to speed up the computation. To our best knowledge, there's no public project on GitHub that implements Poisson image editing with either OpenMP, or MPI, or CUDA. All of them can only handle a small size image workload ([link](https://github.com/PPPW/poisson-image-editing/issues/1)).

## Method

### Solver

We implemented two different solvers: EquSolver and GridSolver.

EquSolver directly constructs the equations $(4-A)\vec{x}=\vec{b}$ and use Jacobi method to get the solution via $\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4$.

GridSolver uses the same Jacobi iteration, however, it keeps the 2D structure of the original image instead of re-labeling the pixel in the mask. It may take some advantage when the mask region covers all of the image, because in this case GridSolver can save 4 read instructions by directly calculating the neighborhood's coordinate.

### Parallelization Strategy

#### OpenMP

#### MPI

#### CUDA



## Experiments

If the GridSolver's parameter `grid_x` and `grid_y` is carefully tuned, it can always perform better than EquSolver with different backend configuration.

![](/_static/images/benchmark.png)

### Benchmark Analysis for 3rd-party Backend

### Benchmark Analysis for Non 3rd-party Backend

### Case study: OpenMP

![](/_static/images/openmp.png)

### Case study: MPI

![](/_static/images/mpi.png)

### Case study: CUDA

![](/_static/images/cuda.png)



## REFERENCE

[1] Pérez, Patrick, Michel Gangnet, and Andrew Blake. "Poisson image editing." *ACM SIGGRAPH 2003 Papers*. 2003. 313-318.

