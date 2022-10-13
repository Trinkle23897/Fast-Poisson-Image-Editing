# Poisson Image Editing - A Parallel Implementation

[![PyPI](https://img.shields.io/pypi/v/fpie)](https://pypi.org/project/fpie/)
[![Docs](https://readthedocs.org/projects/fpie/badge/?version=main)](https://fpie.readthedocs.io)
[![Test](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/workflows/Test/badge.svg?branch=main)](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/actions)

> Jiayi Weng (jiayiwen), Zixu Chen (zixuc)

[Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) is a technique that can fuse two images together without producing artifacts. Given a source image and its corresponding mask, as well as a coordination on the target image, the algorithm always yields amazing result.

This project aims to provide a fast poisson image editing algorithm (based on [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method)) that can utilize multi-core CPU or GPU to handle a high-resolution image input.

## Installation

### Linux/macOS/Windows

```bash
$ pip install fpie

# or install from source
$ pip install .
```

### Extensions

| Backend                                        | EquSolver          | GridSolver         | Documentation                                                | Dependency for installation                                  |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NumPy                                          | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#numpy) | `pip install numpy`                                          |
| [Numba](https://github.com/numba/numba)        | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#numba) | `pip install numba`                                          |
| GCC                                            | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#gcc) | cmake, gcc                                                   |
| OpenMP                                         | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#openmp) | cmake, gcc (on macOS you need to change clang to gcc-11)     |
| CUDA                                           | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#cuda) | nvcc                                                         |
| MPI                                            | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#mpi) | `pip install mpi4py` and mpicc (on macOS: `brew install open-mpi`) |
| [Taichi](https://github.com/taichi-dev/taichi) | :heavy_check_mark: | :heavy_check_mark: | [docs](https://fpie.readthedocs.io/en/main/backend.html#taichi) | `pip install taichi`                                         |

After installation, you can use `--check-backend` option to verify:

```bash
$ fpie --check-backend
['numpy', 'numba', 'taichi-cpu', 'taichi-gpu', 'gcc', 'openmp', 'mpi', 'cuda']
```

The above output shows all extensions have successfully installed.

## Usage

We have prepared the test suite to run:

```bash
$ cd tests && ./data.py
```

This script will download 8 tests from GitHub, and create 10 images for benchmarking (5 circle, 5 square). To run:

```bash
$ fpie -s test1_src.jpg -m test1_mask.jpg -t test1_tgt.jpg -o result1.jpg -h1 -150 -w1 -50 -n 5000 -g max
$ fpie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result2.jpg -h1 130 -w1 130 -n 5000 -g src
$ fpie -s test3_src.jpg -m test3_mask.jpg -t test3_tgt.jpg -o result3.jpg -h1 100 -w1 100 -n 5000 -g max
$ fpie -s test4_src.jpg -m test4_mask.jpg -t test4_tgt.jpg -o result4.jpg -h1 100 -w1 100 -n 5000 -g max
$ fpie -s test5_src.jpg -m test5_mask.png -t test5_tgt.jpg -o result5.jpg -h0 -70 -w0 0 -h1 50 -w1 0 -n 5000 -g max
$ fpie -s test6_src.png -m test6_mask.png -t test6_tgt.png -o result6.jpg -h1 50 -w1 0 -n 5000 -g max
$ fpie -s test7_src.jpg -t test7_tgt.jpg -o result7.jpg -h1 50 -w1 30 -n 5000 -g max
$ fpie -s test8_src.jpg -t test8_tgt.jpg -o result8.jpg -h1 90 -w1 90 -n 10000 -g max
```

Here are the results:

| #    | Source image                                                 | Mask image                                                   | Target image                                                 | Result image                                                 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test1_src.jpg) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test1_mask.jpg) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test1_target.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result1.jpg) |
| 2    | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_src.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_mask.png) | ![](https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_target.png) | ![](https://fpie.readthedocs.io/en/main/_images/result2.jpg) |
| 3    | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/1/fg.jpg) | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/1/mask.jpg) | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/1/bg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result3.jpg) |
| 4    | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/2/fg.jpg) | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/2/mask.jpg) | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/2/bg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result4.jpg) |
| 5    | ![](https://github.com/PPPW/poisson-image-editing/raw/master/figs/example1/source1.jpg) | ![](https://github.com/PPPW/poisson-image-editing/raw/master/figs/example1/mask1.png) | ![](https://github.com/PPPW/poisson-image-editing/raw/master/figs/example1/target1.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result5.jpg) |
| 6    | ![](https://github.com/willemmanuel/poisson-image-editing/raw/master/input/1/source.png) | ![](https://github.com/willemmanuel/poisson-image-editing/raw/master/input/1/mask.png) | ![](https://github.com/willemmanuel/poisson-image-editing/raw/master/input/1/target.png) | ![](https://fpie.readthedocs.io/en/main/_images/result6.jpg) |
| 7    | ![](https://github.com/peihaowang/PoissonImageEditing/raw/master/showcases/case0/src.jpg) | /                                                            | ![](https://github.com/peihaowang/PoissonImageEditing/raw/master/showcases/case0/dst.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result7.jpg) |
| 8    | ![](https://github.com/peihaowang/PoissonImageEditing/raw/master/showcases/case3/src.jpg) | /                                                            | ![](https://github.com/peihaowang/PoissonImageEditing/raw/master/showcases/case3/dst.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result8.jpg) |

### GUI

```bash
$ fpie-gui -s test3_src.jpg -t test3_tgt.jpg -o result.jpg -b cuda -n 10000
```

![](https://fpie.readthedocs.io/en/main/_images/gui.png)

We provide a simple GUI for real-time seamless cloning. You need to use your mouse to draw a rectangle on top of the source image, and click a point in target image. After that the result will automatically be generated. In the end, you can press ESC to terminate the program.

### Backend and Solver

We have provided 7 backends. Each backend has two solvers: EquSolver and GridSolver. You can find the difference between these two solvers in the next section.

For different backend usage, please check out the related documentation [here](https://fpie.readthedocs.io/en/main/backend.html).

For other usage, please run `fpie -h` or `fpie-gui -h` to see the hint.

## Benchmark Result

![](https://fpie.readthedocs.io/en/main/_images/benchmark.png)

See [benchmark result](https://fpie.readthedocs.io/en/main/benchmark.html) and [report](https://fpie.readthedocs.io/en/main/report.html#result-and-analysis).

## Algorithm Detail

The general idea is to keep most of gradient in source image, while matching the boundary of source image and target image pixels.

The gradient is computed by

![](https://latex.codecogs.com/svg.latex?\nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1))

After calculating the gradient in source image, the algorithm tries to solve the following problem: given the gradient and the boundary value, calculate the approximate solution that meets the requirement, i.e., to keep target image's gradient as similar as the source image. It can be formulated as ![](https://latex.codecogs.com/svg.latex?{(4-A)\vec{x}=\vec{b}}), where ![](https://latex.codecogs.com/svg.latex?{A\in\mathbb{R}^{N\times%20N},\vec{x}\in\mathbb{R}^N,\vec{b}\in\mathbb{R}^N}), N is the number of pixels in the mask, A is a giant sparse matrix because each line of A only contains at most 4 non-zero value (neighborhood), b is the gradient from source image, and x is the result value.

N is always a large number, i.e., greater than 50k, so the Gauss-Jordan Elimination cannot be directly applied here because of the high time complexity O(N^3). People use [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) to solve the problem. Thanks to the sparsity of matrix A, the overall time complexity is O(MN) where M is the number of iteration performed by poisson image editing.

This project parallelizes Jacobi method to speed up the computation. To our best knowledge, there's no public project on GitHub that implements poisson image editing with either OpenMP, or MPI, or CUDA. All of them can only handle a small size image workload.

### EquSolver vs GridSolver

Usage: `--method {equ,grid}`

EquSolver directly constructs the equations ![](https://latex.codecogs.com/svg.latex?(4-A)\vec{x}=\vec{b}) by re-labeling the pixel, and use Jacobi method to get the solution via ![](https://latex.codecogs.com/svg.latex?{\vec{x}'=(A\vec{x}+\vec{b})/4).

GridSolver uses the same Jacobi iteration, however, it keeps the 2D structure of the original image instead of re-labeling the pixel in the mask. It may take some advantage when the mask region covers all of the image, because in this case GridSolver can save 4 read instructions by directly calculating the neighborhood's coordinate.

If the GridSolver's parameter is carefully tuned (`--grid-x` and `--grid-y`), it can always perform better than EquSolver with different backend configuration.

### Gradient for PIE

Usage: `-g {max,src,avg}`

The [PIE paper](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) states some variant of gradient calculation such as Equ. 12: using the maximum gradient to perform "mixed seamless cloning". We also provide such an option in our program:

- `src`: only use the gradient from source image
- `avg`: use the average gradient of source image and target image
- `max`: use the max gradient of source and target image

The following example shows the difference between these three methods:

| #    | target image                                                 | --gradient=src                                             | --gradient=avg                                             | --gradient=max                                               |
| ---- | ------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| 3    | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/1/bg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/3gsrc.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/3gavg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result3.jpg) |
| 4    | ![](https://github.com/cheind/poisson-image-editing/raw/master/etc/images/2/bg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/4gsrc.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/4gavg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result4.jpg) |
| 8    | ![](https://github.com/peihaowang/PoissonImageEditing/raw/master/showcases/case3/dst.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/8gsrc.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/8gavg.jpg) | ![](https://fpie.readthedocs.io/en/main/_images/result8.jpg) |

## Miscellaneous (for 15-618 course project)

[Project proposal and milestone](docs/misc.md)

[Final report](https://fpie.readthedocs.io/en/main/report.html) and [5min video](https://trinkle23897.github.io/images/fpie.mp4)
