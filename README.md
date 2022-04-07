# Poisson Image Editing - a parallel implementation

> Jiayi Weng (jiayiwen), Zixu Chen (zixuc)

This project aims to provide a fast poisson image editing algorithm that can utilize multi-core CPU or GPU to handle a high-resolution image input.

## Installation & Usage

On a Linux machine:

```bash
$ git submodule update --init --recursive
$ pip install .
```

Run test suite:

```bash
$ cd tests
$ ./data.py
$ pie -s test1_src.jpg -m test1_mask.jpg -t test1_tgt.jpg -o result1.png -h0 0 -w0 0 -h1 -150 -w1 -50 -n 5000 -p 1000 -g mix -b openmp
$ pie -s test2_src.png -m test2_mask.png -t test2_tgt.png -o result2.png -h0 0 -w0 0 -h1 130 -w1 130 -n 5000 -p 1000 -g src -b openmp
$ pie -s test3_src.jpg -m test3_mask.jpg -t test3_tgt.jpg -o result3.png -h0 0 -w0 0 -h1 100 -w1 100 -n 5000 -p 0 -g mix -b openmp
$ pie -s test4_src.jpg -m test4_mask.jpg -t test4_tgt.jpg -o result4.png -h0 0 -w0 0 -h1 100 -w1 100 -n 5000 -p 0 -g mix -b openmp
$ pie -s test5_src.jpg -m test5_mask.png -t test5_tgt.jpg -o result5.png -h0 -70 -w0 0 -h1 50 -w1 0 -n 25000 -p 0 -g mix -b openmp
$ pie -s test6_src.png -m test6_mask.png -t test6_tgt.png -o result6.png -h0 0 -w0 0 -h1 50 -w1 0 -n 25000 -p 0 -g mix -b openmp
$ pie -s test7_src.jpg -t test7_tgt.jpg -o result.png -h0 0 -w0 0 -h1 50 -w1 30 -n 25000 -p 0 -g mix -b openmp
$ pie -s test8_src.jpg -t test8_tgt.jpg -o result.png -h0 0 -w0 0 -h1 90 -w1 90 -n 25000 -p 0 -g mix -b openmp
```

## Background

[Poisson image editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) is a technique that can blend two images together without artifacts. Given a source image and its corresponding mask, and a coordination on target image, this algorithm can always generate amazing result:

Source image:

![](https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/test2_src.png?raw=true)

Mask on source image:

![](https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/test2_mask.png?raw=true)

Target image:

![test2_target.png](https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/test2_target.png?raw=true)

Result:

![test2_result.png](https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/test2_result.png?raw=true)

## Algorithm detail

The general idea is to keep most of gradient in source image, while matching the boundary of source image and target image pixels.

The gradient is computed by
$$
\nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1)
$$
After computing the gradient in source image, the algorithm tries to solve the following problem: given the gradient and the boundary value, calculate the approximate solution that meets the requirement, i.e., to keep target image's gradient as similar as the source image. It can be formulated as
$$
A\vec{x}=\vec{b}
$$
where $A\in \mathbb{R}^{N\times N}$, $\vec{x}\in \mathbb{R}^N$, $\vec{b}\in \mathbb{R}^N$, where $N$ is the number of pixels in the mask. Therefore, $A$ is a giant sparse matrix because each line of A only contains at most 5 non-zero value.

$N$ is always a large number, i.e., greater than 500k, so the Gauss-Jordan Elimination cannot be directly applied here because of the high time complexity $O(N^3)$. People always use [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) to solve the problem. Thanks to the sparsity of matrix $A$, the overall time complexity is $O(MN)$ where $M$ is the number of iteration performed by poisson image editing.

In this project, we are going to parallelize Jacobi method to speed up the computation. To our best knowledge, there's no public project on GitHub that implements poisson image editing with either OpenMP, or MPI, or CUDA. All of them can only handle a small size image workload.

## Implement Method

- [x] NumPy
- [x] OpenMP
- [ ] MPI
- [ ] CUDA

## Miscellaneous (for 15-618 course project)

Challenge: How to implement a fully-parallelized Jacobi Iteration to support a real-time image fusion?

- Workload/constrains: similar to the 2d-grid example demonstrated in class.

Resources:

- Codebase: https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/image_fusion.cpp, written by Jiayi Weng, one of our group members
- Computation Resources: GHC (CPU, GPU - 2080), PSC (CPU, GPU), laptop (CPU, GPU - 1060)

Goals:

- [x] 75%: implement one parallel version of PIE
- [ ] 100%: benchmark the algorithm with OpenMP/MPI/CUDA implementation
- [ ] 125%: include a interactive python frontend that can demonstrate the result in a user-friendly style. Real-time computation on a single GTX-1060 Nvidia GPU.

Platform choice:

- OS: Linux, Ubuntu machine
- Language: C++/CUDA for core development, Python for interactive frontend

Schedule:

- 3.28 - 4.9: implement parallel version of PIE (OpenMP, MPI, CUDA)
- 4.11 - 4.22: benchmarking, optimizing, write Python interactive frontend
- 4.25 - 5.5: write report
