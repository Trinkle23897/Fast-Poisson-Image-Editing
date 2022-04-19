Final Report
============

Summary
-------

We implemented a parallelized Poisson image editor with Jacobi method. It can compute results using
seven extensions: NumPy, `Numba <https://github.com/numba/numba>`__,
`Taichi <https://github.com/taichi-dev/taichi>`__, single-thread c++, OpenMP, MPI, and CUDA. In
terms of performance, we have a detailed benchmarking result that the CUDA backend can achieve 31 to
42 times faster on GHC machines compared to the single-threaded c++ implementation. In terms of
user-experience, we have a simple GUI to demonstrate the results interactively, released a standard
`PyPI package <https://pypi.org/project/fpie/>`__, and provide `a
website <https://fpie.readthedocs.io/>`__ for project documentation.

============ ========== ============ ============
Source image Mask image Target image Result image
============ ========== ============ ============
|image0|     |image1|   |image2|     |image3|
============ ========== ============ ============

Background
----------

Poisson Image Editing
~~~~~~~~~~~~~~~~~~~~~

`Poisson Image Editing <https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf>`__ is a technique
that can blend two images together without artifacts. Given a source image and its corresponding
mask, and a coordination on target image, this algorithm can always generate amazing result. The
general idea is to keep most of gradient in source image, while matching the boundary of source
image and target image pixels.

The gradient per pixel is computed by

.. math:: \nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1)

After computing the gradient in source image, the algorithm tries to solve the following problem:
given the gradient and the boundary value, calculate the approximate solution that meets the
requirement, i.e., to keep target image’s gradient as similar as the source image.

This process can be formulated as :math:`(4-A)\vec{x}=\vec{b}`, where
:math:`A\in\mathbb{R}^{N\times N}`, :math:`\vec{x}\in\mathbb{R}^N`, :math:`\vec{b}\in\mathbb{R}^N`,
:math:`N` is the number of pixels in the mask, :math:`A` is a giant sparse matrix because each line
of A only contains at most 4 non-zero value (neighborhood), :math:`\vec{b}` is the gradient from
source image, and :math:`\vec{x}` is the result value.

:math:`N` is always a large number, i.e., greater than 50k, so the Gauss-Jordan Elimination cannot
be directly applied here because of the high time complexity :math:`O(N^3)`. People use `Jacobi
Method <https://en.wikipedia.org/wiki/Jacobi_method>`__ to solve the problem. Thanks to the sparsity
of matrix :math:`A`, the overall time complexity is :math:`O(MN)` where :math:`M` is the number of
iteration performed by Poisson image editing. The iterative equation is
:math:`\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4`.

This project parallelizes Jacobi method to speed up the computation. To our best knowledge, there’s
no public project on GitHub that implements Poisson image editing with either OpenMP, or MPI, or
CUDA. All of them can only handle a small size image workload
(`link <https://github.com/PPPW/poisson-image-editing/issues/1>`__).

PIE Solver
~~~~~~~~~~

We implemented two different solvers: EquSolver and GridSolver.

EquSolver directly constructs the equations :math:`(4-A)\vec{x}=\vec{b}` by re-labeling the pixel,
and use Jacobi method to get the solution via :math:`\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4`.

.. code:: python

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

GridSolver uses the same Jacobi iteration, however, it keeps the 2D structure of the original image
instead of re-labeling the pixel in the mask. It may take some advantage when the mask region covers
all of the image, because in this case GridSolver can save 4 read instructions by directly
calculating the neighborhood’s coordinate. Meanwhile, it has a better locality of fetching required
data per iteration if we properly setup the access pattern (will be discussed in Section `Access
Pattern <#access-pattern>`__).

.. code:: python

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

The bottleneck for both solvers is the for-loop and can be easily parallelized. The implementation
detail and parallelization strategies will be discussed in Section `Parallelization
Strategy <#parallelization-strategy>`__.

Method
------

Language and Hardware Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start to build PIE with the help of `pybind11 <https://github.com/pybind/pybind11>`__ because our
goal is to benchmark multiple parallelization approaches, including hand-written CUDA code and other
3rd-party libraries such as NumPy.

One of our project goal is to let the algorithm run on any \*nix machine and can have a real-time
interactive result demonstration. For this reason, we don’t choose super computing cluster as the
hardware setup. Instead, we choose GHC machine to develop and measure the performance, which has 8x
i7-9700 cores and an Nvidia RTX 2080Ti.

Access Pattern
~~~~~~~~~~~~~~

For EquSolver, we can re-organize the pixel order to achieve a better locality when performing
parallel operations. Specifically, we can group all pixels into two folds by ``(x + y) % 2``. Here
is a small example:

::

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

By doing so, every pixel’s 4 neighbors are closer with each other. The ideal access pattern is to
separately iterate these two groups, i.e.,

.. code:: python

   for _ in range(n_iter):
       parallel for i in range(1, p):
           # i < p, neighbor >= p
           x_[i] = calc(b[i], neighbor(x, i))

       parallel for i in range(p, N):
           # i >= p, neighbor < p
           x[i] = calc(b[i], neighbor(x_, i))

Unfortunately, we only observe a clear advantage with OpenMP EquSolver. For other backend, the
sequential id assignment is much better than re-order. The related discussion is in Section
`Parallelization Strategy - OpenMP <#openmp>`__.

For GridSolver, since it keeps most of the 2D structure of the image, we can use block-level access
pattern instead of a sequential one to improve cache hit rate. Here is a Python pseudocode to show
how it works:

.. code:: python

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

Synchronization vs Converge Speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Jacobi Method is an iterative method to solve a matrix equation, there is a trade-off between
the quality of solution and the frequency of synchronization.

Share-memory Programming Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The naive approach is to create another matrix to store the solution. Once all pixels’ calculation
has been finished, the algorithm will refresh the original array with the new value:

.. code:: python

   for _ in range(n_iter):
       tmp = np.zeros_like(x)
       parallel for i in range(1, N):
           tmp[i] = calc(b[i], neighbor(x, i))
       x = tmp

It’s quite similar to the “gradient decent” method in machine learning by using all data samples to
perform only one step optimization. Interestingly, “stochastic gradient decent”-style Jacobi Method
works quite well:

.. code:: python

   for _ in range(n_iter):
       parallel for i in range(1, N):
           x[i] = calc(b[i], neighbor(x, i))

It’s because Jacobi Method guarantees its convergence, and w/o such a barrier, the error per pixel
will always become smaller. Comparing with the original approach, it also has a faster converge
speed.

Non Share-memory Programming Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above approach works with share-memory programming model such as OpenMP and CUDA. However, for
non share-memory programming model such as MPI, the above approach cannot work well. The solution
will be discussed in Section `Parallelization Strategy - MPI <mpi>`__.

Parallelization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

This section will cover the implementation detail with three different backend (OpenMP/MPI/CUDA) and
two different solvers (EquSolver/GridSolver).

OpenMP
^^^^^^

As discussed before, OpenMP
`EquSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/equ.cc>`__
first groups the pixels into two folds by ``(x + y) % 2``, then parallelizes per-pixel iteration
inside a group in each step.

This strategy can utilize the thread-local assessment because the position of four neighborhood
become closer. However, it needs to go over the entire array twice because of the split of pixels.
In some cases, such as CUDA, this approach introduces an overhead that exceeds the original
computational cost. However, in OpenMP, it has a significant runtime improvement.

OpenMP
`GridSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/openmp/grid.cc>`__
assigns equal amount of blocks for each threads, with size ``(grid_x, grid_y)`` per block. Each
process simply iterates all pixels in each block independently.

We use static assignment for both solvers to minimize the runtime task-assignment overhead, since
the workload per pixel/grid is even.

MPI
^^^

MPI cannot use share-memory program model. We need to reduce the amount of data communicated while
maintaining the quality of the solution.

Each MPI process is only responsible for a part of computation, and synchronized with other process
per ``mpi_sync_interval`` steps, denoted as :math:`S` in this section. When :math:`S` is too small,
the synchronization overhead dominates the computation; when :math:`S` is too large, each process
computes solution independently without global information, therefore the quality of the solution
gradually deteriorates.

For MPI
`EquSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/equ.cc>`__,
it’s hard to say which part of the data should be exchanged to other process, since it relabels all
pixels at the very beginning of this process. We assign each process with equal amount of equations
and use ``MPI_Bcast`` to force sync all data per :math:`S` iterations.

MPI
`GridSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/mpi/grid.cc>`__
uses line partition: process ``i`` exchanges its first and last line data with process ``i-1`` and
``i+1`` separately per :math:`S` iterations. This strategy has a continuous memory layout to
exchange, thus has less overhead comparing with block-level partition.

The workload per pixel is small and fixed. In fact, this type of workload is not suitable for MPI.

CUDA
^^^^

The strategy used on the CUDA backend is quite similar to OpenMP.

CUDA
`EquSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/equ.cu>`__
performs equation-level parallelization. It has sequential labeling instead of grouping to two folds
as OpenMP. Each block is assigned with equal amount of equations to perform Jacobi Method
independently. A thread in a block performs iteration only for a single equation. We also tested the
share-memory kernel, but it’s much slower than non share-memory version kernel.

For
`GridSolver <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/fpie/core/cuda/grid.cu>`__,
each grid with size ``(grid_x, grid_y)`` will be in the same block. A thread in a block performs
iteration only for a single pixel.

There’s no barrier in both solvers’ iteration process. The reason has been discussed in Section
`Share-memory Programming Model <#share-memory-programming-model>`__.

Experiments
-----------

Experiment Setting
~~~~~~~~~~~~~~~~~~

Hardware and Software
^^^^^^^^^^^^^^^^^^^^^

We use GHC83 to run all of the following experiments. Here is the hardware and software
configuration:

-  OS: Red Hat Enterprise Linux Workstation 7.9 (Maipo)
-  CPU: 8x Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
-  GPU: GeForce RTX 2080 8G
-  Python: 3.6.8
-  Python package version:

   -  numpy==1.19.5
   -  opencv-python==4.5.5.64
   -  mpi4py==3.1.3
   -  numba==0.53.1
   -  taichi==1.0.0

Data
^^^^

We generate 10 images for benchmarking performance, 5 square and 5 circle. The script is
`tests/data.py <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/blob/main/tests/data.py>`__.
You can find the detail information in this table:

======== ========= ======== ================= =========
ID       Size      # pixels # unmasked pixels Image
======== ========= ======== ================= =========
square6  66x66     4356     4356              |image4|
square7  130x130   16900    16900             |image5|
square8  258x258   66564    66564             |image6|
square9  514x514   264196   264196            |image7|
square10 1026x1026 1052676  1052676           |image8|
circle6  74x74     5476     4291              |image9|
circle7  146x146   21316    16727             |image10|
circle8  290x290   84100    66043             |image11|
circle9  579x579   335241   262341            |image12|
circle10 1157x1157 1338649  1049489           |image13|
======== ========= ======== ================= =========

We try to keep the number of unmasked pixels of circleX and squareX to be the same level. For
EquSolver there’s no difference, but for GridSolver it cannot be ignored, since it needs to process
all pixels no matter it is masked.

Metric
^^^^^^

We measure the performance by “Time per Operation” (TpO for short) and “Cache Miss per Operation”
(CMpO for short). TpO is derived by ``total time / total number of iteration / number of pixel``.
The smaller the TpO, the more efficient the parallel algorithm is. CMpO is derived by
``total cache miss / total number of iteration / number of pixel``.

Result and Analysis
~~~~~~~~~~~~~~~~~~~

We use all seven backend to run benchmark experiments. ``GCC`` (single-thread C++ implementation) is
the baseline. The detail of the following experiment (command and table) can be found at
`Benchmark <./benchmark.html>`__ page. For simplicity, we only demonstrate the plot in the following
sections. All plots are with log-log scale.

|image14|

-  Provide graphs of speedup or execute time. Please precisely define the configurations being
   compared. Is your baseline single-threaded CPU code? It is an optimized parallel implementation
   for a single CPU?
-  Recall the importance of problem size. Is it important to report results for different problem
   sizes for your project? Do different workloads exhibit different execution behavior?
-  **IMPORTANT:** What limited your speedup? Is it a lack of parallelism? (dependencies)
   Communication or synchronization overhead? Data transfer (memory-bound or bus transfer bound).
   Poor SIMD utilization due to divergence? As you try and answer these questions, we strongly
   prefer that you provide data and measurements to support your conclusions. If you are merely
   speculating, please state this explicitly. Performing a solid analysis of your implementation is
   a good way to pick up credit even if your optimization efforts did not yield the performance you
   were hoping for.
-  Deeper analysis: Can you break execution time of your algorithm into a number of distinct
   components. What percentage of time is spent in each region? Where is there room to improve?
-  Was your choice of machine target sound? (If you chose a GPU, would a CPU have been a better
   choice? Or vice versa.)

EquSolver vs GridSolver
^^^^^^^^^^^^^^^^^^^^^^^

If the GridSolver’s parameter ``grid_x`` and ``grid_y`` is carefully tuned, most of the time it can
perform better than EquSolver with hand-written backend configuration (OpenMP/MPI/CUDA). The
analysis will be in the following sections. However, it’s hard to say which one is better by using
other 3rd-party backend. This may due to the internal design of these libraries.

Analysis for 3rd-party Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy
'''''

NumPy is 10~11x slower than GCC with EquSolver, and 8~9x slower than GCC with GridSolver. This
result indicates the overhead in NumPy solver is not negligible. Each iteration it needs to transfer
data between C and Python repeatedly, and create some temporary array to calculate the result. It
cannot utilize the memory layout even though we have already use vectorized operation for all
computations.

Numba
'''''

Numba is a just-in-time compiler for numerical functions in Python. For EquSolver, Numba is about
twice faster than NumPy; however, for GridSolver, Numba is about twice slower than NumPy. This
result shows Numba cannot provide a general speedup for any NumPy operations, not to mention it is
still slower than GCC.

Taichi
''''''

Taichi is an open-source, imperative, parallel programming language for high-performance numerical
computation. If we use Taichi with a small size input image, it won’t get too much benefit. However,
when increasing the problem size to a very large scale, the advantage becomes much clear. We think
it is because of pre-processing step in Taichi.

With CPU backend, EquSolver is faster than GCC, while GridSolver’s performance is almost equal to
GCC. This shows the access pattern largely affects the actual performance.

With GPU backend, though the TpO is twice slower than CUDA with extremely large-scale input, it is
still faster than any other backend. We are quite interested in other 3rd-party GPU solution’s
performance, and leave it as future work.

Analysis for Non 3rd-party Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TL; DR: OpenMP and MPI can achieve almost the same speed, but MPI’s converge speed is slower because
of the synchronization trade-off. CUDA is the fastest in all conditions.

.. _openmp-1:

OpenMP
''''''

EquSolver is 8~9x faster than GCC; GridSolver is 6~7x faster than GCC. However, there is a huge
performance drop when the problem size is greater than 1M for both two solvers. The threshold is
300k ~ 400k for EquSolver and 500k ~ 600k for GridSolver. We suspect that is because of cache-miss,
confirmed by the following numerical result:

.. raw:: html

   <!--openmp-->

========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======
OpenMP     # of pixels 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000
========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======
EquSolver  Time (s)    0.1912 0.3728 0.6033 1.073  2.0081 3.4242 4.1646 5.6254 6.2875 7.6159
EquSolver  TpO (ns)    0.3824 0.3728 0.4022 0.5365 0.8032 1.1414 1.1899 1.4063 1.3972 1.5232
EquSolver  CMpO        0.0341 0.0201 0.1104 0.3713 0.5799 0.6757 0.7356 0.8083 0.8639 0.9232
GridSolver Time (s)    0.2870 0.5722 0.8356 1.1321 1.4391 2.2886 3.0738 4.1967 5.5097 6.0635
GridSolver TpO (ns)    0.5740 0.5722 0.5571 0.5661 0.5756 0.7629 0.8782 1.0492 1.2244 1.2127
GridSolver CMpO        0.0330 0.0174 0.0148 0.0522 0.1739 0.3346 0.3952 0.4495 0.5132 0.5394
========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======

.. raw:: html

   <!--openmp-->

|image15|

We also investigated the impact of the number of threads on the performance of the OpenMP backend.
There is a linear speedup when the aforementioned cache-miss problem does not occur; when the
cache-miss problem is encountered, its performance quickly saturates, especially for EquSolver. We
think the reason behind is GridSolver can better use the locality comparing with EquSolver, since it
has no re-labeling pixel process and keep all of the 2D information.

|image16|

.. _mpi-1:

MPI
'''

EquSolver and GridSolver is 6~7x faster than GCC. Like OpenMP, there is a huge performance drop. The
threshold is 300k ~ 400k for EquSolver and 400k ~ 500k for GridSolver. Fortunately, the following
table and plot confirms our assumption of cache-miss:

.. raw:: html

   <!--mpi-->

========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======
MPI        # of pixels 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000
========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======
EquSolver  Time (s)    0.2696 0.6584 0.9549 1.6435 2.6920 3.6933 4.7265 5.7762 6.8305 7.7894
EquSolver  TpO (ns)    0.5392 0.6584 0.6366 0.8218 1.0768 1.2311 1.3504 1.4441 1.5179 1.5579
EquSolver  CMpO        0.5090 0.2743 0.2998 0.4646 0.5995 0.7006 0.7525 0.7951 0.8204 0.8391
GridSolver Time (s)    0.2994 0.5948 0.9088 1.3075 1.6024 2.1239 2.8969 3.7388 4.4776 5.3026
GridSolver TpO (ns)    0.5988 0.5948 0.6059 0.6538 0.6410 0.7080 0.8277 0.9347 0.9950 1.0605
GridSolver CMpO        0.5054 0.2570 0.1876 0.2008 0.2991 0.3783 0.4415 0.4866 0.5131 0.5459
========== =========== ====== ====== ====== ====== ====== ====== ====== ====== ====== =======

.. raw:: html

   <!--mpi-->

|image17|

A similar phenomenon occurs on the MPI backend when the number of processes is changed:

|image18|

.. _cuda-1:

CUDA
''''

EquSolver is 27~44x faster than GCC; GridSolver is 38~42x faster than GCC. The performance is
consistent over different input size.

We studied with the impact of different block size on CUDA EquSolver. For better demonstration, we
don’t use GridSolver because it needs to set two parameters ``grid_x`` and ``grid_y``. By increasing
the block size, the performance gets better first, reaches the peak, and drops down finally. The
best configuration is block size = 256.

When the block size is too small, it will use more grids to compute, thus the cross-grid
communication overhead will be increased. When the block size is too large, though it uses less
amount of grids, the cache-invalidation issue dominates – since we don’t use share memory inside
this CUDA kernel and there’s no barrier when calling this kernel, we guess the program frequently
read values that cannot be cached, and also frequently write values to invalidate cache.

|image19|

Contribution
------------

The contribution for each group member is on
`GitHub <https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/graphs/contributors>`__.

REFERENCE
---------

[1] Pérez, Patrick, Michel Gangnet, and Andrew Blake. “Poisson image editing.” *ACM SIGGRAPH 2003
Papers*. 2003. 313-318.

[2] Harris, Charles R., et al. “Array programming with NumPy.” *Nature* 585.7825 (2020): 357-362.

[3] Lam, Siu Kwan, Antoine Pitrou, and Stanley Seibert. “Numba: A llvm-based python jit compiler.”
*Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*. 2015.

[4] Hu, Yuanming, et al. “Taichi: a language for high-performance computation on spatially sparse
data structures.” *ACM Transactions on Graphics (TOG)* 38.6 (2019): 1-16.

.. |image0| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_src.png
.. |image1| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_mask.png
.. |image2| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_target.png
.. |image3| image:: /_static/images/result2.jpg
.. |image4| image:: /_static/images/square6.png
.. |image5| image:: /_static/images/square7.png
.. |image6| image:: /_static/images/square8.png
.. |image7| image:: /_static/images/square9.png
.. |image8| image:: /_static/images/square10.png
.. |image9| image:: /_static/images/circle6.png
.. |image10| image:: /_static/images/circle7.png
.. |image11| image:: /_static/images/circle8.png
.. |image12| image:: /_static/images/circle9.png
.. |image13| image:: /_static/images/circle10.png
.. |image14| image:: /_static/images/benchmark.png
.. |image15| image:: /_static/images/openmp0.png
.. |image16| image:: /_static/images/openmp.png
.. |image17| image:: /_static/images/mpi0.png
.. |image18| image:: /_static/images/mpi.png
.. |image19| image:: /_static/images/cuda.png
