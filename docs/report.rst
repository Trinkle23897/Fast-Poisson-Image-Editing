Final Report
============

Summary
-------

We implemented a parallelized Poisson image editor with Jacobi method.
It can compute results using seven extensions: NumPy,
`Numba <https://github.com/numba/numba>`__,
`Taichi <https://github.com/taichi-dev/taichi>`__, single-thread c++,
OpenMP, MPI, and CUDA. In terms of performance, we have a detailed
benchmarking result that the CUDA backend can achieve 31 to 42 times
faster on GHC machines compared to the single-threaded c++
implementation. In terms of user-experience, we have a simple GUI to
demonstrate the results interactively, released a standard `PyPI
package <https://pypi.org/project/fpie/>`__, and provide `a
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

`Poisson Image
Editing <https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf>`__ is
a technique that can blend two images together without artifacts. Given
a source image and its corresponding mask, and a coordination on target
image, this algorithm can always generate amazing result. The general
idea is to keep most of gradient in source image, while matching the
boundary of source image and target image pixels.

The gradient per pixel is computed by

.. math:: \nabla(x,y)=4I(x,y)-I(x-1,y)-I(x,y-1)-I(x+1,y)-I(x,y+1)

After computing the gradient in source image, the algorithm tries to
solve the following problem: given the gradient and the boundary value,
calculate the approximate solution that meets the requirement, i.e., to
keep target image’s gradient as similar as the source image.

This process can be formulated as :math:`(4-A)\vec{x}=\vec{b}`, where
:math:`A\in\mathbb{R}^{N\times N}`, :math:`\vec{x}\in\mathbb{R}^N`,
:math:`\vec{b}\in\mathbb{R}^N`, :math:`N` is the number of pixels in the
mask, :math:`A` is a giant sparse matrix because each line of A only
contains at most 4 non-zero value (neighborhood), :math:`\vec{b}` is the
gradient from source image, and :math:`\vec{x}` is the result value.

:math:`N` is always a large number, i.e., greater than 50k, so the
Gauss-Jordan Elimination cannot be directly applied here because of the
high time complexity :math:`O(N^3)`. People use `Jacobi
Method <https://en.wikipedia.org/wiki/Jacobi_method>`__ to solve the
problem. Thanks to the sparsity of matrix :math:`A`, the overall time
complexity is :math:`O(MN)` where :math:`M` is the number of iteration
performed by Poisson image editing. The iterative equation is
:math:`\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4`.

This project parallelizes Jacobi method to speed up the computation. To
our best knowledge, there’s no public project on GitHub that implements
Poisson image editing with either OpenMP, or MPI, or CUDA. All of them
can only handle a small size image workload
(`link <https://github.com/PPPW/poisson-image-editing/issues/1>`__).

PIE Solver
~~~~~~~~~~

We implemented two different solvers: EquSolver and GridSolver.

EquSolver directly constructs the equations :math:`(4-A)\vec{x}=\vec{b}`
by re-labeling the pixel, and use Jacobi method to get the solution via
:math:`\vec{x}' \leftarrow (A\vec{x}+\vec{b})/4`.

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

GridSolver uses the same Jacobi iteration, however, it keeps the 2D
structure of the original image instead of re-labeling the pixel in the
mask. It may take some advantage when the mask region covers all of the
image, because in this case GridSolver can save 4 read instructions by
directly calculating the neighborhood’s coordinate. Meanwhile, it has a
better locality of fetching related data.

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

The bottleneck for both solvers is the for-loop and can be easily
parallelized. The implementation detail and parallelization strategies
will be discussed in the next section.

Method
------

-  APPROACH: Tell us how your implementation works. Your description
   should be sufficiently detailed to provide the course staff a basic
   understanding of your approach. Again, it might be very useful to
   include a figure here illustrating components of the system and/or
   their mapping to parallel hardware.

   -  Break down the workload. Where are the dependencies in the
      program? How much parallelism is there? Is it data-parallel? Where
      is the locality? Is it amenable to SIMD execution?
   -  Describe the technologies used. What language/APIs? What machines
      did you target?
   -  Describe how you mapped the problem to your target parallel
      machine(s). IMPORTANT: How do the data structures and operations
      you described in part 2 map to machine concepts like cores and
      threads. (or warps, thread blocks, gangs, etc.)
   -  Did you change the original serial algorithm to enable better
      mapping to a parallel machine?
   -  If your project involved many iterations of optimization, please
      describe this process as well. What did you try that did not work?
      How did you arrive at your solution? The notes you have been
      writing throughout your project should be helpful here. Convince
      us you worked hard to arrive at a good solution.
   -  If you started with an existing piece of code, please mention it
      (and where it came from) here.

Synchronization
~~~~~~~~~~~~~~~

Parallelization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

OpenMP
^^^^^^

MPI
^^^

CUDA
^^^^

Experiments
-----------

-  RESULTS: How successful were you at achieving your goals? We expect
   results sections to differ from project to project, but we expect
   your evaluation to be very thorough (your project evaluation is a
   great way to demonstrate you understood topics from this course).
   Here are a few ideas:

   -  If your project was optimizing an algorithm, please define how you
      measured performance. Is it wall-clock time? Speedup? An
      application specific rate? (e.g., moves per second, images/sec)
   -  Please also describe your experimental setup. What were the size
      of the inputs? How were requests generated?
   -  Provide graphs of speedup or execute time. Please precisely define
      the configurations being compared. Is your baseline
      single-threaded CPU code? It is an optimized parallel
      implementation for a single CPU?
   -  Recall the importance of problem size. Is it important to report
      results for different problem sizes for your project? Do different
      workloads exhibit different execution behavior?
   -  **IMPORTANT:** What limited your speedup? Is it a lack of
      parallelism? (dependencies) Communication or synchronization
      overhead? Data transfer (memory-bound or bus transfer bound). Poor
      SIMD utilization due to divergence? As you try and answer these
      questions, we strongly prefer that you provide data and
      measurements to support your conclusions. If you are merely
      speculating, please state this explicitly. Performing a solid
      analysis of your implementation is a good way to pick up credit
      even if your optimization efforts did not yield the performance
      you were hoping for.
   -  Deeper analysis: Can you break execution time of your algorithm
      into a number of distinct components. What percentage of time is
      spent in each region? Where is there room to improve?
   -  Was your choice of machine target sound? (If you chose a GPU,
      would a CPU have been a better choice? Or vice versa.)

If the GridSolver’s parameter ``grid_x`` and ``grid_y`` is carefully
tuned, it can always perform better than EquSolver with different
backend configuration.

|image4|

Benchmark Analysis for 3rd-party Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark Analysis for Non 3rd-party Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Case study: OpenMP
~~~~~~~~~~~~~~~~~~

|image5|

Case study: MPI
~~~~~~~~~~~~~~~

|image6|

Case study: CUDA
~~~~~~~~~~~~~~~~

|image7|

REFERENCE
---------

[1] Pérez, Patrick, Michel Gangnet, and Andrew Blake. “Poisson image
editing.” *ACM SIGGRAPH 2003 Papers*. 2003. 313-318.

.. |image0| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_src.png
.. |image1| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_mask.png
.. |image2| image:: https://github.com/Trinkle23897/DIP2018/raw/master/1/image_fusion/test2_target.png
.. |image3| image:: /_static/images/result2.jpg
.. |image4| image:: /_static/images/benchmark.png
.. |image5| image:: /_static/images/openmp.png
.. |image6| image:: /_static/images/mpi.png
.. |image7| image:: /_static/images/cuda.png
