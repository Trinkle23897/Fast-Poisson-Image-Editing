.. fpie documentation master file, created by
   sphinx-quickstart on Sun Apr 17 16:39:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fpie
===============

`Poisson Image
Editing <https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf>`__ is
a technique that can blend two images together without artifacts. Given
a source image and its corresponding mask, and a coordination on target
image, this algorithm can always generate amazing result.

This project aims to provide a fast poisson image editing algorithm
(based on `Jacobi
Method <https://en.wikipedia.org/wiki/Jacobi_method>`__) that can
utilize multi-core CPU or GPU to handle a high-resolution image input.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   get_start
   backend
   benchmark
   report
