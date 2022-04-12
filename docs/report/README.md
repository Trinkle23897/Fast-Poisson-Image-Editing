Challenge: How to implement a fully-parallelized Jacobi Iteration to support a real-time image fusion?

- Workload/constrains: similar to the 2d-grid example demonstrated in class.

Resources:

- Codebase: https://github.com/Trinkle23897/DIP2018/blob/master/1/image_fusion/image_fusion.cpp, written by Jiayi Weng, one of our group members
- Computation Resources: GHC (CPU, GPU - 2080), PSC (CPU, GPU), laptop (CPU, GPU - 1060)

Goals:

- [x] 75%: implement one parallel version of PIE
- [x] 100%: benchmark the algorithm with OpenMP/MPI/CUDA implementation
- [ ] 125%: include a interactive python frontend that can demonstrate the result in a user-friendly style. Real-time computation on a single GTX-1060 Nvidia GPU.

Platform choice:

- OS: Linux, Ubuntu machine
- Language: C++/CUDA for core development, Python for interactive frontend

Schedule:

- 3.28 - 4.9: implement parallel version of PIE (OpenMP, MPI, CUDA)
- 4.11 - 4.22: benchmarking, optimizing, write Python interactive frontend
- 4.25 - 5.5: write report
