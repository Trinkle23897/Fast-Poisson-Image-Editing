## Environment configuration

OS: Red Hat Enterprise Linux Workstation 7.9 (Maipo)

CPU: 8x Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

GPU: GeForce RTX 2080 8G

## Problem size vs backend

To run and get the time spend:

```bash
$ pie -s $NAME.png -t $NAME.png -m $NAME.png -o result.png -n 5000 -b $BACKEND --method $METHOD ...
```

The following table shows the best performance of corresponding backend choice, i.e., tuning other parameters on square6/circle6 and apply them to other tests, instead of using the default value.

### EquSolver

The benchmark command for squareX and circleX:

```bash
# numpy

# gcc

# openmp

# mpi

# cuda

# taichi-cpu

# taichi-gpu

```

| backend    | square2 | square3 | square4 | square5 | square6 |
| ---------- | ------- | ------- | ------- | ------- | ------- |
| NumPy      |         |         |         |         |         |
| GCC        |         |         |         |         |         |
| OpenMP     |         |         |         |         |         |
| MPI        |         |         |         |         |         |
| CUDA       |         |         |         |         |         |
| Taichi-CPU |         |         |         |         |         |
| Taichi-GPU |         |         |         |         |         |

| backend    | circle2 | circle3 | circle4 | circle5 | circle6 |
| ---------- | ------- | ------- | ------- | ------- | ------- |
| NumPy      |         |         |         |         |         |
| GCC        |         |         |         |         |         |
| OpenMP     |         |         |         |         |         |
| MPI        |         |         |         |         |         |
| CUDA       |         |         |         |         |         |
| Taichi-CPU |         |         |         |         |         |
| Taichi-GPU |         |         |         |         |         |

### GridSolver

The benchmark command for squareX and circleX:

```bash
# numpy

# gcc

# openmp

# mpi

# cuda

# taichi-cpu

# taichi-gpu

```
| backend    | square2 | square3 | square4 | square5 | square6 |
| ---------- | ------- | ------- | ------- | ------- | ------- |
| NumPy      |         |         |         |         |         |
| GCC        |         |         |         |         |         |
| OpenMP     |         |         |         |         |         |
| MPI        |         |         |         |         |         |
| CUDA       |         |         |         |         |         |
| Taichi-CPU |         |         |         |         |         |
| Taichi-GPU |         |         |         |         |         |

| backend    | circle2 | circle3 | circle4 | circle5 | circle6 |
| ---------- | ------- | ------- | ------- | ------- | ------- |
| NumPy      |         |         |         |         |         |
| GCC        |         |         |         |         |         |
| OpenMP     |         |         |         |         |         |
| MPI        |         |         |         |         |         |
| CUDA       |         |         |         |         |         |
| Taichi-CPU |         |         |         |         |         |
| Taichi-GPU |         |         |         |         |         |


## Per backend performance

In this section, we will perform ablation studies on OpenMP/MPI/CUDA backend. We use circle5 with 25000 iterations as the experiment setting.

### OpenMP

Command to run:

```bash

```

### MPI


Command to run:

```bash
```


### CUDA


Command to run:

```bash
```
