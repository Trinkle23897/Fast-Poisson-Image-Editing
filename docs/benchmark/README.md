## Environment configuration

OS: Red Hat Enterprise Linux Workstation 7.9 (Maipo)

CPU: 8x Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

GPU: GeForce RTX 2080 8G

## Problem size vs backend

To run and get the time spend:

```bash
$ pie -s $NAME.png -t $NAME.png -m $NAME.png -o result.png -n 5000 -b $BACKEND --method $METHOD ...
```

The following table shows the best performance of corresponding backend choice, i.e., tuning other parameters on square10/circle10 and apply them to other tests, instead of using the default value.

### EquSolver

The benchmark commands for squareX and circleX:

```bash
# numpy
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b numpy --method equ
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b numpy --method equ
# gcc
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b gcc --method equ
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b gcc --method equ
# openmp
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b openmp --method equ -c 4 # 4~6
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b openmp --method equ -c 4 # 4~6
# mpi

# cuda

# taichi-cpu

# taichi-gpu

```

| EquSolver  | square6 | square7 | square8 | square9 | square10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  |         |         |         |         |          |
| NumPy      |         |         |         |         |          |
| GCC        |         |         |         |         |          |
| OpenMP     |         |         |         |         |          |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |

| EquSolver  | circle6 | circle7 | circle8 | circle9 | circle10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  |         |         |         |         |          |
| NumPy      |         |         |         |         |          |
| GCC        |         |         |         |         |          |
| OpenMP     |         |         |         |         |          |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |

### GridSolver

The benchmark commands for squareX and circleX:

```bash
# numpy
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b numpy --method grid
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b numpy --method grid
# gcc
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b gcc --method grid
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b gcc --method grid --grid-x --grid-y 
# openmp

# mpi

# cuda

# taichi-cpu

# taichi-gpu

```
| GridSolver | square6 | square7 | square8 | square9 | square10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  |         |         |         |         |          |
| NumPy      |         |         |         |         |          |
| GCC        |         |         |         |         |          |
| OpenMP     |         |         |         |         |          |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |

| GridSolver | circle6 | circle7 | circle8 | circle9 | circle10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  |         |         |         |         |          |
| NumPy      |         |         |         |         |          |
| GCC        |         |         |         |         |          |
| OpenMP     |         |         |         |         |          |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |


## Per backend performance

In this section, we will perform ablation studies on OpenMP/MPI/CUDA backend. We use circle9 with 25000 iterations as the experiment setting.

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
