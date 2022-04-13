## Environment configuration

OS: Red Hat Enterprise Linux Workstation 7.9 (Maipo)

CPU: 8x Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

GPU: GeForce RTX 2080 8G

Python: 3.6.8

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
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b openmp --method equ -c 8
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b openmp --method equ -c 8
# mpi

# cuda

# taichi-cpu

# taichi-gpu

```

| EquSolver  | square6 | square7 | square8 | square9 | square10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  | 4097    | 16385   | 65537   | 262145  | 1048577  |
| NumPy      | 0.84s   | 3.24s   | 12.25s  | 52.12s  | 222.44s  |
| GCC        | 0.08s   | 0.30s   | 1.21s   | 4.99s   | 22.00s   |
| OpenMP     | 0.02s   | 0.04s   | 0.14s   | 0.59s   | 8.63s    |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |

| EquSolver  | circle6 | circle7 | circle8 | circle9 | circle10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  | 4256    | 16676   | 65972   | 262338  | 1049486  |
| NumPy      | 0.86s   | 3.27s   | 12.33s  | 52.42s  | 222.85s  |
| GCC        | 0.08s   | 0.31s   | 1.21s   | 4.84s   | 22.16s   |
| OpenMP     | 0.02s   | 0.04s   | 0.13s   | 0.49s   | 8.07s    |
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
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b gcc --method grid --grid-x 8 --grid-y 8
pie -s circle10.png -t circle10.png -m circle10.png -o result.png -n 5000 -b gcc --method grid --grid-x 8 --grid-y 8 
# openmp
pie -s square10.png -t square10.png -m square10.png -o result.png -n 5000 -b openmp --method grid -c 8 --grid-x 2 --grid-y 16

# mpi

# cuda

# taichi-cpu

# taichi-gpu

```
| GridSolver | square6 | square7 | square8 | square9 | square10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  | 4356    | 16900   | 66564   | 264196  | 1052676  |
| NumPy      | 0.79s   | 2.84s   | 12.35s  | 50.62s  | 208.60s  |
| GCC        | 0.09s   | 0.35s   | 1.38s   | 5.53s   | 24.73s   |
| OpenMP     | 0.02s   | 0.06s   | 0.20s   | 0.79s   | 5.44s    |
| MPI        |         |         |         |         |          |
| CUDA       |         |         |         |         |          |
| Taichi-CPU |         |         |         |         |          |
| Taichi-GPU |         |         |         |         |          |

| GridSolver | circle6 | circle7 | circle8 | circle9 | circle10 |
| ---------- | ------- | ------- | ------- | ------- | -------- |
| # of vars  | 5476    | 21316   | 84100   | 335241  | 1338649  |
| NumPy      | 0.85s   | 3.09s   | 13.20s  | 56.32s  | 224.65s  |
| GCC        | 0.10s   | 0.38s   | 1.48s   | 5.83s   | 25.06s   |
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
