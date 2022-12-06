# mlprof

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/alcf-mlops/mlprof)

Profiling tools for performance studies of competing ML frameworks on HPC systems


## To do

- [ ] Write DeepSpeed Trainer that wraps [`src/mlprof/network/pytorch/network.py`](./src/mlprof/network/pytorch/network.py)
    - Reference: [DeepSpeed -- Getting Started](https://www.deepspeed.ai/getting-started/)
- [ ] MPI Profiling to get all collective comm. ops with same model in DeepSpeed, DDP, and Horovod
- [ ] Start with 2 nodes first and next scale w/ increasing number of nodes
- [ ] Get profiles for DeepSpeed Zero 1, 2, 3 and Mixture of experts (MoE)
- [ ] Identify what parameters can impact performance such as NCCL environment variables and framework-specific parameters
- [ ] Do the analysis for standard models and large language models (LLMs)
- [ ] (optional for now) Develop auto-tuning methods to set these parameters for optimal performance


## Setup

To install:

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

To run:

```bash
cd src/mlprof
./train.sh > train.log 2>&1 &
```

To view output:

```bash
tail -f train.log $(tail -1 logs/latest)
```

Configuration options can be overridden on the command line, e.g.
(and are specified in [`src/mlprof/conf/config.yaml`](src/mlprof/conf/config.yaml))

```bash
./train.sh data.batch_size=256 network.hidden_size=64 > train.log 2>&1 &
```

## Profiling

To run an experiment with `mpitrace` enabled, on Polaris, we can explicitly set the `LD_PRELOAD` environment variable, e.g.

```bash
LD_PRELOAD=/soft/perftools/mpitrace/lib/libmpitrace.so ./train.sh > train.log 2>&1 &
```

which will write MPI Profiling information to a `mpi_profile.XXXXXX.Y` file containing the following information:

```bash
Data for MPI rank 0 of 8:
Times from MPI_Init() to MPI_Finalize().
-----------------------------------------------------------------------
MPI Routine                        #calls     avg. bytes      time(sec)
-----------------------------------------------------------------------
MPI_Comm_rank                           3            0.0          0.000
MPI_Comm_size                           1            0.0          0.000
MPI_Bcast                               2           16.5          0.000
-----------------------------------------------------------------------
total communication time = 0.000 seconds.
total elapsed time       = 232.130 seconds.
user cpu time            = 122.013 seconds.
system time              = 96.950 seconds.
max resident set size    = 4064.422 MiB.

-----------------------------------------------------------------
Message size distributions:

MPI_Bcast                 #calls    avg. bytes      time(sec)
                               1           4.0          0.000
                               1          29.0          0.000

-----------------------------------------------------------------

Summary for all tasks:

  Rank 0 reported the largest memory utilization : 4064.42 MiB
  Rank 0 reported the largest elapsed time : 232.13 sec

  minimum communication time = 0.000 sec for task 6
  median  communication time = 0.000 sec for task 5
  maximum communication time = 0.000 sec for task 4


MPI timing summary for all ranks:
taskid             host    cpu    comm(s)  elapsed(s)     user(s)   system(s)   size(MiB)    switches
     0   x3210c0s37b1n0      0       0.00      232.13      122.01       96.95     4064.42   240460957
     1   x3210c0s37b1n0      1       0.00      227.60      126.06       95.88     4001.15   231353798
     2   x3210c0s37b1n0      2       0.00      227.63      135.59       85.93     3965.89   230507191
     3   x3210c0s37b1n0      3       0.00      227.63      126.33       95.75     4003.07   230342296
     4    x3210c0s7b0n0      0       0.00      227.66      137.07       83.80     4039.70   209534784
     5    x3210c0s7b0n0      1       0.00      227.64      125.65       96.13     4004.05   230622703
     6    x3210c0s7b0n0      2       0.00      227.64      134.53       87.16     3968.59   229010244
     7    x3210c0s7b0n0      3       0.00      227.67      125.24       96.90     4004.26   233186459
```
