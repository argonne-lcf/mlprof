# mlprof 

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20 style="padding: 0% 0%;">](https://wandb.ai/l2hmc-qcd/mlprof?workspace=user-saforem2) <a href="https://pytorch.org/get-started/locally/"><img alt="pyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white" style="padding: 0% 0% 0% 0%;" height=20></a> <a href="https://www.tensorflow.org"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white" style="padding: 0% 0%;" height=20></a> 

Profiling tools for performance studies of competing ML frameworks on HPC systems

</div>


<details closed><summary><b>TODO</b></summary>
<p>

## TODO
### 04/17/2023
- [ ] Work on repeating MPI profile experiments with larger batch size / network size using `module load conda/2023-01-10-unstable` on Polaris
- [ ] Try with single + multiple nodes to measure performance impact

### Older
- [x] Write DeepSpeed Trainer that wraps [`src/mlprof/network/pytorch/network.py`](./src/mlprof/network/pytorch/network.py)
    - Reference: [DeepSpeed -- Getting Started](https://www.deepspeed.ai/getting-started/)
- [ ] MPI Profiling to get all collective comm. ops with same model in DeepSpeed, DDP, and Horovod
   - Reference: [Profiling](https://github.com/argonne-lcf/mlprof#profiling) using `libmpitrace.so` on Polaris
- [ ] Start with 2 nodes first and next scale w/ increasing number of nodes
- [ ] Get profiles for DeepSpeed Zero 1, 2, 3 and Mixture of experts (MoE)
- [ ] Identify what parameters can impact performance such as NCCL environment variables and framework-specific parameters
- [ ] Do the analysis for standard models and large language models (LLMs)
- [ ] Develop auto-tuning methods to set these parameters for optimal performance
    
#### 2023-02-20

- [ ] Associate `mpiprofile`'s with backend + attach logs to keep everything together

- [ ] Scale up message sizes in mpiprofiles
- [ ] Aggregate into table, grouped by backend 
- [ ] Test `fp16` support w/ all backends
- [ ] Ensure all GPUs being utilized 
	- w/ `conda/2022-09-08-hvd-nccl` all processes get mapped to GPU0 for some reason     

</p>
</details>

## Setup

> **Note**
> <br> These instructions assume that your active environment already has 
> the required ML libraries installed.
>
> This allows us to perform an isolated editable installation _inside_ our
existing environment, and allows it to access previously installed libraries.

To install:

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

## Running Experiments

We support distributed training using the following backends:

- [microsoft/DeepSpeed](https://github.com/microsoft/deepspeed) (`backend=deepspeed`)
- [horovod/horovod](https://github.com/horovod/horovod) (`backend=horovod`)
- [pytorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`backend=DDP`)

which we specify via `backend=BACKEND` as an argument to the [src/mlprof/train.sh](./src/mlprof/train.sh) script:

```bash
cd src/mlprof
./train.sh backend=BACKEND train.log 2>&1 &
```

and view the resulting output:

```bash
tail -f train.log $(tail -1 logs/latest)
```

### Configuration

Configuration options can be overridden on the command line, e.g.
(and are specified in [`src/mlprof/conf/config.yaml`](src/mlprof/conf/config.yaml))

```bash
./train.sh backend=DDP data.batch_size=256 network.hidden_size=64 > train.log 2>&1 &
```

### Running on Polaris

<details closed><summary><b>Run on Polaris:</b></summary>
<p>

```bash
$ qsub \
    -A <project-name> \
    -q debug-scaling \
    -l select=2 \
    -l walltime=12:00:00,filesystem=eagle:home:grand \
    -I
$ module load conda/2023-01-10-unstable
$ conda activate base
$ git clone https://www.github.com/argonne-lcf/mlprof
$ cd mlprof
$ mkdir -p venvs/polaris/2023-01-10
$ python3 -m venv venvs/polaris/2023-01-10 --system-site-packages
$ source venvs/polaris/2023-01-10
$ python3 -m pip install --upgrade pip setuptools wheel
$ python3 -m pip install -e .
$ cd src/mlprof
$ # -------------------------------------------------------------
$ # the following are necessary when using the DeepSpeed backend
$ export CFLAGS="-I${CONDA_PREFIX}/include/"
$ export LDFLAGS="-L${CONDA_PREFIX}/lib/" 
$ echo "PATH=${PATH}" > .deepspeed_env 
$ echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
$ echo "https_proxy=${https_proxy}" >> .deepspeed_env
$ echo "http_proxy=${http_proxy}" >> .deepspeed_env 
$ echo "CFLAGS=${CFLAGS}" >> .deepspeed_env
$ echo "LDFLAGS=${LDFLAGS}" >> .deepspeed_env
$ # -------------------------------------------------------------
$ # TO TRAIN:
$ ./train.sh backend=deepspeed > train.log 2>&1 &
$ # TO VIEW OUTPUT:
$ tail -f train.log $(tail -1 logs/latest)
```

</p>
</details>


### Profiling

To run an experiment with `mpitrace` enabled, on Polaris, we can explicitly set the `LD_PRELOAD` environment variable, e.g.

```bash
LD_PRELOAD=/soft/perftools/mpitrace/lib/libmpitrace.so ./train.sh > train.log 2>&1 &
```

which will write MPI Profiling information to a `mpi_profile.XXXXXX.Y` file containing the following information:

<details closed><summary>MPI Profile Results</summary>
<p>

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

</p>
</details>
