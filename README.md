# mlprof

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/alcf-mlops/mlprof)

Profiling tools for performance studies of competing ML frameworks on HPC systems


## To do

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

