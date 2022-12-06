# mlprof

Profiling tools for performance studies of competing ML frameworks on HPC systems

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
