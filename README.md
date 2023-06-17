# pylibffm

pylibffm is a wrapper around libffm to allow using scipy and numpy arrays as input.
pylibffm requires SSE and openmp.

## Installing
If you're running python 3.7~3.11 on linux
```bash
pip install pylibffm
```
Otherwise
```bash
git clone --recurse-submodules https://github.com/Sinacam/pylibffm
cd pylibffm
make
pip install .
```
Currently, the distribution is incredibly hacky and cannot be pip installed from source.
Windows build support is lacking, but code should be correct.

## Documentation
The API consists of
+ `train`
+ `load`
+ `Model`
    + `Model.save`
    + `Model.predict`
+ `make_bin`
+ `train_with_bin`

Use `help` or read their docstring for their usage.

## Diff with libffm
To be deterministic, set openmp threads to 1. For pylibffm, do
```bash
OMP_NUM_THREADS=1 python <script>.py
```

For libffm, run with `-s 1` (the default). This should yield identical models.