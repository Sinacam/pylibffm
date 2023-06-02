# pylibffm

pylibffm is a wrapper around libffm to allow using scipy and numpy arrays as input.

## Installing
```bash
git clone --recurse-submodules https://github.com/Sinacam/pylibffm
cd pylibffm
pip install .
```

## Documentation
The API only consists of
+ `train`
+ `load`
+ `Model`
    + `Model.save`
    + `Model.predict`

Use `help` or read their docstring for their usage.

## Diff with libffm
To be deterministic, set openmp threads to 1. For pylibffm, do
```bash
OMP_NUM_THREADS=1 python <script>.py
```

For libffm, run with `-s 1` (the default). This should yield identical models.