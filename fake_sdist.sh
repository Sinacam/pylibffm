#! /bin/bash

# To run this script, create conda environments py3x for each python version 3.x

versions=({7..11})

eval "$(conda shell.bash hook)"
for v in "${versions[@]}"; do
    conda activate py3$v
    make
    conda deactivate
done

python3 setup.py sdist
