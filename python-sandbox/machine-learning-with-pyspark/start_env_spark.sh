#!/bin/bash -x

eval "$(conda shell.bash hook)"
conda activate spark

jupyter notebook
