#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate hyperopt

jupyter notebook
