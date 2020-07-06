#!/bin/bash
python3 main.py -dataset fusion \
                -network s2mpmr -protocol p1  \
                -batch_size 10 -num_epochs 100

python3 main.py -dataset fusion \
                -network s2mpmr -protocol p2  \
                -batch_size 10 -num_epochs 100