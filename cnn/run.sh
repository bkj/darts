#!/bin/bash

# run.sh

cd cnn
CUDA_VISIBLE_DEVICES=6 python train_search.py --unrolled

# CUDA_VISIBLE_DEVICES=7 python train.py --arch BKJ_DARTS
# CUDA_VISIBLE_DEVICES=6 python train.py --arch DARTS
# CUDA_VISIBLE_DEVICES=6 python train.py --arch DARTS --cutout --auxiliary

# CUDA_VISIBLE_DEVICES=7 python train.py --arch HYPERDARTS --cutout --auxiliary
# CUDA_VISIBLE_DEVICES=7 python train.py --arch HYPERDARTS2 --cutout --auxiliary

# CUDA_VISIBLE_DEVICES=5 python train.py --arch /home/bjohnson/projects/frog/genotype.pkl --cutout --auxiliary