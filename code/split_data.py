#!/usr/bin/env python3
import os
import numpy as np

TRAIN = '../dataset/train/'
TEST = '../dataset/test/'
VAL = '../dataset/val/'
ROOT = '../dataset/'

SEED = 1337

if __name__ == '__main__':
    if not os.path.exists(TRAIN + str(SEED)):
        os.makedirs(TRAIN + str(SEED))
    if not os.path.exists(TEST + str(SEED)):
        os.makedirs(TEST + str(SEED))
    if not os.path.exists(VAL + str(SEED)):
        os.makedirs(VAL + str(SEED))

    data = np.loadtxt(ROOT + 'train.tsv', delimiter=",")
    print(data.shape)

