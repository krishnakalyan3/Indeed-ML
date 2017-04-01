#!/usr/bin/env python3

import bcolz
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def scale_model(X):
    scale_d = preprocessing.StandardScaler()
    return scale_d.fit(X)


def load_array(fname):
    return bcolz.open(fname)[:]


def max_model(X):
    return X.max(axis=0)


def max_transform(max_val, X):
    max_val[max_val == 0] = 1
    return X/max_val


def fix_tags(data):
    cat_array = np.zeros([12], dtype=int)
    data_arry = ['part-time-job',
                 'full-time-job',
                 'hourly-wage',
                 'salary',
                 'associate-needed',
                 'bs-degree-needed',
                 'ms-or-phd-needed',
                 "licence-needed",
                 '1-year-experience-needed',
                 '2-4-years-experience-needed',
                 '5-plus-years-experience-needed',
                 'supervising-job']

    split_str = data.split(" ")
    for i in split_str:
        index = data_arry.index(i)
        cat_array[index] = 1
    return cat_array

def eval_models(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    return (acc, f1)

def save_np(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()