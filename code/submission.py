#!/usr/bin/env python
import numpy as np
from utils import load_array

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

if __name__ == '__main__':
    sub_data = load_array('../save_data/save_array.bc')
    labels = [['tags']]
    for k1, v1 in enumerate(sub_data):
        each_row = []
        for k2, v2 in enumerate(v1):
            if v2 == 1:
                each_row.append(data_arry[k2])

        labels.append(each_row)

    res_file = open("../submissions/submit.tsv", 'w')

    for r in labels:
        res_file.write(' '.join(r) + "\n")
    res_file.close()