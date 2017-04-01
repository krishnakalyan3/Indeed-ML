#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from utils import fix_tags, eval_models, save_np
import numpy as np
import codecs
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import itemfreq
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

TRAIN = '../dataset/train.tsv'
TEST = '../dataset/test.tsv'


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


def train_pre1(data):
    data['part-time-job'], \
    data['full-time-job'], \
    data['hourly-wage'], \
    data['salary'], \
    data['associate-needed'], \
    data['bs-degree-needed'], \
    data['ms-or-phd-needed'], \
    data['licence-needed'], \
    data['1-year-experience-needed'], \
    data['2-4-years-experience-needed'], \
    data['5-plus-years-experience-needed'], \
    data['supervising-job'] = zip(*data['tags'].map(fix_tags))
    return data


def vec_data(train, test):
    vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                           stop_words='english')
    merge_df = train['description'].append(test['description'])
    vect.fit(merge_df)
    return vect


def svc_model(X, y):
    svc = SVC(kernel='rbf', cache_size=5000, verbose=False)
    params = {}
    params['C'] = np.logspace(-5, 4.5, 15)
    clf = RandomizedSearchCV(svc, param_distributions=params, n_iter=10, n_jobs=-1, verbose=0)
    model = clf.fit(X, y)
    return model.best_estimator_

def subset_data(X, y):
    rus = RandomUnderSampler(random_state=1337)
    X_res, y_res = rus.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                        test_size=0.2, stratify=y_res,
                                                        random_state=1337)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data = pd.read_csv(TRAIN, sep='\t')
    test = pd.read_csv(TEST, sep='\t')
    data = data.dropna()

    vect = vec_data(data, test)
    train_X = vect.transform(data['description']).todense()
    test_X = vect.transform(test['description']).todense()
    train_y = train_pre1(data)

    y_hats = np.empty((2921, 12))
    for k, i in enumerate(data_arry):
        y = train_y[i]
        X = train_X
        tr_X, val_X, tr_y, val_y = subset_data(X, y)
        model = svc_model(tr_X, tr_y)
        y_hat_tr = model.predict(tr_X)
        y_hat_val = model.predict(val_X)
        eval_train = eval_models(y, y_hat_tr)
        eval_val = eval_models(y, y_hat_val)
        print("Train Eval ", eval_train)
        print("Validation Eval ", eval_val)
        y_hats[:, k] = model.predict(test_X)

    print(y_hats.shape)
    save_np('../save_data/save_array.bc', y_hats)
