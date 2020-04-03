import pandas as pn
from sklearn.model_selection import train_test_split
import numpy as np


def read_input(train_size, test_size, data_path, novel_class):
    data = pn.read_csv(data_path, header=None)
    features = len(data.columns) - 1
    # novel data index
    n_index = [0 for i in range(0, 30)]

    # # preprocess data
    data = data.fillna((data.mean(axis=0)))
    # removing NAN value
    # Prepare test phase
    c=0
    # bellow loop will generate 10 test csv
    for i in range(0,10):
        n_class = []
        x = data.sample(n=test_size, replace=False)
        # taking test data in x

        # from x separate nobel class data
        for j in novel_class:
            n = x.loc[x[features] == j]
            n_class.append(n)
            x = x.loc[x[features] != j]

        # merging all type of novel class data to n_class
        n_class = pn.concat(n_class)

        # creating test dataset where novel data were at end
        test = pn.DataFrame(x)
        s_n = len(test)
        test = test.append(n_class)
        e_n = len(test)
        test.to_csv('../datasets/data/test' + str(i) + '.csv', header=None, index=False, encoding="utf-8")

        # n_index stores novel data starting and end index
        n_index[c] = s_n
        n_index[c + 1] = e_n - 1
        c+=2

    # here we create train csv
    x = data.sample(n=train_size, replace=False)

    # removing novel data from train set
    for i in novel_class:
        x = x.loc[x[features] != i]

    train = pn.DataFrame(x)
    train.to_csv('../datasets/data/train.csv', header=None, index=False, encoding="utf-8")
    t = pn.read_csv('../datasets/data/test0.csv', header=None)
    X_train = train.iloc[:, 0:features]
    y_train = train.iloc[:, features]
    X_test = t.iloc[:, 0:features]
    y_test = t.iloc[:, features]

    return X_train, X_test, y_train, y_test, features, n_index
