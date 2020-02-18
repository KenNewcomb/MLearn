'''preprocessing.py: A collection of functions that are useful for manipulating input data prior to learning.'''
from math import floor

def train_val_test_split(X, y, test_size=0.2, val_size=0.0):
    '''Splits X, y data into training, validation, and testsets.'''
    # Get indices of splits.
    m = len(X)
    train_index = floor(m*(1-test_size-val_size))
    val_index = floor(m*(1-test_size))

    # Split X, y
    X_train = X[:train_index]
    y_train = y[:train_index]

    X_val = X[train_index:val_index]
    y_val = y[train_index:val_index]

    X_test = X[val_index:]
    y_test = y[val_index:]

    if X_val == []:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
