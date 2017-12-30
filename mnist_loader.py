import sys

if sys.version_info >= (3,0):
    import pickle
    pckle = pickle
else:
    import cPickle
    pckle = cPickle
import gzip

import numpy as np


def load_data_from_file(filepath):

    filename = filepath
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pckle.load(f)
    f.close()

    return (training_data, validation_data, test_data)


def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data(filepath):

    tr_d, va_d, te_d = load_data_from_file(filepath)
    
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

