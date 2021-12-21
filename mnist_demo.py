# -- coding: utf-8 --
import gzip
import os
import pickle
import numpy as np
from neuron_network.activation_function import sigmoid, softmax_avoid_overflow


def one_hot_function(x, num_class=None):
    """one_hot label into vector"""
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros([len(x), num_class])
    ohx[range(len(x)), x] = 1

    return ohx


def load_mnist(data_folder=r'D:\AI\myData\mnist', flattern=False, one_hot=False, normalize=False):
    """加载本地mnist数据集返回numpy数组"""
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            if one_hot:
                y_train = one_hot_function(y_train)

    with gzip.open(paths[1], 'rb') as imgpath:
        if flattern:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28 * 28)
        else:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        if normalize:
            x_train = (x_train / 127.5) - 1

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        if one_hot:
            y_test = one_hot_function(y_test)

    with gzip.open(paths[3], 'rb') as imgpath:
        if flattern:
            x_test = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28 * 28)
        else:
            x_test = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
        if normalize:
            x_test = (x_test / 127.5) - 1

    return (x_train, y_train), (x_test, y_test)


def get_data():
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(r'D:\AI\myData\mnist', flattern=True, one_hot=True, normalize=False)

    return x_test, y_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_avoid_overflow(a3)

    return y


def calculate_accuracy_onebyone():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        index = np.argmax(y)
        if index == t[i]:
            accuracy_cnt += 1

    return float(accuracy_cnt/len(x))


def calculate_accuracy_batch(batchsize):
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0

    for i in range(0, len(x), batchsize):
        x_batch = x[i: i + batchsize]
        y_batch = predict(network, x_batch)
        index = np.argmax(y_batch, axis=1)  # 按行取索引
        accuracy_cnt += np.sum(index == t[i: i + batchsize])

    return float(accuracy_cnt/len(x))