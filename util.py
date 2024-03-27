import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import numpy as np
import warnings

import random
import copy
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import importlib
import matplotlib as mpl
# import imageio


from tqdm import tqdm
import datetime
import sys
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

from scipy import signal
import dgl
import dgl.nn as gnn
import math


from joblib import dump, load

def init(module, weight_init, bias_init, gain=1):
    try:
        weight_init(module.weight.data, gain=gain)
    except:
        weight_init(module.data, gain=gain)
    try:
        if module.bias is not None:
            bias_init(module.bias.data)
    except:
        pass
    return module


def ka(x, eps, a):
    return (2/eps) * (x - a - 2/eps)

def ka_(x, eps, a):
    return (-2/eps) * (x - a + 2/eps)

def sigma(x, a, eps):
    if x<=a :
        x = 0
    elif (a<x) and (x<=a+eps/2):
        x = (-0.5 * ka(x, eps, a)**4 - ka(x, eps, a)**3 + ka(x, eps, a) + 0.5)
    elif (a+eps/2<x) and (x<=a+eps):
        x = (0.5 * ka(x, eps, a)**4 - ka(x, eps, a)**3 + ka(x, eps, a) + 0.5)
    elif  a+eps < x:
        x = 1
    return x

def sigma_(x, a, eps):
    if x<=a-eps:
        x = 1
    elif (a-eps<x) and (x<=a - eps/2) :
        x = (0.5 * ka_(x, eps, a)**4 - ka_(x, eps, a)**3 + ka_(x, eps, a) + 0.5)
    elif (a-eps/2<x) and (x<=a):
        x = (-0.5 * ka_(x, eps, a)**4 - ka_(x, eps, a)**3 + ka_(x, eps, a) + 0.5)
    elif a < x:
        x = 0
    return x

def pen(x, x_):
    temp = 1 + 1.5 * sigma_(x, 85, 30) * sigma(x_, x, 10) +\
                1 * sigma(x, 155, 100) * sigma_(x_, x, 20)
    return temp
def cal_gmse(y, y_):
    y_ = y_[:, 0]
    temp_list = []
    for i in range(y.shape[0]):
        temp_list.append(((y[i] - y_[i])**2) * pen(y[i], y_[i]))

    return np.sqrt(np.mean(temp_list))

def cal_time_lag(y, predicted_np, seq_st_ed, interval=5):  
    predicted_np = predicted_np[:, 0]
    temp_list = []
    for i in range(seq_st_ed.shape[0]):
        seq1 = y[seq_st_ed[i, 0]:seq_st_ed[i, 1] + 1]
        seq2 = predicted_np[seq_st_ed[i, 0]:seq_st_ed[i, 1] + 1]
        correlation = signal.correlate(seq1, seq2, mode="full")
        lags = signal.correlation_lags(seq1.shape[0], seq2.shape[0], mode="full")
        lag = lags[np.argmax(correlation)]
        temp_list.append(lag * interval)
    return -np.mean(temp_list)