import numpy as np
import scipy.io as scio
import scipy.sparse as scsp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from numpy.random import randint
import numpy.ma as npm
import pandas as pd
from get_sn import get_sn
import copy
from sklearn.preprocessing import StandardScaler


transform = transforms.Compose([
    transforms.ToTensor(),
])


def read_mymat(path, name, sp, missrate, sparse=False):
    '''
    :param path: dataset path
    :param name: dataset name
    :param sp: sp
    :param missrate: missing rate
    :param sparse: sparse
    :return: X, Y, Sn
    '''
    mat_file = path + name
    f = scio.loadmat(mat_file)

    if 'Y' in sp:
        if (name == 'handwritten0.mat') or (name == 'BRAC.mat') or (name == 'ROSMAP.mat'):
            Y = (f['gt']).astype(np.int32)
        else:
            Y = (f['gt']-1).astype(np.int32)
    else:
        Y = None

    if 'X' in sp:
        Xa = f['X']
        Xa = Xa.reshape(Xa.shape[1], )
        X = []
        if sparse:
            for x in Xa:
                X.append(scsp.csc_matrix(x).toarray().astype(np.float64))
        else:
            for x in Xa:
                X.append(x.astype(np.float64))
    else:
        X = None
    n_sample = len(X[0][0])
    n_view = len(X)
    Sn = get_sn(n_view, n_sample, missrate).astype(np.float32)

    for i in range(n_view):
        X[i] = X[i].T
    return X, Y, Sn



def build_ad_dataset(Y, p, seed=999):
    '''
    Converting the original multi-class multi-view dataset into an anomaly detection dataset
    :param seed: Random seed
    :param p: proportion of normal samples for training
    :param neg_class_id: the class used as negative class (outliers)
    :param Y: the original class indexes
    :return:
    '''
    np.random.seed(seed=seed)
    Y = np.squeeze(Y)
    Y_idx = np.array([x for x in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * p))
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'train': train_idx, 'test': test_idx}
    return partition



def get_validation_set_Sn(X_train, Sn_train, Y_train, val_p=0.1):
    tot_num = X_train[0].shape[0]
    perm = np.random.permutation(tot_num)
    val_idx = perm[:np.int_(np.ceil(tot_num * val_p))]
    train_idx = perm[np.int_(np.ceil(tot_num * val_p)):]
    val_set = [X_train[i][val_idx] for i in range(len(X_train))]
    val_train_set = [X_train[i][train_idx] for i in range(len(X_train))]
    val_train_set_Y = Y_train[train_idx]
    val_val_set_Y = Y_train[val_idx]
    val_train_set_Sn = Sn_train[train_idx]
    val_val_set_Sn = Sn_train[val_idx]
    return val_train_set, val_set, val_train_set_Sn, val_val_set_Sn, val_train_set_Y, val_val_set_Y


def get_validation_set(Y, p, seed=999):
    np.random.seed(seed=seed)
    Y = np.squeeze(Y)
    Y_idx = np.array([x for x in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * p))
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'train': train_idx, 'val': test_idx}
    return partition



def process_data(X, n_view):
    eps = 1e-10
    if (n_view == 1):
        X = StandardScaler().fit_transform(X)
    else:
        X = [StandardScaler().fit_transform(X[i]) for i in range(n_view)]
    return X


class partial_mv_dataset(Dataset):
    def __init__(self, data, Sn, Y):
        '''
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y
        self.Sn = Sn

    def __getitem__(self, item):
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Sn), torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]


class mv_dataset(Dataset):
    def __init__(self, data, Y):
        '''
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y

    def __getitem__(self, item):
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]

def partial_mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        Sn_data = batch[y][1]
        label_data = batch[y][2]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))], torch.cat(new_Sn, dim=0), torch.cat(new_label, dim=0)

def mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        label_data = batch[y][1]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))],  torch.cat(new_label, dim=0)

def tensor_intersection(x, y):
    return torch.tensor(list(set(x.tolist()).intersection(set(y.tolist()))))



