import os 
import pdb 
import json 
import random
from re import M 

from typing import Dict, Optional, Tuple, List
from numpy.matrixlib.defmatrix import asmatrix

import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 

from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils import resample


def select_col(data):
    """
    Args:
        select column of which the number of nan is the least
    """
    label = data[:, -1]
    data_no_nan = []
    column_idx, final_column = -1, None 
    nan_cnt = data.size(0)
    for i in range(data.size(1)-1):
        column = data[:, i] 
        _nan = torch.isnan(column).sum()
        if _nan == 0:
            data_no_nan.append(column)
        else:
            if _nan < nan_cnt:
                nan_cnt = _nan 
                final_column = column 
                column_idx = i
    train_idx, test_idx = [], []
    if final_column is not None:
        assert column_idx != -1
        for i, item in enumerate(torch.isnan(final_column)):
            if item:
                test_idx.append(i)
            else:
                train_idx.append(i)
    else:
        return None, None, None, None, None  
    data_no_nan.append(label)
    all_data = torch.stack(data_no_nan, dim=1)
    # pdb.set_trace()
    train_data = all_data[train_idx, :]
    train_label = final_column[train_idx]
    test_data = all_data[test_idx, :]
    return train_data, train_label, test_data, test_idx, column_idx


def impute(data, _imputer="simple"):
    if _imputer == "simple":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = imputer.fit_transform(data)
    elif _imputer == "knn":
        imputer = KNNImputer(n_neighbors=5)
        data = imputer.fit_transform(data)
    elif _imputer == "none":
        data = torch.tensor(data).nan_to_num()
        pass 
    else:
        raise ValueError
    return data


def standardize(data):
    feature = data[:, :-1]
    label = data[:, -1:]
    norm_feature = StandardScaler().fit_transform(feature)
    data = np.concatenate((norm_feature, label), axis=-1)
    return np.array(data)


def upsample(data, seed, sample_rate):
    assert sample_rate > 1.0
    minority = []
    majority = []
    for item in data:
        if int(item[-1]) == 1:
            minority.append(item)
        elif int(item[-1]) == 0:
            majority.append(item)
        else:
            raise ValueError
    upsampled_minority = resample(minority, 
                                    replace=True,
                                    n_samples=int(len(minority)*sample_rate),
                                    random_state=seed)
    print("Before upsampling: %d, after upsampling: %d" % (len(minority), len(upsampled_minority)))
    data = np.stack(majority+upsampled_minority, axis=0)
    return data    


def downsample(data, seed, sample_rate):
    assert sample_rate < 1.0
    minority = []
    majority = []
    for item in data:
        if int(item[-1]) == 1:
            minority.append(item)
        elif int(item[-1]) == 0:
            majority.append(item)
        else:
            raise ValueError
    downsampled_majority = resample(majority, 
                                    replace=False,
                                    n_samples=int(len(majority)*sample_rate),
                                    random_state=seed)
    print("Before downsampling: %d, after downsampling: %d" % (len(majority), len(downsampled_majority)))
    data = np.stack(downsampled_majority+minority, axis=0)
    return data  


def split_data(args,
                input_file: str=None, 
                all_data: np.ndarray=None,
                train_rate: float=0.8, 
                dev_rate: float=0.1, 
                test_rate: float=0.1, 
                equal_split: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    if all_data is not None:
        end_of_train = int(train_rate * len(all_data))
        end_of_dev = int((train_rate+dev_rate) * len(all_data))
        train = np.array(all_data[:end_of_train])
        dev = np.array(all_data[end_of_train:end_of_dev])
        test = np.array(all_data[end_of_dev:])
        # weight
        pos = all_data[:, -1].sum()
        neg = len(all_data) - pos 
        weight = {1: float(neg/pos)}
    else:
        # load and shuffle
        random.seed(args.seed)
        data, meta = arff.loadarff(input_file)
        data = data.tolist()
        # random.shuffle(data)
        data = np.array(data, dtype=np.float32)
        data = standardize(impute(data, _imputer=args.imputer))
        # split
        assert int(train_rate+dev_rate+test_rate)== 1
        if not equal_split:
            shuffle_index = np.random.choice(list(range(len(data))), size=len(data), replace=False)
            data = data[shuffle_index]
            end_of_train = int(train_rate * len(data))
            end_of_dev = int((train_rate+dev_rate) * len(data))
            train = data[:end_of_train]
            dev = data[end_of_train:end_of_dev]
            test = data[end_of_dev:]
        else:
            label = data[:, -1]
            pure_data = data[:, :-1]
            negative_data = pure_data[label == 0]
            positive_data = pure_data[label == 1]

            negative_train_end = int(train_rate * len(negative_data))
            positive_train_end = int(train_rate * len(positive_data))
            negative_dev_end = negative_train_end + int(dev_rate * len(negative_data))
            positive_dev_end = positive_train_end + int(dev_rate * len(positive_data))

            negative_train = negative_data[:negative_train_end]
            positive_train = positive_data[:positive_train_end]
            negative_dev = negative_data[negative_train_end:negative_dev_end]
            positive_dev = positive_data[positive_train_end:positive_dev_end]
            negative_test = negative_data[negative_dev_end:]
            positive_test = positive_data[positive_dev_end:]

            train_data = np.concatenate([negative_train, positive_train])
            train_label = np.array([0] * len(negative_train) + [1] * len(positive_train))
            random_numbers = np.random.choice(list(range(len(train_data))), len(train_data), replace=False)
            train_data = train_data[random_numbers]
            train_label = train_label[random_numbers]
            dev_data = np.concatenate([negative_dev, positive_dev])
            dev_label = np.array([0] * len(negative_dev) + [1] * len(positive_dev))
            test_data = np.concatenate([negative_test, positive_test])
            test_label = np.array([0] * len(negative_test) + [1] * len(positive_test))

            train = np.concatenate([train_data, train_label[:, None]], axis=1)
            dev = np.concatenate([dev_data, dev_label[:, None]], axis=1)
            test = np.concatenate([test_data, test_label[:, None]], axis=1)
        # sample 
        if args.sample == "upsample":
            train = upsample(train, args.seed, args.sample_rate)
        elif args.sample == "downsample":
            train = downsample(train, args.seed, args.sample_rate)
        else:
            pass 
        # weight
        pos = train[:, -1].sum()
        neg = len(train) - pos 
        weight = {1: float(neg/pos)}
    print("#Train: %d, #Dev: %d, #Test: %d" % (len(train), len(dev), len(test)))
    print("#Train: %d, #Dev: %d, #Test: %d" % (train[:, -1].sum(), dev[:, -1].sum(), test[:, -1].sum()))

    return train, dev, test, weight


class DatasetForBF(Dataset):
    def __init__(self, args, data):
        self.data = data 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.data[index][:-1], dtype=torch.float32)
        y = torch.tensor(self.data[index][-1], dtype=torch.float32)
        return x, y


class DatasetForFilling(Dataset):
    def __init__(self, args, data, label=None):
        self.data = data 
        self.label = label 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index].to(torch.float32)
        if self.label is not None:
            y = self.label[index].to(torch.float32)
            return x, y
        else:
            return x 

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--imputer', default='simple')
    parser.add_argument('--sample', default='upsample')
    parser.add_argument('--sample_rate', default=2)
    args = parser.parse_args()

    split_data(args, './data/1year.arff', None, equal_split=True)