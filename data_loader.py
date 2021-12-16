import os 
import pdb 
import json 
import random 

from typing import Dict, Tuple, List

import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 

from scipy.io import arff
from sklearn.preprocessing import StandardScaler


def standardize(data):
    feature = data[:, :-1]
    label = data[:, -1:]
    norm_feature = StandardScaler().fit_transform(feature)
    data = torch.tensor(np.concatenate((norm_feature, label), axis=-1)).nan_to_num()
    return np.array(data)


def split_data(input_file: str, 
                train_rate: float=0.8, 
                dev_rate: float=0.1, 
                test_rate: float=0.1, 
                norm: bool=True,
                seed: int=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    # load and shuffle
    random.seed(seed)
    data, meta = arff.loadarff(input_file)
    data = data.tolist()
    random.shuffle(data)
    data = np.array(data, dtype=np.float32)
    # split
    assert int(train_rate+dev_rate+test_rate)== 1
    end_of_train = int(train_rate * len(data))
    end_of_dev = int((train_rate+dev_rate) * len(data))
    train = standardize(data[:end_of_train])
    dev = standardize(data[end_of_train:end_of_dev])
    test = standardize(data[end_of_dev:])
    # weight
    pos = data[:, -1].sum()
    neg = len(data) - pos 
    weight = {1: float(neg/pos)}
    # logging
    print(weight)
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






