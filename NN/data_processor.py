
from enum import Flag
import os
import pdb
import sys
from typing import List
sys.path.append("..")
import json
import time
import random
import time
import logging
import argparse
from collections import Counter
from pathlib import Path
from sklearn.utils import class_weight, shuffle

from tqdm import trange
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter 

from model import MLPForFilling
from data_loader import select_col, DatasetForFilling


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def fill_empty(args, all_data, max_epoches=10):
    train_data, train_label, test_data, test_idx, column_idx = select_col(all_data)
    while train_data is not None:
        model = MLPForFilling(args, train_data.shape[1]).cuda()
        train_dataloader = DataLoader(DatasetForFilling(args, train_data, train_label), batch_size=64, shuffle=False)
        test_dataloader = DataLoader(DatasetForFilling(args, test_data), batch_size=64, shuffle=False)
        # total step
        step_tot = len(train_dataloader) * max_epoches
        # optimizer
        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise ValueError
        # print("We will train model in %d steps" % step_tot)
        print("Train for column: %d" % column_idx)
        global_step = 0
        for i in range(max_epoches):
            for batch in train_dataloader:
                batch = [o.cuda() for o in batch]
                inputs = {
                    "x":batch[0],
                    "y":batch[1],
                }
                model.train()
                loss = model(**inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                sys.stdout.flush()
                sys.stdout.write("epoch: %d, loss: %.6f\r" % (i, loss))
        all_preds = []
        for batch in test_dataloader:
            inputs = {
                "x": batch.cuda(),
            }
            model.eval()
            preds = model(**inputs).cpu().detach().tolist()
            all_preds.extend(preds)
        assert len(all_preds) == len(test_idx)
        column_idx = torch.tensor([column_idx]*len(test_idx), dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        all_data[test_idx, column_idx] = torch.tensor(all_preds)
        assert torch.isnan(torch.tensor(all_preds)).sum() == 0
        train_data, train_label, test_data, test_idx, column_idx = select_col(all_data)
        print("-"*50)
    return all_data
    
