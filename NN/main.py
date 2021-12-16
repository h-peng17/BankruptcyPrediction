
import os
import pdb
import sys
sys.path.append("..")
import json
import time
import random
import time
import logging
import argparse
from collections import Counter
from pathlib import Path
from sklearn.utils import class_weight

from tqdm import trange
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from model import MLPForBF
from data_loader import split_data, DatasetForBF


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, model, train_dataloader, dev_dataloader, test_dataloader):
    # total step
    step_tot = len(train_dataloader) * args.max_epoches
    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    writter = SummaryWriter()
    for i in range(args.max_epoches):
        crr, tot = 0, 0
        for batch in train_dataloader:
            batch = [o.cuda() for o in batch]
            inputs = {
                "x":batch[0],
                "y":batch[1],
            }
            model.train()
            loss, preds = model(**inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            preds = preds.cpu().detach().numpy()
            label = inputs["y"].cpu().detach().numpy()
            crr += (preds == label).sum()
            tot += label.shape[0]
            sys.stdout.flush()
            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr/tot))

            writter.add_scalar("Loss", loss.item(), global_step)
            writter.add_scalar("Accuracy on train", crr/tot, global_step)

        # dev
        with torch.no_grad():
            print("deving....")
            model.eval()            
            score = eval_F1(args, model, dev_dataloader)
            writter.add_scalar("Macro F1 on test", score, i)
            if score > best_dev_score:
                best_dev_score = score
                best_test_score = eval_F1(args, model, test_dataloader)
                print("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
            else:
                print("Dev score: %.3f" % score)
    print("@RESULT: Best Dev score is %.3f, Test score is %.3f\n" %(best_dev_score, best_test_score))


def eval_F1(args, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        batch = [o.cuda() for o in batch]
        inputs = {
            "x": batch[0],
            "y": batch[1],
        }
        _, preds = model(**inputs)
        tot_label.extend(inputs["y"].cpu().detach().tolist())
        tot_output.extend(preds.cpu().detach().tolist())
    f1 = f1_score(tot_label, tot_output, average="macro")     
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognize")
    parser.add_argument("--input_file", dest="input_file", type=str,
                        default="", help="input file")


    parser.add_argument("--model_type", dest="model_type", type=str,
                        default="cnn", help="model type")
    parser.add_argument("--loss_fn", dest="loss_fn", type=str,
                        default="sigmoid", help="loss_fn loss_fn")

    parser.add_argument("--max_epoches", dest="max_epoches", type=int, 
                        default=3, help="max epoch")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=64, help="batch size pre gpu")

    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=160,help='hidden size')
    parser.add_argument("--num_classes", dest="num_classes", type=int,
                        default=50,help='number of classes')
    
    parser.add_argument("--optim", dest="optim", type=str,
                        default="adam", help="optim")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,
                        default=3e-5, help='learning rate')
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    args = parser.parse_args()
    print(args)
    # set seed
    set_seed(args)    
    train_set, dev_set, test_set, weight = split_data(args.input_file)

    if args.model_type == "svm":
        clf = SVC(kernel="poly", class_weight=weight, degree=2, probability=True, random_state=args.seed)
        train_set = np.concatenate([train_set, dev_set], axis=0)
        clf.fit(train_set[:, :-1], train_set[:, -1])
        y = clf.predict(test_set[:, :-1])
        xx = clf.decision_function(test_set[:, :-1])
        # pdb.set_trace()
        acc = (test_set[:, -1] == y).sum() / len(y)
        macro_f1 = f1_score(test_set[:, -1], y, average="macro")     
        print("Accuracy: %.4f, Macro-F1: %.4f" % (acc, macro_f1))
    elif args.model_type == "lr":
        clf = LogisticRegression(class_weight=weight, random_state=args.seed, max_iter=500)
        train_set = np.concatenate([train_set, dev_set], axis=0)
        clf.fit(train_set[:, :-1], train_set[:, -1])
        y = clf.predict(test_set[:, :-1])
        acc = (test_set[:, -1] == y).sum() / len(y)
        macro_f1 = f1_score(test_set[:, -1], y, average="macro")     
        print("Accuracy: %.4f, Macro-F1: %.4f" % (acc, macro_f1))
    else:
        train_set = DatasetForBF(args, train_set)
        dev_set = DatasetForBF(args, dev_set)
        test_set = DatasetForBF(args, test_set)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=0)        
        dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=0)
        
        if args.model_type == "mlp":
            model = MLPForBF(args, weight)
        else:
            raise ValueError
        model.cuda()
        train(args, model, train_dataloader, dev_dataloader, test_dataloader)
    
    


