import sklearn
# from dataset import BankDataset
import sys
sys.path.append('..')
from argparse import ArgumentParser
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statistics import mean
import os
from functools import partial
from data_loader import split_data
import numpy as np

def prepare_model(args):
    if args.model == 'tree':
        model = DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,   
            min_samples_split=args.min_samples_split,
            max_features=args.max_features, 
            class_weight='balanced'
        )
    elif args.model == 'forest':
        model = RandomForestClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features, 
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            class_weight='balanced'
        )
    elif args.model == 'lr':
        model = LogisticRegressionCV(
            max_iter=10000,
            n_jobs=args.n_jobs,
        )
    return model

def get_data(args):
    train, dev, test, _ = split_data(args, args.data_path)
    train_data = train[:, :-1]
    train_label = train[:, -1]
    dev_data = dev[:, :-1]
    dev_label = dev[:, -1]
    test_data = test[:, :-1]
    test_label = test[:, -1]
    return train_data, train_label, dev_data, dev_label, test_data, test_label

def calculate_metric(pred, pred_prob, label):
    f1 = f1_score(label, pred, average='macro')
    roc_auc = roc_auc_score(label, pred_prob)
    acc = accuracy_score(label, pred)
    return f1, roc_auc, acc

def null_type(x, real_type):
    if x == 'None':
        return None
    else:
        return real_type(x)

def set_seed(seed):
    import random
    import numpy.random
    random.seed(seed)
    numpy.random.seed(seed)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['tree', 'forest', 'lr'], required=True)
    
    parser.add_argument('--criterion', type=str, choices=['gini', 'entropy'], default='gini')
    parser.add_argument('--max_depth', type=partial(null_type, real_type=int), default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--max_features', type=partial(null_type, real_type=str), default=None)

    parser.add_argument('--imputer', type=partial(null_type, real_type=str), default='simple')
    parser.add_argument('--sample', type=partial(null_type, real_type=str), default='upsample')
    parser.add_argument('--sample_rate', type=float, default=1)
    
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=None)

    parser.add_argument('--do_cross_val', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--output_path', type=str, default='log/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train_data, train_label, dev_data, dev_label, test_data, test_label = get_data(args)
    model = prepare_model(args)
    model.fit(train_data, train_label)
    pred = model.predict(dev_data)
    pred_prob = model.predict_proba(dev_data)[:, 1]
    f1, roc_auc, acc = calculate_metric(pred, pred_prob, dev_label)
    
    log_str = "Dev result. F1: %f, AUC: %f, Accuracy: %f" % (f1, roc_auc, acc)
    print(log_str)
    with open(os.path.join(args.output_path, 'result.txt'), 'w') as f:
        f.write(str(args) + '\n')
        f.write(log_str + '\n')
        
    if args.do_test:
        model.fit(np.concatenate([train_data, dev_data], axis=0), np.concatenate([train_label, dev_label], axis=0))
        pred = model.predict(test_data)
        pred_prob = model.predict_proba(test_data)[:, 1]
        f1, roc_auc, acc = calculate_metric(pred, pred_prob, test_label)
        print("Test F1:%f, Roc-Auc:%f, Accuracy:%f" % (f1, roc_auc, acc))
        with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
            f.write(str(args) + '\n')
            f.write("Test F1:%f, Roc-Auc:%f, Accuracy:%f\n" % (f1, roc_auc, acc))

if __name__ == '__main__':
    main()
