from scipy.io import arff
import pandas as pd
import numpy as np
from numpy.random import choice
from argparse import ArgumentParser
import os
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm, trange
from pd_to_arff import pandas2arff

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data_path, 'clean')):
        os.makedirs(os.path.join(args.data_path, 'clean'))
    file_list = os.listdir(args.data_path)
    for file in tqdm(file_list):
        if os.path.isdir(os.path.join(args.data_path, file)):
            continue
        data_path = os.path.join(args.data_path, file)
        data = pd.DataFrame(arff.loadarff(data_path)[0])
        data = data.drop('Attr37', axis=1)
        data['class'] = data['class'].astype('int')

        # Index of columns that have missing values
        is_na = data.isna()
        na_num = is_na.sum(axis=0)
        na_col_index = np.where(na_num != 0)[0]
        na_col_index = na_col_index[np.argsort(na_num[na_col_index])]
        # for col in na_col_index:
        progress = trange(len(na_col_index))
        while na_col_index.size != 0:
            droped_data = data.drop(data.columns[na_col_index], axis=1)
            droped_data = droped_data.iloc[:, :-1]
            col = na_col_index[0]
            na_row_index = np.where(is_na.iloc[:, col])[0]
            train_row_index = np.array([i for i in range(len(data)) if i not in set(na_row_index)])

            test_data = droped_data.iloc[na_row_index]
            train_data = droped_data.iloc[train_row_index]
            train_label = data.iloc[train_row_index, col]
            model = RandomForestRegressor()
            try:
                model.fit(train_data, train_label)
                pred = model.predict(test_data)
                data.iloc[na_row_index, col] = pred
                na_col_index = na_col_index[1:]
            except:
                import pdb; pdb.set_trace()
            progress.update()
            break
        # pandas2arff(data, os.path.join(args.data_path, 'clean', file))