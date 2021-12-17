from scipy.io import arff
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.impute import SimpleImputer, KNNImputer

class BankDataset():
    def __init__(self, data_path, train_ratio=0.9, test_ratio=0.1, standarize_method='knn') -> None:
        assert train_ratio + test_ratio == 1
        data = pd.DataFrame(arff.loadarff(data_path)[0])
        self.data = []
        self.label = []
        for id, row in data.iterrows():
            self.data.append(row.iloc[0:-1].to_numpy().astype(np.float))
            self.label.append(int(row.iloc[-1]))
        self.data = np.array(self.data)
        self.label = np.array(self.label)

        self.standarize(standarize_method)

        negative_data = self.data[self.label == 0]
        positive_data = self.data[self.label == 1]

        negative_train = negative_data[:int(train_ratio * len(negative_data))]
        positive_train = positive_data[:int(train_ratio * len(positive_data))]
        negative_test = negative_data[int(train_ratio * len(negative_data)):]
        positive_test = positive_data[int(train_ratio * len(positive_data)):]
        self.train_data = np.concatenate([negative_train, positive_train])
        self.train_label = np.array([0] * len(negative_train) + [1] * len(positive_train))
        random_numbers = choice(list(range(len(self.train_data))), len(self.train_data), replace=False)
        self.train_data = self.train_data[random_numbers]
        self.train_label = self.train_label[random_numbers]
        self.test_data = np.concatenate([negative_test, positive_test])
        self.test_label = np.array([0] * len(negative_test) + [1] * len(positive_test))

    def standarize(self, standarize_method):
        if standarize_method is None:
            return
        elif standarize_method == 'knn':
            standarizer = KNNImputer(copy=False)
        elif standarize_method == 'simple':
            standarizer = SimpleImputer(copy=False)
        standarizer.fit_transform(self.data)

if __name__ == '__main__':
    dataset = BankDataset('./data/1year.arff', standarize_method='knn')