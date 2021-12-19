from naive_bayes.dataloader import load_and_fill_na

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np


class NaiveBayesProcessor(object):
    def __init__(self, args, train_set, test_set):
        self.classifier_type = args.nb_classifier

        self.train, self.test = train_set, test_set
        self.pos_means, self.pos_var = None, None
        self.neg_means, self.neg_var = None, None
        self.pos_prob, self.neg_prob = None, None
        if self.classifier_type == 'gaussian':
            self.init_gaussian_train_stats()
        elif self.classifier_type == 'bernoulli':
            self.train_means = None
            self.pos_cond_prob, self.neg_cond_prob = None, None
            self.init_bernoulli_train_stats()

    def init_bernoulli_train_stats(self, mode='mean'):
        pos = self.train.loc[self.train['class'] == 1]
        pos = pos.drop(['class'], axis=1).to_numpy(dtype=np.float32)

        neg = self.train.loc[self.train['class'] == 0]
        neg = neg.drop(['class'], axis=1).to_numpy(dtype=np.float32)

        self.pos_prob = np.log(len(pos) / len(self.train))
        self.neg_prob = np.log(len(neg) / len(self.train))

        if mode == 'zero':
            pos = (pos > 0.0).astype(dtype=np.float32)
            neg = (neg > 0.0).astype(dtype=np.float32)
        else:
            if mode == 'mean':
                self.train_means = np.mean(self.train.drop(['class'], axis=1).to_numpy(dtype=np.float32), axis=0)
            elif mode == 'median':
                self.train_means = np.median(self.train.drop(['class'], axis=1).to_numpy(dtype=np.float32), axis=0)
            pos = (pos > self.train_means).astype(dtype=np.float32)
            neg = (neg > self.train_means).astype(dtype=np.float32)
        pos_cnt = np.sum(pos, axis=0) / len(pos)
        neg_cnt = np.sum(neg, axis=0) / len(neg)
        self.pos_cond_prob = np.stack([1 - pos_cnt, pos_cnt], axis=0)
        self.neg_cond_prob = np.stack([1 - neg_cnt, neg_cnt], axis=0)

    def init_gaussian_train_stats(self):
        pos = self.train.loc[self.train['class'] == 1]
        pos = pos.drop(['class'], axis=1)
        self.pos_means, self.pos_var = pos.mean(0), pos.std(0)
        neg = self.train.loc[self.train['class'] == 0]
        neg = neg.drop(['class'], axis=1)
        self.neg_means, self.neg_var = neg.mean(0), neg.std(0)
        self.pos_prob = np.log(len(pos) / len(self.train))
        self.neg_prob = np.log(len(neg) / len(self.train))

    def bernoulli_prob(self, x):
        threshold = 0.0
        if self.train_means is not None:
            threshold = self.train_means
        x = x.to_numpy(dtype=np.float32)
        pos = (x > threshold).astype(dtype=np.int32)
        pos_prob = sum(np.log(self.pos_cond_prob[idx, fidx] + 1e-6) for fidx, idx in enumerate(pos.tolist())) + self.pos_prob
        neg = (x > threshold).astype(dtype=np.int32)
        neg_prob = sum(np.log(self.neg_cond_prob[idx, fidx] + 1e-6) for fidx, idx in enumerate(neg.tolist())) + self.neg_prob

        _pos_prob = np.exp(pos_prob - np.max([pos_prob, neg_prob]))
        _neg_prob = np.exp(neg_prob - np.max([pos_prob, neg_prob]))
        summed = _pos_prob + _neg_prob

        return _pos_prob / summed, _neg_prob / summed

    def gaussian_prob(self, x):
        sqr1 = (x - self.pos_means) * (x - self.pos_means)
        sqr2 = self.pos_var * self.pos_var
        sqr1, sqr2 = sqr1.to_numpy(dtype=np.float32), sqr2.to_numpy(dtype=np.float32)
        pos_probs = np.exp(- sqr1 / (2 * sqr2)) / np.sqrt(2 * sqr2 * np.pi)
        pos_prob = np.log(pos_probs + 1e-6).sum() + self.pos_prob

        sqr1 = (x - self.neg_means) * (x - self.neg_means)
        sqr2 = self.neg_var * self.neg_var
        sqr1, sqr2 = sqr1.to_numpy(dtype=np.float32), sqr2.to_numpy(dtype=np.float32)
        neg_probs = np.exp(- sqr1 / (2 * sqr2)) / np.sqrt(2 * sqr2 * np.pi)
        neg_prob = np.log(neg_probs + 1e-6).sum() + self.neg_prob

        _pos_prob = np.exp(pos_prob - np.max([pos_prob, neg_prob]))
        _neg_prob = np.exp(neg_prob - np.max([pos_prob, neg_prob]))
        summed = _pos_prob + _neg_prob

        return _pos_prob / summed, _neg_prob / summed

    def predict(self):
        scores, predictions, labels = [], [], []
        for index, row in self.test.iterrows():
            label = row['class']
            labels.append(int(label))
            if self.classifier_type == 'gaussian':
                pred = self.gaussian_prob(row.drop(['class'], axis=0))
            else:
                pred = self.bernoulli_prob(row.drop(['class'], axis=0))
            scores.append(pred[0])
            predictions.append(int(pred[0] > 0.5))

        return predictions, scores


if __name__ == '__main__':
    processor = NaiveBayesProcessor(year=5)
    processor.predict()
