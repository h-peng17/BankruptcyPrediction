from scipy.io import arff
import pandas as pd


def load_and_fill_na(year):
    data, meta = arff.loadarff('../data/{}year.arff'.format(year))
    df = pd.DataFrame(data)

    for index, column in df.iteritems():
        column.fillna(column.mean(), inplace=True)

    return df


if __name__ == '__main__':
    load_and_fill_na(1)
