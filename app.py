import argparse
import numpy as np
import pandas as pd
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
import seaborn as sn


class DprepWrongFileFormatError(Exception):
    pass


class DprepIndexError(Exception):
    pass


class DprepClassNumberError(Exception):
    pass


class Dprep:
    @staticmethod
    def prepare_data(file, index):  # prepares data to further work
        try:
            data_frame = pd.read_csv(file, header=None)
            if data_frame.shape[1] <= 1:
                raise DprepWrongFileFormatError
            elif index >= data_frame.shape[1]:
                raise DprepIndexError
            data_frame[index] = pd.Categorical(data_frame[index]).codes  # categorical to numerical
            class_number = np.amax(data_frame[index].values) + 1  # number of classes in decisional attribute
            if class_number != 2:
                raise DprepClassNumberError
            train_data, test_data = skms.train_test_split(data_frame, random_state=45, train_size=0.75)  # split data
        except FileNotFoundError:
            print('{} no such file'.format(file))
        except pd.errors.EmptyDataError:
            print('{} is empty'.format(file))
        except DprepWrongFileFormatError:
            print('{} has a wrong format'.format(file))
        except DprepIndexError:
            print('Index={} is out of boundary'.format(index))
        except DprepClassNumberError:
            print('To low number of classes in decisional attribute')
        else:
            return train_data.to_numpy(), test_data.to_numpy()


class Logreg:  # logistic regression
    def __init__(self, train_data, index, theta=0.005):
        self._train_data = train_data
        self._index = index
        self._theta = theta
        self._weights = np.random.RandomState(45).normal(size=train_data.shape[1])

    def train(self, acc=0.95):  # training classificator
        correct_predictions = float('inf')
        while correct_predictions > int(self._train_data.shape[0] * (1 - acc)):
            netvals = self._net(self._train_data[:, :self._index:])
            output = self._output_signal(netvals)
            err = self._train_data[:, self._index] - output
            self._weights[0] += self._theta * err.sum()
            self._weights[1:] += self._theta * np.dot(self._train_data[:, :self._index:].T, err)
            correct_predictions = self._correct_predictions()

    def _net(self, vectors):  # returns net value
        return np.dot(vectors, self._weights[1:]) + self._weights[0]

    def _output_signal(self, netvals): # returns sigmoid's value
        return 1.0 / (1.0 + np.exp(-netvals))

    def predict(self, vectors):  # predicts class
        return np.where(self._net(vectors) >= 0.0, 1, 0)

    def _correct_predictions(self):  # returns number of correct predictions
        correct_predictions = (self.predict(self._train_data[:, :self._index:]) - self._train_data[:, self._index])
        correct_predictions[correct_predictions < 0] = 1
        return correct_predictions.sum()

    def confusion_matrix(self, vectors):  # returns confusion matrix
        predictions = self.predict(vectors[:, :self._index:])
        tp = predictions[predictions == 0].shape[0]
        fn = vectors[:, self._index][vectors[:, self._index] == 0].shape[0] - tp
        if fn < 0: fn = 0
        tn = predictions[predictions == 1].shape[0]
        fp = vectors[:, self._index][vectors[:, self._index] == 1].shape[0] - tn
        if fp < 0: fp = 0
        return np.array([[tp, fp], [fn, tn]])


def main(file, index):  # main body
    print('Preparing data...')
    train_data, test_data = Dprep.prepare_data(file, index)
    print('Data prepared')
    print('Training...')
    logreg = Logreg(train_data, index)
    logreg.train()
    print('Training finished')
    print('Creating confusion matrix...')
    confusion_matrix = logreg.confusion_matrix(test_data)
    mat = pd.DataFrame(confusion_matrix, index=['positive', 'negative'], columns=['positive', 'negative'])
    sn.heatmap(mat, annot=True)
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path)
    print('Confusion matrix saved to {}'.format(cm_path))


def parse_arguments():  # parsing arguments
    parser = argparse.ArgumentParser(description='Logistic regression classification for two classes')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Input file containing numerical data matrix without headers')
    parser.add_argument('-i', '--index', type=int, required=True, help='Index of decisional attribute in data matrix')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.file, args.index)
