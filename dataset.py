import os

import pickle
from math import ceil
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.DEBUG)
# import tensorflow as tf
import matplotlib.pyplot as plt


def plot_sample_denormed(X, y, plot: bool=True) -> None:
    y = denormalize_pos(y)
    plot_sample(X, y, plot=plot)

def plot_sample(X, y, plot: bool=True):
    # logging.debug("labels looks like this {}".format(denormalize_pos(y)))
    n_item = len(y)
    xs: np.ndarray = y[0::2]
    ys: np.ndarray = y[1::2]
    if plot:
        img = X.reshape(96, 96)
        plt.imshow(img, cmap='gray')
        '''
            sort of scatter points
        '''
        plt.scatter(xs, ys, marker="x")

        '''
            showing picture
        '''
        plt.show()

def plot_testing_sample(X, plot: bool=True):
    # logging.debug("labels looks like this {}".format(denormalize_pos(y)))
    n_item = len(X)
    if plot:
        img = X.reshape(96, 96)
        plt.imshow(img, cmap='gray')
        '''
            showing picture
        '''
        plt.show()



def plot_sample(X, y, plot: bool=True):
    # logging.debug("labels looks like this {}".format(denormalize_pos(y)))
    n_item = len(y)
    xs: np.ndarray = y[0::2]
    ys: np.ndarray = y[1::2]
    if plot:
        img = X.reshape(96, 96)
        plt.imshow(img, cmap='gray')
        '''
            sort of scatter points
        '''
        plt.scatter(xs, ys, marker="x")

        '''
            showing picture
        '''
        plt.show()


def normalize_img(img_arr: np.ndarray) -> np.ndarray:
    X = img_arr / 255.  # scale pixel values to [0, 1]
    return X

def denormalize_img(img_arr: np.ndarray) -> np.ndarray:
    X = img_arr * 255.  # scale pixel values to [0, 255]
    return X

def normalize_pos(pos: np.ndarray) -> np.ndarray:
    X = (pos - 48) / 48
    return X

def denormalize_pos(pos: np.ndarray) -> np.ndarray:
    X = pos * 48 + 48
    return X

# class kaggle_face_dataset(Dataset):

class kaggle_face_dataset(Dataset):
    def __init__(self, data_dir: str, batch_size: int, cache: bool=True, 
            cache_location: str="ds_cache", test=False):
        self.train_file = os.path.join(data_dir, 'training.csv')
        self.test_file = os.path.join(data_dir, 'test.csv')
        self.lookup_file = os.path.join(data_dir, 'IdLookupTable.csv')
        saved_file = os.path.join(cache_location, "ds_cache.pkl")
        saved_test_file = os.path.join(cache_location, "ds_test_cache.pkl")
        self.batch_size = batch_size

        if not test and cache and os.path.exists(saved_file):
            logging.debug("cached hit, reading from cache")
            with open(saved_file, 'rb') as f:
                self.X, self.y, self.X_ori, self.y_ori = pickle.load(f)
        elif test and cache and os.path.exists(saved_test_file):
            logging.debug("cached hit, reading from cache")
            with open(saved_test_file, 'rb') as f:
                self.X, self.y, self.X_ori, self.y_ori = pickle.load(f)
        else:
            self.X, self.y, self.X_ori, self.y_ori = self.load(test)

        self.n_samples = len(self.X)
        self.inds = [i for i in range(self.n_samples)]
        random.shuffle(self.inds)
        if not test and cache and not os.path.exists(saved_file):
            logging.debug("saving cache at location {}".format(saved_file))
            os.makedirs(cache_location, exist_ok=True)
            with open(saved_file, 'wb') as f:
                pickle.dump((self.X, self.y, self.X_ori, self.y_ori), f)
                logging.debug("ds saved to {}".format(saved_file))
        elif test and cache and not os.path.exists(saved_test_file):
            logging.debug("saving cache at location {}".format(saved_test_file))
            os.makedirs(cache_location, exist_ok=True)
            with open(saved_test_file, 'wb') as f:
                pickle.dump((self.X, self.y, self.X_ori, self.y_ori), f)

    def __len__(self):
        return ceil(self.n_samples/self.batch_size)

    def __get_item__(self, ind: int):
        return self.X[ind], self.y[ind]

    def __iter__(self):
        self.cur_ind = 0
        return self

    def __next__(self):
        if self.cur_ind > self.n_samples:
            return StopIteration
        to_ret_inds = self.inds[self.cur_ind: self.cur_ind+self.batch_size]
        self.cur_ind += self.batch_size

        # batch_size_ = min(self.n_samples, 
        #         self.cur_ind+self.batch_size) - self.cur_ind
        reshaped_X = np.reshape(self.X[to_ret_inds], 
                (-1, 1, 96, 96))
        return reshaped_X, self.y[to_ret_inds]

    def _ind_yielder(self):
        inds = [i for i in range(self.n_samples)]
        random.shuffle(inds)
        cur_ind = 0
        yield inds[cur_ind: cur_ind + self.batch_size]

    def load(self, test: bool=False, cols=None):
        fname = self.test_file if test else self.train_file
        df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

        # The Image column has pixel values separated by space; convert
        # the values to numpy arrays:
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]

        # logging.debug("num entries\n {}".format(df.count()))
        logging.debug("entry sample\n {}".format(df[:2]))
        df = df.dropna()

        X_ori = np.vstack(df['Image'].values)
        X = X_ori / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)

        if not test:  # only self.train_file has any target columns
            y_ori = df[df.columns[:-1]].values
            y = (y_ori - 48) / 48  # scale target coordinates to [-1, 1]
            X, y = shuffle(X, y, random_state=42)  # shuffle train data
            y = y.astype(np.float32)
        else:
            y = None
            y_ori = None

        return X, y, X_ori, y_ori
