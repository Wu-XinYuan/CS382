import argparse
from itertools import product
import nltk
import nltk.lm
from pathlib import Path
import os
from smooth import smooth, find_best_smooth, perplexity


def load_data(data_dir):
    """
    Args:
        data_dir: relative or absolute path of dataset folder,
        which include three .txt file: 'train_set.txt', 'dev_set.txt', 'test_set.txt'
    Returns:
        three part of data in the form of list
    """
    print('loading data ...')
    train_path = os.path.join(data_dir, 'train_set.txt')
    with open(train_path) as f:
        train_data = [l.strip() for l in f.readlines()]
    train_data = ''.join(train_data).split(' ')
    print('loading train data from {} with total {} words'.format(train_path, len(train_data)))

    dev_path = os.path.join(data_dir, 'dev_set.txt')
    with open(dev_path) as f:
        dev_data = [l.strip() for l in f.readlines()]
    dev_data = ''.join(dev_data).split(' ')
    print('loading dev data from {} with total {} words'.format(dev_path, len(dev_data)))

    test_path = os.path.join(data_dir, 'test_set.txt')
    with open(test_path, 'r') as f:
        test_data = [l.strip() for l in f.readlines()]
    test_data = ''.join(test_data).split(' ')
    print('loading train data from {} with total {} words'.format(test_path, len(test_data)))
    print('loading done!!!')

    return train_data, dev_data, test_data


if __name__ == '__main__':
    # Load and prepare data
    data_path = r'./dataset'
    trainData, devData, testData = load_data(data_path)
    print("=" * 50)
    n = 3
    n_grams = nltk.ngrams(trainData, n, pad_right=True, pad_left=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
    vocab = nltk.lm.Vocabulary(trainData + ['<s>', '</s>'])
    freq = nltk.FreqDist(n_grams)
    smoothChoice, smoothParam = find_best_smooth(trainData, n_grams, n, freq, devData)
    freq_s = smooth(trainData, n_grams, n, freq, smoothChoice, smoothParam)
    ppl = perplexity(n, testData, freq_s)
    print('ppl:{}', ppl)


    print('=' * 50)
#    print(lm.perplexity.py(testData))
