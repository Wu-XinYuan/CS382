import argparse
from itertools import product
import nltk
import nltk.lm
from pathlib import Path
import os
from smooth import smooth, find_best_smooth, perplexity


def load_data(data_dir, file_name, nn):
    """
    加载数据并预处理，将出现频率少于n=3次的词替换为<UNK>
    Args:
        data_dir: relative or absolute path of dataset folder,
        file_name: 'train_set.txt' 或 'dev_set.txt' 或 'test_set.txt'
        nn: ngram中的n
    Returns:
        预处理后的数据
    """
    file_path = os.path.join(data_dir, file_name)
    with open(file_path) as f:
        data = [l.strip() for l in f.readlines()]
    data = ''.join(data).split(' ')
    print('loading data from {} with total {} words'.format(file_path, len(data)))
    for _ in range(nn-1):  # 手动padding
        data.insert(0, '<s>')
        data.append('</s>')
    frequency = nltk.FreqDist(data)
    return [word if frequency[word] >= 3 else '<UNK>' for word in data]


if __name__ == '__main__':
    # Load and prepare data
    data_path = r'dataset'
    n = 3
    trainData = load_data(data_path, 'train_set.txt', n)
    devData = load_data(data_path, 'dev_set.txt', n)
    testData = load_data(data_path, 'test_set.txt', n)
    print("=" * 50)
    n_grams = nltk.ngrams(trainData, n)
    vocab = nltk.lm.Vocabulary(trainData + ['<s>', '</s>'])
    freq = nltk.FreqDist(n_grams)
    smoothChoice, smoothParam = find_best_smooth(trainData, n_grams, n, freq, devData)
    freq_s = smooth(trainData, n_grams, n, freq, smoothChoice, smoothParam)
    ppl = perplexity(n, testData, freq_s)
    print('ppl:{}', ppl)

    print('=' * 50)
#    print(lm.perplexity.py(testData))
