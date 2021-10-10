import nltk
import math


def perplexity(nn, test_data, model):
    print('calculating perplexity')
    test_ngrams = nltk.ngrams(test_data, nn, pad_right=True, pad_left=True, left_pad_symbol='<s>',
                              right_pad_symbol='</s>')
    test_ngrams = [ngram if (ngram in model) else '<UNK>' for ngram in test_ngrams]  # 把不在模型中的词组置为'<UNK>'
    probabilities = [model[ngram] for ngram in test_ngrams]
    N = len(probabilities)
    return math.exp((-1 / N) * sum(map(math.log, probabilities)))


def additive_smoothing(data, n, freq, delta):
    """
    加法平滑法，所有概率均加1
    Args:
        n: n_gram中的n
        ngrams: 所有n元组
        delta: 参数
        data: 语料库
        freq: 列表，词组出现的频率

    Returns:
        dict, 每个词组与其概率对应
        p_unk: 没出现过的词组的概率
    """
    V = len(data) + n - 1  # 词组总数, 因为两边都有padding
    n1_grams = nltk.ngrams(data, n - 1)
    n1_freq = nltk.FreqDist(n1_grams)

    def additive(gram, count):
        n1_gram = gram[:-1]
        n1_count = n1_freq[n1_gram]
        return (count + delta) / (n1_count + delta * V)
    model = {gram: additive(gram, count) for gram, count in freq.items()}
    model['<UNK>'] = 1/V
    return model


def GoodTuring_smooth(data, ngrams, n, freq, delta):
    """
    加法平滑法，所有概率均加1
    Args:
        n: n_gram中的n
        ngrams: 所有n元组
        delta: 参数
        data: 语料库
        freq: 列表，词组出现的频率

    Returns:
        dict, 每个词组与其概率对应
        p_unk: 没出现过的词组的概率
    """
    V = len(data) + n - 1  # 词组总数, 因为两边都有padding
    n1_grams = nltk.ngrams(data, n - 1)
    n1_freq = nltk.FreqDist(n1_grams)

    def additive(gram, count):
        n1_gram = gram[:-1]
        n1_count = n1_freq[n1_gram]
        return (count + delta) / (n1_count + delta * V)
    model = {gram: additive(gram, count) for gram, count in freq.items()}
    model['<UNK>'] = 1/V
    return model


def find_best_smooth(data, ngram, n, freq, test_data):
    min_ppl = float('inf')
    best_choice = 0
    best_param = 0
    for delta in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]:
        print('testing additive smoothing with delta({})...'.format(delta))
        lm = additive_smoothing(data, n, freq, delta)
        ppl = perplexity(n, test_data, lm)
        print('  ppl: ', ppl)
        if ppl < min_ppl:
            best_choice = 1
            best_param = delta
            min_ppl = ppl
    print('best choice: {} best parameter:{} best ppl: {}'.format(best_choice, best_param, min_ppl))
    return best_choice, best_param


def smooth(data, ngram, n, freq, choice, param):
    """
    平滑
    Args:
        data: 语料库
        ngram: 词组几何
        n: n_gram中的n
        freq: 频率分布
        choice: 进行哪种平滑，对应关系见find_best_smooth函数
        param: 用于平滑的参数，比如加法平滑中的delta

    Returns:

    """
    print('smoothing model ...')
    if choice == 1:
        return additive_smoothing(data, n, freq, param)
