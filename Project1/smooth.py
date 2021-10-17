import nltk
import math
import itertools

use_n1 = 0


def substitute(ngram, model, n):
    """
    如果三元组在语言模型中不存在，用<UNK>进行适当代替，如果一个词替换为<UNK>后在语言模型中，则将其中一个词替换，
    否则，接着尝试测试两个词、三个词
    Args:
        ngram: 等待被替换的测试集的n元词组
        model: 语言模型
        n: ngram中的n

    Returns:
        被替换后的ngram
    """
    global use_n1
    ngram_sub = ngram[:-1]+('<UNK>',)
    if ngram in model:
        return ngram
    if ngram_sub in model:
        use_n1 += 1
        return ngram_sub
    else:
        return 'UNK'  # 从未出现过的词组


def perplexity(n, test_data, model):
    # print('calculating perplexity')
    test_ngrams = nltk.ngrams(test_data, n)
    global use_n1
    use_n1 = 0
    test_ngrams = [substitute(ngram, model, n) for ngram in test_ngrams]  # 把不在模型中的词组置为'<UNK>'
    probabilities = [model[ngram] for ngram in test_ngrams]
    N = len(probabilities)
    return math.exp((-1 / N) * sum(map(math.log, probabilities)))


def additive_smoothing(data, n, freq, delta):
    """
    加法平滑法，所有概率均加1
    Args:
        n: n_gram中的n
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
        if n1_count == 0:
            print('frequency of', n1_gram, 'is 0')
        return (count + delta) / (n1_count + delta * V)

    model = {gram: additive(gram, count) for gram, count in freq.items()}
    global cnt
    for gram, freq in n1_freq.items():  # 对于三元组（a,b,c)如果(a,b)出现过，那么概率就是delta/(P(ab)+delta*v)
        if gram+('<UNK>',) not in model:
            model[gram+('<UNK>',)] = delta / (freq + delta * V)
    model['UNK'] = 1/V  # 那些频率为0的项的概率
    return model


def goodTuring_smooth(data, n, freq):
    """
    Good-Turing平滑法，原来出现r次的语句变成(r+1)*n_(r+1)/n_r次,
    因为n_r可能为0, 所以这里做出一些变化，改为原来出现r次的语句变成s*n_s/n_r次,s为从r+1往上数第一个满足n_s不为0的数
    Args:
        freq: 列表，词组出现的频率
        data: 语料库
        n: n_gram中的n

    Returns:
        dict, 每个词组与其概率对应
    """
    V = len(data) + n - 1  # 词组总数, 因为两边都有padding
    freq['<UNK>'] = 0
    grams = list(freq)
    fre = list(freq.values())
    values = list(set(fre))
    values.sort(reverse=True)  # 所有可能的r从大到小排列
    s = values[0]+1
    s_cnt = 0
    for r in values:
        r_cnt = fre.count(r)
        r_new = s * s_cnt / r_cnt
        fre = [r_new if val == r else val for val in fre]
        s, s_cnt = r, r_cnt
    fre = [nr/V for nr in fre]
    return dict(zip(grams, fre))


def find_best_smooth(data, ngram, n, freq, test_data):
    """

    Args:
        data:
        ngram:
        n:
        freq:
        test_data:

    Returns:
        ppl最小的方案：
            1 -- additive
            2 -- Good-Turing
    """
    min_ppl = float('inf')
    best_choice = 0
    best_param = 0
    for delta in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]:
        print('testing additive smoothing with delta({})...'.format(delta))
        lm = additive_smoothing(data, n, freq, delta)
        ppl = perplexity(n, test_data, lm)
        print('  ppl: ', ppl)
        if ppl < min_ppl:
            best_choice = 1
            best_param = delta
            min_ppl = ppl

    print('testing Good-Turing smoothing...')
    # lm = goodTuring_smooth(data, n, freq)
    # ppl = perplexity(n, test_data, lm)
    # print('  ppl: ', ppl)
    # if ppl < min_ppl:
    #     best_choice = 2
    #     min_ppl = ppl

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
    print('running on test...')
    if choice == 1:
        return additive_smoothing(data, n, freq, param)
    if choice == 2:
        return goodTuring_smooth(data, n, freq)
