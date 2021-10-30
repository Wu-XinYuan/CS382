import pickle
import numpy as np


def viterbi(param, sentence):
    """
    viterbi算法计算最大可能性标注
    :param param: hmm_param,包括三个矩阵
    :param sentence: 待标注的句子
    :return: 标注结果，与sentence一样长的一维数组，0不断，1断
    """
    kinds = param['start_prob'].shape[0]  # 词性个数
    tags = np.zeros((kinds, len(sentence)), dtype=int)  # tag[i][j]保存j+1个字tag为i取到最大可能性时j的词性
    prob = np.zeros((kinds, len(sentence)))  # prob[i][j]保存到第j个字为止且第j个字词性为i的最大可能
    for i in range(kinds):
        prob[i, 0] = param['start_prob'][i] * param['emission_mat'][i, ord(sentence[0])]
    for j in range(1, len(sentence)):
        for i in range(kinds):
            for k in range(kinds):
                prob_tmp = prob[k, j - 1] * param['trans_mat'][k, i] * param['emission_mat'][i, ord(sentence[j])]
                if prob_tmp > prob[i, j]:
                    prob[i, j] = prob_tmp
                    tags[i, j] = k
    prob_max = 0
    result = [0 for _ in range(len(sentence))]
    for i in range(kinds):
        if prob[i, len(sentence) - 1] > prob_max:
            prob_max = prob[i, len(sentence) - 1]
            result[-1] = i
    # 回溯，寻找最大可能对应的标记链
    for i in range(len(sentence) - 1, 0, -1):
        result[i - 1] = tags[result[i], i]
    return result


def forward(param, sentence):
    """
    viterbi算法计算最大可能性标注
    :param param: hmm_param,包括三个矩阵
    :param sentence: 待计算概率的句子
    :return: 句子的概率
    """
    kinds = param['start_prob'].shape[0]  # 词性个数
    alpha = np.zeros((kinds, len(sentence)))  # alpha[i][j]保存alpha_j(i)
    # 初始化第一个字的概率
    for i in range(kinds):
        alpha[i, 0] = param['start_prob'][i] * param['emission_mat'][i, ord(sentence[0])]
    for t in range(1, len(sentence)):
        for j in range(kinds):
            for i in range(kinds):
                alpha[j, t] += alpha[i, t - 1] * param['trans_mat'][i, j]
            alpha[j, t] *= param['emission_mat'][j, ord(sentence[t])]
    print(alpha)
    prob = sum(alpha[:, -1])
    return prob


def backward(param, sentence):
    """
    后向算法计算句子可能性
    :param param: hmm_param,包括三个矩阵
    :param sentence: 待计算概率的句子
    :return: 句子的概率
    """
    kinds = param['start_prob'].shape[0]  # 词性个数
    beta = np.zeros((kinds, len(sentence)))  # alpha[i][j]保存alpha_j(i)
    # 初始化最后一个字的概率
    for i in range(kinds):
        beta[i, -1] = 1
    for t in range(len(sentence)-2, -1, -1):
        for j in range(kinds):
            for i in range(kinds):
                beta[j, t] += beta[i, t + 1] * param['trans_mat'][j, i] * param['emission_mat'][i, ord(sentence[t+1])]
    print(beta)
    prob = 0
    for i in range(kinds):
        prob += beta[i, 0] * param['start_prob'][i] * param['emission_mat'][i, ord(sentence[0])]
    return prob


def print_result(sentence, mask):
    for i in range(len(sentence)):
        print(sentence[i], end='')
        if mask[i] == 1:
            print('/', end='')
    print('')


with open("hmm_parameters.pkl", "rb") as f:
    hmm_param = pickle.load(f)
sentence_test = '邬心远是个好同学'
for character in sentence_test:
    print(character, ord(character), hmm_param['emission_mat'][0, ord(character)],
          hmm_param['emission_mat'][1, ord(character)])
result_viterbi = viterbi(hmm_param, sentence_test)
print('viterbi算法结果：')
print_result(sentence_test, result_viterbi)
print('前向算法概率：')
print(forward(hmm_param, sentence_test))
print('后向算法概率：')
print(backward(hmm_param, sentence_test))
