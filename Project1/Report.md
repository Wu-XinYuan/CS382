# 《自然语言处理》大作业1项目报告

邬心远

519021910604

## 项目简述

本项目是用python实现的ngram语言模型，利用了一些`nltk`包的语言统计功能。为了得到更好的模型，加入了加法平滑算法。

## 项目实现

### 数据预处理

为了使模型拥有更好的适应能力，在从文件读取词库后，预处理部分，将所有出现频率不到5次的词否替换为了符号`<UNK>`，实现改功能的核心代码为

`return [word if frequency[word] >= 5 else '<UNK>' for word in data]`

所有的训练、测试数据在进行操作前都要经过如上的处理。

### 平滑算法

项目中实现了加法平滑算法：具体的公式为$P=\frac{\delta+c(w_{i-n+1}^i)}{\delta|V| + \sum_{w_i}c(w_{i-n+1}^i)}$，其中，对于每个$w_{i-n+1}^{i-1}$如果不存在$w_i=\text{<UNK>}$存在，则在统计时加上这一条并假定它出现的频率为0。同时，增加一条单独的$\text{'UNK'}$记录为那些测试时可能出现的却从没有在训练集出现过的词条匹配，它的概率为$\frac{1}{|V|}$，其中$|V|$为此表的总量。

### 自适应选择参数

在加法平滑算法中，公式中的$\delta$选多少没有明确的定论，并且需要根据样本的大小做出调整，因此，项目中采用了尝试不同大小的$\delta$值，并且在交叉验证集上验证，利用困惑度评价模型，选择最好的模型应用于测试数据之上，根据实践结果，也发现$\delta$在0到1之间增大的过程中，困惑度先减小后增大，从而可以选择到最合适的参数。

## 项目结果：

在采用3元语法，的情况下，找到最适合的$\delta$是$10^{-6}$，在交叉集上的ppl为6000左右，而训练集上的ppl为4665.6

## 参考资料

https://github.com/joshualoehr/ngram-language-model参考了nltk库的用法

此外，感谢郭倩昀和陈畅同学在我完成作业的过程中为我解答了关于项目要求等方面的一些疑惑。