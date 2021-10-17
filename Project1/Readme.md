# Readme

本项目是用python实现的ngram语言模型，加入了加法平滑算法。主要实现基于`nltk`的语言统计功能。

文件结构如下：

```
├─dataset 保存数据集文件夹
|	├─ train_set.txt 训练文本
|	├─ dev_set.txt 交叉验证文本
|	└─ test_set.txt 测试文本
├─smooth.py 实现了平滑算法
└─model.py 实现了语言模型
```

如果要运行语言模型，在终端输入`python3 model.py -n %num%` 其中num是规定在ngram中的n为多少，在不指定的情况下默认为3。

程序执行时，会自动利用交叉验证文本在1e-9到1中选择最佳的参数用于加法平滑，并在测试文本上测试困惑度。如果运行正常，将得到类似下面的输出：

```
loading data from dataset\train_set.txt with total 13604165 words
loading data from dataset\dev_set.txt with total 1700521 words
loading data from dataset\test_set.txt with total 1700521 words
==================================================
testing additive smoothing with delta(1e-09)...
  ppl:  18698.307375004457
testing additive smoothing with delta(1e-08)...
  ppl:  11167.521219768421
testing additive smoothing with delta(1e-07)...
  ppl:  7180.712440432768
testing additive smoothing with delta(1e-06)...
  ppl:  5970.755969140392
testing additive smoothing with delta(1e-05)...
  ppl:  7609.070620072362
testing additive smoothing with delta(0.0001)...
  ppl:  15109.231272592677
testing additive smoothing with delta(0.001)...
  ppl:  42579.40462761079
testing additive smoothing with delta(0.01)...
  ppl:  152002.7081203332
testing additive smoothing with delta(0.1)...
  ppl:  594985.5359197085
testing additive smoothing with delta(1)...
  ppl:  2141330.68388231
testing Good-Turing smoothing...
best choice: 1 best parameter:1e-06 best ppl: 5970.755969140392
running on test...
ppl:{} 4665.5853378627
==================================================
```

