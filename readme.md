


本书的数据有三种形式访问：

1. 内置的 R 包。比如 `ames` 数据集在 `modeldata` 包内，对应章节会有提及。
2. 通过 `torchvision` 包获取。比如 `CIFAR-10` 等，对应章节会有提及。
3. 需要额外单独下载的数据集。

下面的内容记录了额外单独下载的数据集地址。如果无法下载数据，请发邮件联系我。

## 第 7 章 卷积网络

**1. MNIST 数据集**

- 包含了 60,000 张 28x28 像素的灰度图像，用于训练和测试手写数字识别模型。
- 每个图像都被标记为 0 到 9 之间的一个数字。
- 地址：[Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

各列说明：

- train-images-idx3-ubyte.gz: training set images (9912422 bytes)
- train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

## 第 8 章 循环网络

**1. 北京 PM2.5 数据**

- 该数据集包含了 2010 年至 2014 年北京每个小时的 PM2.5 浓度数据。
- 位于 `bookdata` 目录下，文件名：`PRSA_data_2010.1.1-2014.12.31.csv`
- 或者在 [UCI Machine Learning Repository](https://archive.ics.uci.edu)，搜索关键词：`Beijing PM2.5`

**2. IMDB 电影评论数据集**

- 该数据集包含了：50,000 条极其明显的两极分化（正面/负面）的电影评论，广泛用于二分类情感分析任务。数据被平均分为训练集（25,000条）和测试集（25,000条），且每个集合中正负样本各占一半。此外，还包含 50,000 条未标记的评论用于无监督学习。
- 地址：[Stanford AI Lab - Andrew Maas](https://ai.stanford.edu/~amaas/data/sentiment/)
- 文件名： `aclImdb_v1.tar.gz` (解压后包含 `aclImdb` 文件夹)

> 原始数据存储为文本文件（每个评论一个 `.txt` 文件），分布在 `train/pos`, `train/neg`, `test/pos`, `test/neg` 文件夹中。

**3. GloVe 预训练词向量**

- 本书使用的 GloVe 预训练词向量文件为 `glove.2024.wikigiga.50d.zip`
- 地址：[Stanford NLP Group - GloVe](https://nlp.stanford.edu/projects/glove/)
- 除了 50d 版本，GloVe 还提供了 100d、200d、300d 版本的预训练词向量。
- 该数据集包含了 400,000 个单词的预训练词向量，每个向量的维度为 50、100、200 或 300。这些向量是通过在大规模文本语料库上使用无监督学习训练得到的，能够捕捉到单词之间的语义关系。

## 第 9 章 Transformer

为了演示 LoRA 逻辑生成的 730 条伪造的 AI 风格的数据。

```r
[1] "The quantum simulation scenes were mind-blowing, but the AI antagonist lacked true neural depth."      
[2] "A cyberpunk masterpiece that questions the very nature of our virtual reality and human consciousness."
[3] "This movie proves that the singularity is near, featuring the most realistic cyborgs on screen."   
```

对应文件为 `bookdata` 目录的 `scifi_synthetic.csv.zip`。

## 第 10 章 深度推荐模型

本数据集由 GroupLens 研究组（明尼苏达大学）收集并公开，包含 6,040 名用户 对 约 3,900 部电影 的 1,000,209 条评分。所有用户均为 2000 年加入 MovieLens 平台的匿名用户，每位用户至少提供了 20 条评分。

拥有三个文件：

- `ratings.dat` 格式：`UserID::MovieID::Rating::Timestamp`
- `users.dat` 格式：`UserID::Gender::Age::Occupation::Zip-code`
- `movies.dat` 格式：`MovieID::Title::Genres`

下载地址：[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)


## 第 11 章 图神经网络

**1. Cora 数据集**

常用于图神经网络和节点分类任务，文件结构:

- `cora.content`：每行包含论文 ID、1,433 个词特征、类别标签
- `cora.cites`：每行 `被引论文ID 引用论文ID`，表示引用关系

位于 `bookdata` 目录下，文件名 `cora.tgz`，或 [这里](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

**2. ESOL 数据集**

该数据集包含 1128 个分子样本，每个样本都提供了分子的 SMILES 表示式以及对应的实验测得水溶性值 (LogS)。
由于数据集规模适中、标注可靠，它常被用作分子性质预测的基准测试数据。

位于 `bookdata` 目录下，文件名 `delaney-processed.csv.zip`

## 第 12 章 变分自编码器

ECG5000 数据集包含了5000 条心电图（ECG）信号，每条信号长度为 140 个时间步（微观视角下单次心跳周期的波形高频切片）。数据集中包含了正常的规律心跳，也包含了心律不齐、心肌梗死等多种异常信号。

- 正常样本（Class 1）：约 2900 条。
- 异常样本（Class 2-5）：约 2100 条（包含 R-on-T 室性早搏等多种心脏异常）。

位于 `bookdata` 目录下，文件名 `ECG5000.zip`

## 第 13 章 GAN

数据地址。