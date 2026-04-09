
# 数据集说明

本书使用的数据有三种获取方式：

1. 内置 R 包：例如 `ames` 数据集位于 `modeldata` 包中，对应章节会有详细说明。
2. 通过 `torchvision` 包获取：例如 `CIFAR-10` 等计算机视觉数据集，对应章节会有使用方法。
3. 需要单独下载的数据集：以下内容详细记录了这些数据集的下载地址和使用说明。

> 若遇到数据集下载问题，请通过邮件联系作者。

## 第 7 章 卷积网络

### 1. MNIST 数据集

- 数据集描述：包含 60,000 张 28×28 像素的灰度图像，用于训练和测试手写数字识别模型。
- 数据标签：每个图像标记为 0-9 之间的数字。
- 下载地址：[Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- 文件说明：
  - `train-images-idx3-ubyte.gz`：训练集图像 (9,912,422 bytes)
  - `train-labels-idx1-ubyte.gz`：训练集标签 (28,881 bytes)
  - `t10k-images-idx3-ubyte.gz`：测试集图像 (1,648,877 bytes)
  - `t10k-labels-idx1-ubyte.gz`：测试集标签 (4,542 bytes)

### 第 8 章 循环网络

#### 1. 北京 PM2.5 数据集

- 数据集描述：包含 2010 年至 2014 年北京每小时的 PM2.5 浓度数据。
- 本地路径：`bookdata` 目录下，文件名为 `PRSA_data_2010.1.1-2014.12.31.csv`
- 替代下载：[UCI Machine Learning Repository](https://archive.ics.uci.edu)，搜索关键词：`Beijing PM2.5`

#### 2. IMDB 电影评论数据集

- 数据集描述：包含 50,000 条明显两极分化（正面/负面）的电影评论，广泛用于二分类情感分析任务。数据平均分为训练集（25,000条）和测试集（25,000条），且每个集合中正负样本各占一半。此外，还包含 50,000 条未标记的评论用于无监督学习。
- 下载地址：[Stanford AI Lab - Andrew Maas](https://ai.stanford.edu/~amaas/data/sentiment/)
- 文件信息：`aclImdb_v1.tar.gz`（解压后包含 `aclImdb` 文件夹）

> 数据结构：原始数据存储为文本文件（每个评论一个 `.txt` 文件），分布在 `train/pos`、`train/neg`、`test/pos`、`test/neg` 文件夹中。

#### 3. GloVe 预训练词向量

- 本书使用版本：`glove.2024.wikigiga.50d.zip`
- 下载地址：[Stanford NLP Group - GloVe](https://nlp.stanford.edu/projects/glove/)
- 可用版本：除 50d 版本外，还提供 100d、200d、300d 版本的预训练词向量。
- 数据集规模：包含 400,000 个单词的预训练词向量，通过在大规模文本语料库上使用无监督学习训练得到，能够捕捉单词之间的语义关系。

### 第 9 章 Transformer

#### 科幻电影评论合成数据集

- 数据集描述：为演示 LoRA 逻辑生成的 730 条伪造的 AI 风格科幻电影评论数据。
- 数据样例：
```r
[1] "The quantum simulation scenes were mind-blowing, but the AI antagonist lacked true neural depth."      
[2] "A cyberpunk masterpiece that questions the very nature of our virtual reality and human consciousness."
[3] "This movie proves that the singularity is near, featuring the most realistic cyborgs on screen."   
```
- 本地路径：`bookdata` 目录下的 `scifi_synthetic.csv.zip`

### 第 10 章 深度推荐模型

#### MovieLens 1M 数据集

- 数据集来源：由 GroupLens 研究组（明尼苏达大学）收集并公开。
- 数据集规模：包含 6,040 名用户对约 3,900 部电影的 1,000,209 条评分。
- 用户信息：所有用户均为 2000 年加入 MovieLens 平台的匿名用户，每位用户至少提供了 20 条评分。
- 文件结构：
  - `ratings.dat` 格式：`UserID::MovieID::Rating::Timestamp`
  - `users.dat` 格式：`UserID::Gender::Age::Occupation::Zip-code`
  - `movies.dat` 格式：`MovieID::Title::Genres`
- 下载地址：[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)


### 第 11 章 图神经网络

#### 1. Cora 数据集

- 用途：常用于图神经网络和节点分类任务。
- 文件结构：
  - `cora.content`：每行包含论文 ID、1,433 个词特征、类别标签
  - `cora.cites`：每行 `被引论文ID 引用论文ID`，表示引用关系
- 获取方式：
  - 本地路径：`bookdata` 目录下，文件名为 `cora.tgz`
  - 下载地址：[Cora 数据集](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

#### 2. ESOL 数据集

- 数据集描述：包含 1,128 个分子样本，每个样本都提供了分子的 SMILES 表示式以及对应的实验测得水溶性值 (LogS)。
- 特点：数据集规模适中、标注可靠，常被用作分子性质预测的基准测试数据。
- 本地路径：`bookdata` 目录下，文件名为 `delaney-processed.csv.zip`

### 第 12 章 变分自编码器

#### ECG5000 数据集

- 数据集描述：包含 5,000 条心电图（ECG）信号，每条信号长度为 140 个时间步（微观视角下单次心跳周期的波形高频切片）。数据集中包含了正常的规律心跳，也包含了心律不齐、心肌梗死等多种异常信号。
- 数据分布：
  - 正常样本（Class 1）：约 2,900 条
  - 异常样本（Class 2-5）：约 2,100 条（包含 R-on-T 室性早搏等多种心脏异常）
- 本地路径：`bookdata` 目录下，文件名为 `ECG5000.zip`

### 第 13 章 GAN

#### 数据集说明

- 数据集名称：待补充
- 数据集描述：待补充
- 下载地址：待补充
- 本地路径：待补充

> 请参考对应章节的具体说明或联系作者获取详细信息。