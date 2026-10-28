# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

library(torch)
library(tidyverse)
setwd('~/data/cora/')

# 2. 读取 Content (节点特征与标签)
# 格式: <paper_id> <word_attributes>+ <class_label>
content <- read.table("cora.content", stringsAsFactors = FALSE)

# 提取 ID 映射
paper_ids <- content$V1
id_map <- setNames(1:length(paper_ids), paper_ids) # 将原始ID映射为 1..N 的索引

# 提取特征 (X)，最后一列是分类信息
features <- as.matrix(content[, 2:(ncol(content) - 1)])
# features <- torch_eye(nrow(content)) # 对比：仅考虑网络结构

# 提取标签 (Y) 并转为数值索引
labels_raw <- content[, ncol(content)]
labels <- as.numeric(as.factor(labels_raw))
num_classes <- length(unique(labels))

# 3. 读取 Cites (引用关系/边)
cites <- read.table("cora.cites", stringsAsFactors = FALSE)

# 将引用ID转换为索引
edges_source <- id_map[as.character(cites$V1)]
edges_target <- id_map[as.character(cites$V2)]

# 过滤掉可能存在的无法匹配的边 (Cora有时候会有悬空引用)
valid_idx <- !is.na(edges_source) & !is.na(edges_target)
edges_source <- edges_source[valid_idx]
edges_target <- edges_target[valid_idx]

num_nodes <- nrow(features)

# 4. 构建邻接矩阵 (Adjacency Matrix)
# 初始化全0矩阵
adj <- matrix(0, nrow = num_nodes, ncol = num_nodes)
# 填充边 (无向图通常设为双向)
adj[cbind(edges_source, edges_target)] <- 1
adj[cbind(edges_target, edges_source)] <- 1

# 5. 归一化处理 (Renormalization Trick)
# A_hat = A + I
adj_tilde <- adj + diag(num_nodes)

# 计算度矩阵 D 的逆平方根
row_sum <- rowSums(adj_tilde)
d_inv_sqrt <- diag(1 / sqrt(row_sum))

# result = D^(-1/2) * A_tilde * D^(-1/2)
adj_norm <- d_inv_sqrt %*% adj_tilde %*% d_inv_sqrt

# 转换为 Torch Tensors

features = torch_tensor(features, dtype = torch_float())
adj = torch_tensor(adj_norm, dtype = torch_float())
labels = torch_tensor(labels, dtype = torch_long())
num_features = ncol(features)


# 定义单层图卷积层
GCNLayer <- nn_module(
  "GCNLayer",
  initialize = function(in_features, out_features) {
    # 权重矩阵 W
    self$linear <- nn_linear(in_features, out_features, bias = FALSE)
  },
  
  forward = function(x, adj) {
    # 1. 线性变换 (H * W)
    support <- self$linear(x)
    # 2. 邻接矩阵传播 (A_hat * (H * W))
    output <- torch_matmul(adj, support)
    return(output)
  }
)

# 定义完整的 GCN 模型
GCN <- nn_module(
  "GCN",
  initialize = function(n_feat, n_hidden, n_class, dropout) {
    self$gc1 <- GCNLayer(n_feat, n_hidden)
    self$gc2 <- GCNLayer(n_hidden, n_class)
    self$dropout <- nn_dropout(p = dropout)
  },
  
  forward = function(x, adj) {
    # 第一层: GCN -> ReLU -> Dropout
    x <- self$gc1(x, adj)
    x <- torch_relu(x)
    x <- self$dropout(x)
    
    # 第二层: GCN
    x <- self$gc2(x, adj)
    
    # 输出层: Log Softmax
    return(x)
  }
)

# 选择部分节点为训练集，其余作为测试集。
# --- 配置参数 ---
set.seed(42)
hidden_units <- 16
dropout_rate <- 0.5
learning_rate <- 0.01
weight_decay <- 3e-3
epochs <- 100

# --- 划分数据集 ---
# 创建简单的掩码 (Masks)
train_indices <- sample(1:num_nodes, size = 140) # 标准Cora通常用140个训练样本
test_indices <- setdiff(1:num_nodes, train_indices)

# --- 初始化模型与优化器 ---
model <- GCN(
  n_feat = num_features,
  n_hidden = hidden_units,
  n_class = num_classes,
  dropout = dropout_rate
)

optimizer <- optim_adamw(model$parameters, lr = learning_rate, weight_decay = weight_decay)

# --- 训练循环 ---
for (epoch in 1:epochs) {
  model$train()
  optimizer$zero_grad()
  
  # 前向传播
  output <- model(features, adj)
  
  # 计算 Loss (只基于训练集节点)
  loss <- nnf_cross_entropy(output[train_indices, ], labels[train_indices])
  
  # 反向传播
  loss$backward()
  optimizer$step()
  
  # 每 20 个 epoch 打印一次状态
  if (epoch %% 10 == 0) {
    # 简单计算训练集准确率
    preds <- output[train_indices, ]$argmax(dim = 2)
    acc <- (preds == labels[train_indices])$sum()$item() / length(train_indices)
    cat(sprintf("Epoch: %03d | Loss: %.4f | Train Acc: %.4f\n", epoch, loss$item(), acc))
  }
}

# --- 测试评估 ---
model$eval() # 切换到评估模式 (关闭 Dropout)
with_no_grad({
  output <- model(features, adj)
  test_preds <- output[test_indices, ]$argmax(dim = 2)
  correct <- (test_preds == labels[test_indices])$sum()$item()
  test_acc <- correct / length(test_indices)
})

cat(sprintf("最终测试集准确率: %.2f%%\n", test_acc * 100))


library(Rtsne)
library(ggplot2)
library(showtext)
showtext_auto()

# 1. 获取模型对所有节点的输出 (Embedding)
model$eval()
final_output <- model(features, adj)

# 2. 转换为 R 的矩阵格式
# detach() 用于从计算图中分离，as_array 转为 R 数组
embeddings <- as_array(final_output$detach())

# 3. 运行 t-SNE (降维: 7维 -> 2维)
# perplexity 是 t-SNE 的关键参数，通常设为 30-50
cat("正在进行 t-SNE 降维...\n")
tsne_result <- Rtsne(embeddings, dims = 2, perplexity = 30, verbose = TRUE, check_duplicates = FALSE)

# 4. 构建绘图数据框
plot_data <- data.frame(
  X = tsne_result$Y[, 1],
  Y = tsne_result$Y[, 2],
  Label = as.factor(as_array(labels)), # 真实标签
  Type = ifelse(1:num_nodes %in% train_indices, "Train", "Test") # 标记是训练集还是测试集
)

# 5. 用 ggplot2 画图
ggplot(plot_data, aes(x = X, y = Y, color = Label)) +
  geom_point(alpha = 0.7, size = 1) +
  theme_minimal() +
  labs(
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    color = "Paper Class"
  ) +
  scale_color_brewer(palette = "Set1") +
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 14)
  )
