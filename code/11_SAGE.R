# GraphSAGE 的示例

# 在 GCN 中，节点及其邻居的特征是混合在一起通过对称归一化矩阵 $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$ 进行传播的。
# 而在 GraphSAGE（以最常见的均值聚合 Mean Aggregator 为例）中，其核心逻辑是：
# - 聚合 (Aggregate)：对邻居节点的特征求均值。
# - 拼接 (Concat)：将节点自身的特征与邻居的聚合特征拼接（Concatenate）在一起。
# - 变换 (Transform)：通过一个线性层进行特征变换。

# 公式表达为：

# $$
# h_v^{(l)} = W^{(l)} \cdot \text{CONCAT}\left(h_v^{(l-1)}, \text{MEAN}_{u \in N(v)}\left(h_u^{(l-1)}\right)\right)
# $$

library(torch)
library(tidyverse)
# 请确保你的工作目录正确指向了数据集所在位置
setwd('~/data/cora/') 

# 2. 读取 Content (节点特征与标签)
# 格式: <paper_id> <word_attributes>+ <class_label>
content <- read.table("cora.content", stringsAsFactors = FALSE)

# 提取 ID 映射
paper_ids <- content$V1
id_map <- setNames(1:length(paper_ids), paper_ids) # 将原始ID映射为 1..N 的索引

# 提取特征 (X)，最后一列是分类信息
features <- as.matrix(content[, 2:(ncol(content) - 1)])

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

# 5. 归一化处理 (GraphSAGE Mean Aggregation)
# 计算邻居的均值，使用行归一化矩阵 D^(-1) * A
row_sum <- rowSums(adj)
# 防止孤立节点导致除以 0
row_sum[row_sum == 0] <- 1 
d_inv <- diag(1 / row_sum)

# result = D^(-1) * A (行和为1，相当于对邻居特征求均值)
adj_mean <- d_inv %*% adj

# 转换为 Torch Tensors
features = torch_tensor(features, dtype = torch_float())
adj = torch_tensor(adj_mean, dtype = torch_float())
labels = torch_tensor(labels, dtype = torch_long())
num_features = ncol(features)


# --- 定义网络层与模型 ---

# 定义单层 GraphSAGE 层 (Mean Aggregator)
SAGELayer <- nn_module(
  "SAGELayer",
  initialize = function(in_features, out_features) {
    # 由于需要拼接自身特征与邻居特征，输入维度是 in_features * 2
    self$linear <- nn_linear(in_features * 2, out_features, bias = TRUE)
  },
  
  forward = function(x, adj) {
    # 1. 聚合邻居特征 (Mean Aggregation): H_neigh = A_mean * X
    neighbor_features <- torch_matmul(adj, x)
    
    # 2. 将自身特征和邻居特征拼接: [X || H_neigh]
    # dim=2 表示按特征维度拼接 (dim=1 为节点数量)
    combined_features <- torch_cat(list(x, neighbor_features), dim = 2)
    
    # 3. 线性变换
    output <- self$linear(combined_features)
    return(output)
  }
)

# 定义完整的 GraphSAGE 模型
GraphSAGE <- nn_module(
  "GraphSAGE",
  initialize = function(n_feat, n_hidden, n_class, dropout) {
    self$sage1 <- SAGELayer(n_feat, n_hidden)
    self$sage2 <- SAGELayer(n_hidden, n_class)
    self$dropout <- nn_dropout(p = dropout)
  },
  
  forward = function(x, adj) {
    # 第一层: SAGE -> ReLU -> Dropout
    x <- self$sage1(x, adj)
    x <- torch_relu(x)
    x <- self$dropout(x)
    
    # 第二层: SAGE
    x <- self$sage2(x, adj)
    
    # 输出层: (计算Loss时使用CrossEntropy，不需要显式加Softmax)
    return(x)
  }
)


# --- 配置参数与数据划分 ---
set.seed(42)
hidden_units <- 16
dropout_rate <- 0.5
learning_rate <- 0.01
weight_decay <- 3e-3
epochs <- 100

# 创建简单的掩码 (Masks)
train_indices <- sample(1:num_nodes, size = 140) # 标准Cora通常用140个训练样本
test_indices <- setdiff(1:num_nodes, train_indices)

# --- 初始化模型与优化器 ---
model <- GraphSAGE(
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
  
  # 每 10 个 epoch 打印一次状态
  if (epoch %% 10 == 0) {
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


# --- t-SNE 可视化 ---
library(Rtsne)
library(ggplot2)
library(showtext)
showtext_auto()

# 1. 获取模型对所有节点的输出 (Embedding)
model$eval()
final_output <- model(features, adj)

# 2. 转换为 R 的矩阵格式
embeddings <- as_array(final_output$detach())

# 3. 运行 t-SNE (降维: 7维 -> 2维)
cat("正在进行 t-SNE 降维...\n")
tsne_result <- Rtsne(embeddings, dims = 2, perplexity = 30, verbose = TRUE, check_duplicates = FALSE)

# 4. 构建绘图数据框
plot_data <- data.frame(
  X = tsne_result$Y[, 1],
  Y = tsne_result$Y[, 2],
  Label = as.factor(as_array(labels)), 
  Type = ifelse(1:num_nodes %in% train_indices, "Train", "Test") 
)

# 5. 用 ggplot2 画图
ggplot(plot_data, aes(x = X, y = Y, color = Label)) +
  geom_point(alpha = 0.7, size = 1) +
  theme_minimal() +
  labs(
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    title = "GraphSAGE Node Embeddings on Cora Dataset",
    color = "Paper Class"
  ) +
  scale_color_brewer(palette = "Set1") +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 14)
  )
