# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

library(torch)
library(tidyverse)
setwd('~/data/cora/') # 保持你的路径设置

# ==========================================
# 1. 数据准备
# ==========================================
content <- read.table("cora.content", stringsAsFactors = FALSE)

paper_ids <- content$V1
id_map <- setNames(1:length(paper_ids), paper_ids)

features <- as.matrix(content[, 2:(ncol(content) - 1)])

labels_raw <- content[, ncol(content)]
labels <- as.numeric(as.factor(labels_raw))
num_classes <- length(unique(labels))

cites <- read.table("cora.cites", stringsAsFactors = FALSE)

edges_source <- id_map[as.character(cites$V1)]
edges_target <- id_map[as.character(cites$V2)]

valid_idx <- !is.na(edges_source) & !is.na(edges_target)
edges_source <- edges_source[valid_idx]
edges_target <- edges_target[valid_idx]

num_nodes <- nrow(features)

adj <- matrix(0, nrow = num_nodes, ncol = num_nodes)
adj[cbind(edges_source, edges_target)] <- 1
adj[cbind(edges_target, edges_source)] <- 1

adj_binary <- adj + diag(num_nodes)

# 转换为 Tensors
features = torch_tensor(features, dtype = torch_float())
adj = torch_tensor(adj_binary, dtype = torch_float())
labels = torch_tensor(labels, dtype = torch_long())
num_features = ncol(features)


# ==========================================
# 2. 模型定义
# ==========================================
GATLayer <- nn_module(
  "GATLayer",
  initialize = function(in_features, out_features, dropout, alpha = 0.2) {
    self$in_features <- in_features
    self$out_features <- out_features
    self$alpha <- alpha
    self$W <- nn_linear(in_features, out_features, bias = FALSE)
    self$a_src <- nn_parameter(torch_empty(out_features, 1))
    self$a_dst <- nn_parameter(torch_empty(out_features, 1))
    nn_init_xavier_uniform_(self$W$weight)
    nn_init_xavier_uniform_(self$a_src)
    nn_init_xavier_uniform_(self$a_dst)
    self$leakyrelu <- nn_leaky_relu(self$alpha)
    self$dropout <- nn_dropout(dropout)
  },
  
  forward = function(h, adj, return_attention = FALSE) {
    wh <- self$W(h)
    attn_src <- torch_matmul(wh, self$a_src) 
    attn_dst <- torch_matmul(wh, self$a_dst) 
    
    scores <- attn_src + attn_dst$t()
    scores <- self$leakyrelu(scores)
    
    zero_vec <- -1e9 * torch_ones_like(scores)
    attention <- torch_where(adj > 0, scores, zero_vec)
    
    attention <- nnf_softmax(attention, dim = 2)
    attention <- self$dropout(attention)
    
    h_prime <- torch_matmul(attention, wh)
    
    if (return_attention) {
      return(list(output = h_prime, attention = attention))
    } else {
      return(h_prime)
    }
  }
)

GAT <- nn_module(
  "GAT",
  initialize = function(n_feat, n_hidden, n_class, dropout, alpha = 0.2, n_heads = 8) {
    self$dropout <- dropout
    self$heads <- nn_module_list(lapply(1:n_heads, function(i) {
      GATLayer(n_feat, n_hidden, dropout, alpha)
    }))
    self$out_att <- GATLayer(n_hidden * n_heads, n_class, dropout, alpha)
  },
  
  forward = function(x, adj, return_attention = FALSE) {
    x <- nnf_dropout(x, p = self$dropout, training = self$training)
    
    head_outs <- list()
    for (i in 1:length(self$heads)) {
      head_outs[[i]] <- self$heads[[i]](x, adj)
    }
    
    x <- torch_cat(head_outs, dim = 2)
    x <- nnf_elu(x)
    x <- nnf_dropout(x, p = self$dropout, training = self$training)
    
    # 直接返回原始 Logits
    if (return_attention) {
      out_res <- self$out_att(x, adj, return_attention = TRUE)
      return(list(
        output = out_res$output, 
        attention = out_res$attention
      ))
    } else {
      x <- self$out_att(x, adj)
      return(x) 
    }
  }
)


# ==========================================
# 3. 训练配置与执行
# ==========================================
set.seed(42)
n_hidden <- 8       
n_heads <- 8        
dropout_rate <- 0.6 
learning_rate <- 0.001 
weight_decay <- 5e-4
epochs <- 300

train_indices <- sample(1:num_nodes, size = 140)
test_indices <- setdiff(1:num_nodes, train_indices)

model <- GAT(
  n_feat = num_features,
  n_hidden = n_hidden,
  n_class = num_classes,
  dropout = dropout_rate,
  n_heads = n_heads
)

optimizer <- optim_adamw(model$parameters, lr = learning_rate, weight_decay = weight_decay)

cat("开始 GAT 训练...\n")

for (epoch in 1:epochs) {
  model$train()
  optimizer$zero_grad()
  
  # output 现在是原始的 Logits
  output <- model(features, adj)
  
  # 使用 nnf_cross_entropy_loss 处理 Logits 和 Labels
  loss <- nnf_cross_entropy(output[train_indices, ], labels[train_indices])
  
  loss$backward()
  optimizer$step()
  
  if (epoch %% 20 == 0) {
    # 验证准确率：在 Logits 上直接取最大值的索引，等同于在概率上取最大值
    preds <- output[train_indices, ]$argmax(dim = 2)
    acc <- (preds == labels[train_indices])$sum()$item() / length(train_indices)
    cat(sprintf(
      "Epoch: %03d | Loss: %.4f | Train Acc: %.4f\n",
      epoch,
      loss$item(),
      acc
    ))
  }
}

# --- 测试评估 ---
model$eval()
with_no_grad({
  output <- model(features, adj)
  test_preds <- output[test_indices, ]$argmax(dim = 2)
  correct <- (test_preds == labels[test_indices])$sum()$item()
  test_acc <- correct / length(test_indices)
})

cat(sprintf("GAT 最终测试集准确率: %.2f%%\n", test_acc * 100))



## 可视化

# --- 提取注意力矩阵 (画图前的必备桥梁) ---
model$eval()
with_no_grad({
  # 触发 forward 中的 return_attention = TRUE 分支
  out_list <- model(features, adj, return_attention = TRUE)
  
  # 取出注意力张量，剥离计算图，并转换为普通的 R 矩阵
  attn_tensor <- out_list$attention
  attn_matrix <- as.matrix(attn_tensor)
})

# 检查一下维度，应该是一个 2708 x 2708 的方阵
print(dim(attn_matrix))

library(igraph)
library(ggraph)

target_node <- 1619
node_labels <- labels_raw

# --- 1. 寻找 1跳 (1-Hop) 和 2跳 (2-Hop) 邻居 ---
# 找到 1 跳邻居 (排除自己)
hop1_neighbors <- which(adj_binary[target_node, ] > 0 & (1:num_nodes) != target_node)

# 找到 2 跳邻居
hop2_neighbors <- c()
for (n in hop1_neighbors) {
  n_neighbors <- which(adj_binary[n, ] > 0)
  hop2_neighbors <- c(hop2_neighbors, n_neighbors)
}
# 去重，并排除目标节点和 1 跳节点，保持层级清晰
hop2_neighbors <- unique(hop2_neighbors)
hop2_neighbors <- setdiff(hop2_neighbors, c(target_node, hop1_neighbors))

if(length(hop1_neighbors) == 0) stop("节点没有邻居")

# --- 2. 构建边数据框 (Edge Dataframe) ---
# (a) Target -> 1-Hop 的边
edges_hop1 <- data.frame(
  from = as.character(target_node),
  to = as.character(hop1_neighbors),
  weight = attn_matrix[target_node, hop1_neighbors],
  hop_level = "1-Hop Edges"
)

# (b) 1-Hop -> 2-Hop 的边
edges_hop2_list <- list()
for (n in hop1_neighbors) {
  valid_targets <- which(adj_binary[n, ] > 0)
  targets <- intersect(valid_targets, hop2_neighbors)
  if(length(targets) > 0) {
    edges_hop2_list[[as.character(n)]] <- data.frame(
      from = as.character(n),
      to = as.character(targets),
      weight = attn_matrix[n, targets],
      hop_level = "2-Hop Edges"
    )
  }
}
edges_hop2 <- do.call(rbind, edges_hop2_list)

# 合并所有边
edge_df <- rbind(edges_hop1, edges_hop2)

# --- 3. 构建节点数据框 (Node Dataframe) ---
nodes_involved <- unique(c(target_node, hop1_neighbors, hop2_neighbors))

# 为节点打上层级标签
node_type <- rep("2-Hop Node", length(nodes_involved))
node_type[nodes_involved == target_node] <- "Target Node"
node_type[nodes_involved %in% hop1_neighbors] <- "1-Hop Node"

node_df <- data.frame(
  name = as.character(nodes_involved),
  label = node_labels[nodes_involved],
  node_type = factor(node_type, levels = c("Target Node", "1-Hop Node", "2-Hop Node"))
)

# --- 4. 创建 igraph 对象并绘图 ---
g <- graph_from_data_frame(d = edge_df, vertices = node_df, directed = TRUE)

# 使用 ggraph 绘图
ggraph(g, layout = "kk") + 
  # 画边：根据层级区分颜色，依然用粗细和透明度表示 Attention
  geom_edge_link(aes(edge_width = weight, edge_alpha = weight, color = hop_level), 
                 arrow = arrow(length = unit(2.5, 'mm')), 
                 end_cap = circle(4, 'mm')) +
  
  # 画节点：根据类别上色，根据层级决定大小和形状
  geom_node_point(aes(color = label, size = node_type, shape = node_type)) +
  
  # 边的映射设置
  scale_edge_width(range = c(0.3, 2.5)) +
  scale_edge_alpha(range = c(0.3, 1)) +
  scale_edge_color_manual(values = c("1-Hop Edges" = "steelblue", "2-Hop Edges" = "gray60")) +
  
  # 节点的映射设置 (目标最大实心，1跳中等实心，2跳较小空心/实心)
  scale_size_manual(values = c("Target Node" = 8, "1-Hop Node" = 5, "2-Hop Node" = 3.5)) +
  scale_shape_manual(values = c("Target Node" = 19, "1-Hop Node" = 19, "2-Hop Node" = 17)) + # 19=圆, 17=三角
  
  # 节点 ID 文本
  geom_node_text(aes(label = name), repel = TRUE, size = 3, color = "black") +
  
  theme_void() +
  labs(
    color = "Paper Class",
    size = "Network Role",
    shape = "Network Role",
    edge_color = "Connection Type"
  )


## 附录
## 挑选有趣的节点

# --- 寻找能够讲出“好故事”的黄金异配性节点 ---

# 1. 获取模型的最终预测结果 (确保前向传播已经执行过)
model$eval()
with_no_grad({
  output <- model(features, adj)
  preds <- output$argmax(dim = 2)
  # 转换为 R 的向量格式，并对齐 1-based 索引
  preds_vec <- as.integer(as.matrix(preds)) 
})

# 2. 创建不带自环的邻接矩阵
adj_no_self <- adj_binary
diag(adj_no_self) <- 0

# 3. 准备容器存储统计信息 (新增了预测和注意力指标)
node_stats <- data.frame(
  node_id = 1:num_nodes,
  true_label = labels_raw,
  pred_label = labels_raw[preds_vec], # 将预测的数字索引转回标签名称
  is_correct = FALSE,
  degree = rowSums(adj_no_self),
  num_neighbor_classes = 0, 
  same_class_ratio = 0,
  max_attention = 0,    # 分配给单个邻居的最大注意力 (越高说明越果断)
  attention_sd = 0      # 注意力分配的标准差 (越大说明线条粗细越分明)
)

# 4. 遍历提取信息
for (i in 1:num_nodes) {
  # 记录是否预测正确
  node_stats$is_correct[i] <- node_stats$true_label[i] == node_stats$pred_label[i]
  
  neighbors <- which(adj_no_self[i, ] > 0)
  if (length(neighbors) > 0) {
    neighbor_labels <- labels_raw[neighbors]
    node_stats$num_neighbor_classes[i] <- length(unique(neighbor_labels))
    node_stats$same_class_ratio[i] <- sum(neighbor_labels == labels_raw[i]) / length(neighbors)
    
    # 获取该节点分配给 1-hop 邻居的注意力权重
    # 注意：这里我们提取你之前算好的 attn_matrix
    att_weights <- attn_matrix[i, neighbors]
    node_stats$max_attention[i] <- max(att_weights)
    node_stats$attention_sd[i] <- sd(att_weights)
  }
}

# 5. 见证奇迹的时刻：筛选“黄金节点”
golden_nodes <- node_stats %>%
  filter(
    degree >= 3,                  # 邻居数量适中，画出来的图不至于太拥挤或太单调
    same_class_ratio <= 0.35,     # 环境极其恶劣，同类极少
    is_correct == TRUE,           # 【关键】GAT 必须能力挽狂澜预测正确！
    !is.na(attention_sd)          # 排除没有邻居计算不了标准差的节点
  ) %>%
  arrange(desc(attention_sd), desc(max_attention)) # 按照注意力的“果断程度”降序排列

# 打印出最适合写进书里的 Top 10 节点
print(head(golden_nodes, 10))

