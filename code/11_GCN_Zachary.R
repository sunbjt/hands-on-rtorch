library(torch)
library(igraph)
set.seed(42)
torch_manual_seed(42)

# --- 1. Zachary's Karate Club ---
g <- make_graph("Zachary")
A_matrix <- as_adjacency_matrix(g, sparse = FALSE)
A <- torch_tensor(A_matrix, dtype = torch_float())

# --- 2. 核心公式预处理: D^{-1/2} * A_hat * D^{-1/2} ---

num_nodes <- length(V(g))
I <- torch_eye(num_nodes)
A_hat <- A + I # 添加自环，保留自身特征

# 计算度矩阵 D_hat (按行求和)
D_hat_diag <- torch_sum(A_hat, dim = 2)

# 计算 D^{-1/2}
# 注意：先计算 -0.5 次方
D_inv_sqrt_diag <- torch_pow(D_hat_diag, -0.5)
# 将向量转回对角矩阵
D_inv_sqrt <- torch_diag(D_inv_sqrt_diag)

# D^{-1/2} %*% A_hat %*% D^{-1/2}
L_sym <- torch_mm(torch_mm(D_inv_sqrt, A_hat), D_inv_sqrt)

# --- 3. 定义 GCN 模型 ---
GCN <- nn_module(
  "GCN",
  initialize = function(in_feat, hidden_feat, out_feat) {
    # 两个线性层 (对应公式中的 W)
    self$layer1 <- nn_linear(in_feat, hidden_feat)
    self$layer2 <- nn_linear(hidden_feat, out_feat)
  },
  
  forward = function(x, adj) {
    # --- 第一层 GCN ---
    x <- self$layer1(x) # 线性变换 XW
    x <- torch_mm(adj, x) # 向外走一步，直接的邻居
    x <- torch_relu(x)
    
    # --- 第二层 GCN ---
    x <- self$layer2(x)  # 线性变换
    x <- torch_mm(adj, x) # 向外走两步，邻居的邻居
    return(x) # 输出层直接输出 Logits
  }
)

# --- 4. 训练设置 ---

# 特征矩阵 X:
# 因为 Karate Club 没有节点特征，我们使用单位矩阵 (One-Hot编码)
X <- torch_eye(num_nodes)

# 真实标签 (Ground Truth): 
# 1 = Mr. Hi (教练), 2 = John A (主管)
manual_labels <- c(
  1, 1, 1, 1, 1, 1, 1, 1, 2, 2, # 1-10号
  1, 1, 1, 1, 2, 2, 1, 1, 2, 1, # 11-20号
  2, 1, 2, 2, 2, 2, 2, 2, 2, 2, # 21-30号
  2, 2, 2, 2                    # 31-34号
)

# 转换为 torch 张量
labels <- torch_tensor(manual_labels, dtype = torch_long())

# 初始化模型
# 输入 34维 -> 隐藏层 4维 -> 输出 2维 (两个阵营)
model <- GCN(in_feat = 34, hidden_feat = 4, out_feat = 2)
optimizer <- optim_adam(model$parameters, lr = 0.01)
criterion <- nn_cross_entropy_loss()

# --- 5. 训练循环 (Semi-Supervised) ---
for (epoch in 1:100) {
  optimizer$zero_grad()
  
  # 前向传播
  output <- model(x = X, adj = L_sym)
  
  # === 关键点：只计算两个节点的 Loss ===
  # 我们假装只知道 Node 1 (索引1) 和 Node 34 (索引34) 的身份
  # Node 1 是教练 (类别1), Node 34 是主管 (类别2)
  # 掩码索引:
  target_idx <- c(1, 34)
  
  # 只取出这两个节点的预测结果和真实标签来计算误差
  loss <- criterion(output[target_idx], labels[target_idx])
  
  loss$backward()
  optimizer$step()
  
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d | Loss: %.4f\n", epoch, loss$item()))
  }
}

# --- 6. 结果可视化与验证 ---
final_output <- model(x = X, adj = L_sym)
predictions <- torch_argmax(final_output, dim = 2)

cat("预测结果 (0=Mr.Hi, 1=John A):\n")
print(as.numeric(predictions))

# 简单计算准确率 (和全部真实标签对比)
as.numeric(predictions) == as.numeric(labels)
nnf_softmax(final_output, dim = 2)[3,]

# 结果绘图
library(ggplot2)
library(ggrepel)

# 1. 数据准备
emb_mat <- as.matrix(final_output$detach())
true_labels <- as.numeric(labels)

plot_df <- data.frame(
  Dim1 = emb_mat[, 1], # Mr. Hi 得分
  Dim2 = emb_mat[, 2], # John A 得分
  Faction = factor(true_labels, levels = c(1, 2), labels = c("Mr. Hi", "John A")),
  NodeID = 1:34
)

# 2. 绘图
ggplot(plot_df, aes(x = Dim1, y = Dim2, color = Faction)) +
  geom_point(size = 4, alpha = 0.8) +
  geom_text_repel(
    aes(label = NodeID),
    size = 3.5,
    fontface = "bold",
    box.padding = 0.3,   # 标签周围留白
    point.padding = 0.3, # 标签与点的距离
    max.overlaps = Inf,  # 强制显示所有标签
    show.legend = FALSE
  ) +
  scale_color_manual(values = c("Mr. Hi" = "#F8766D", "John A" = "#00BFC4")) +
  theme_minimal() +
  labs(
    title = "Karate Club GCN Embedding",
    x = "Logits for Mr. Hi (Dim 1)",
    y = "Logits for John A (Dim 2)"
  )
## -------------------------------------------- ##
## igraph 绘图
## -------------------------------------------- ##

library(showtext)
showtext_auto()

# 我们复用之前的 labels 向量 (如果不在这里，请重新运行上一段代码的 labels 定义)
node_colors <- ifelse(as.numeric(labels) == 1, "gray20", "tomato")

# 设置颜色
V(g)$color <- node_colors

# 设置标签颜色 (为了在深色点上能看清，设为白色或浅色)
V(g)$label.color <- "white"

# --- 3. 特殊标记“首领” ---
# 把 1号 (Mr. Hi) 和 34号 (John A) 搞得大一点，或者换个形状
# 这样一眼就能看出这两个“种子节点”
V(g)$size <- 12 # 默认大小
V(g)$size[c(1, 34)] <- 15 # 首领变大

V(g)$shape <- "circle"
V(g)$shape[c(1, 34)] <- "square" # 首领变成方形

# --- 4. 绘制 ---
# 使用 Fruchterman-Reingold 布局算法 (最经典的力导向布局)
# set.seed 确保每次画出来的布局方向一致
set.seed(42) 

plot(g, 
     layout = layout_with_fr(g), 
     main = "Zachary 空手道俱乐部",
     vertex.label.font = 2,      # 字体加粗
     vertex.label.cex = 0.8      # 字体大小
)

# 添加图例
legend("bottomleft", 
       legend = c("Mr. Hi 派系", "John A 派系"), 
       col = c("gray20", "tomato"), 
       pch = 15, # 方块图标
       pt.cex = 2,
       bty = "n") # 不画边框

## --------------------------- ##
##  A 进行 SVD 分解
svd_res <- svd(as.matrix(A)) # 转为普通 matrix 计算
svd_embeddings <- svd_res$u[, 1:2]
df_svd <- data.frame(
  Dim1 = svd_embeddings[, 1],
  Dim2 = svd_embeddings[, 2],
  Faction = as.factor(as.numeric(labels)), # 复用之前的标签
  NodeID = 1:34
)

library(ggplot2)
library(ggrepel)

ggplot(df_svd, aes(x = Dim1, y = Dim2, color = Faction, label = NodeID)) +
  geom_point(size = 4, alpha = 0.8) + 
  geom_text_repel(
    size = 4, 
    show.legend = FALSE, 
    max.overlaps = Inf,   # 强制显示所有标签，不隐藏任何点
    box.padding = 0.5     # 给标签一点额外的排斥空间
  ) +
scale_color_manual(values = c("1" = "gray20", "2" = "tomato")) +
  theme_minimal() +
  labs(title = "",
       subtitle = "",
       x = "Singular Vector 1", y = "Singular Vector 2")

## ------------------------------------------- ##
# 邻接矩阵 (Adjacency Matrix) 可视化
A_mat <- as.matrix(A) # 转为 R 标准矩阵
A_vis <- t(A_mat)[, ncol(A_mat):1] # image() 默认是从左下角开始画的。
par(mar = c(3, 3, 3, 1))
image(1:34, 1:34, A_vis,
      # 颜色设置: 0=白色, 1=海军蓝
      col = c("white", "navyblue"), 
      axes = FALSE, 
      main = "",
      xlab = "Target Node",
      ylab = "Source Node"
)
box()
# 添加网格线 (Grid)
grid(nx = 34, ny = 34, col = "gray90", lty = 1)
# X轴: 1 到 34
axis(1, at = 1:34, labels = 1:34, cex.axis = 0.6, tick = FALSE, line = -0.5)
# Y轴: 34 到 1 (注意：因为我们翻转了矩阵，所以Y轴标签要反着标，才能对应视觉上的"从上到下")
axis(2, at = 1:34, labels = 34:1, cex.axis = 0.6, tick = FALSE, las = 2, line = -0.5)

