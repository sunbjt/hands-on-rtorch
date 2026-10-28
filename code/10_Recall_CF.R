# ==============================================================================
# 基于 R Matrix & data.table 的 Item-CF 完整流水线
# ==============================================================================
library(Matrix)

# 载入数据预处理环节
source('~/github/rtorchbook/code/10_Leave_One_Out.R')

# 使用 dt_train_raw (隐式反馈，所有交互视为 1)
# 使用 sparseMatrix 极大降低内存占用并加速矩阵运算
R_train <- sparseMatrix(
  i = dt_train_raw$user_idx,
  j = dt_train_raw$item_idx,
  x = rep(1, nrow(dt_train_raw)), 
  dims = c(num_users, num_items)
)

# 1. 计算物品列的 L2 模长
item_norms <- sqrt(colSums(R_train^2))
item_norms[item_norms == 0] <- 1e-9 # 防止除以零

# 2. 列归一化：R_norm = R * D_item^(-1/2)
# 在 sparseMatrix 中，乘以对角矩阵是最高效的列缩放方式
R_norm <- R_train %*% Diagonal(x = 1 / item_norms)

# 3. 计算 Cosine 相似度：Sim = R_norm^T * R_norm
S_cosine <- crossprod(R_norm)

# 4. 移除自相似度 (对角线置 0)
diag(S_cosine) <- 0 
cat("=> Item-CF Cosine 矩阵构建完成.\n")

# 1. 计算所有用户的预测分数矩阵：Pred = R_train * S_cosine
# 结果是一个 [num_users, num_items] 的稀疏矩阵
Pred_matrix <- R_train %*% S_cosine

# 2. 提取 dt_eval 中对应的 (user, item) 预测分
# 转为 dense matrix 直接利用 C++ 级别的 cbind 索引提取，速度极快
pred_scores <- as.matrix(Pred_matrix)[cbind(dt_eval$user_idx, dt_eval$item_idx)]

# 将预测分写回评估集
dt_eval_ibcf <- copy(dt_eval)
dt_eval_ibcf[, pred_score := pred_scores]

# 3. 使用 data.table 原生语法计算 Recall@10
# 按 user_idx 分组，对 pred_score 降序排列并计算排名
dt_eval_ibcf[, rank := frank(-pred_score, ties.method = "random"), by = user_idx]

# 提取正样本 (label == 1) 的排名，统计排名 <= 10 的比例
recall_10_ibcf <- dt_eval_ibcf[label == 1, mean(rank <= 10)]
cat(sprintf("=> Item-CF Recall@10: %.4f\n", recall_10_ibcf))

