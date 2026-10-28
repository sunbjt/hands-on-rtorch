# ===================================================================
# 推荐系统对比基准: Item-CF vs GNN-Enhanced CF vs LightGCN
# ===================================================================
library(vroom)
library(data.table)
library(torch)
library(Matrix)

# --- 0. 随机种子与配置 ---
set.seed(42)
torch_manual_seed(42)

DATA_DIR <- "~/data/ml-1m" # 请根据实际情况修改
FILE_PATH <- file.path(DATA_DIR, "ratings.dat")

# LightGCN 超参数
TRAIN_BATCH_SIZE <- 16384
EMB_DIM    <- 128
LR         <- 0.002
EPOCHS     <- 50   
LAYERS     <- 2
L2_DECAY   <- 1e-4

# mps 对系数矩阵支持不好，所以只能用 CPU
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

cat(">>> 使用计算设备:", device$type, "\n")

# ===================================================================
# 1. 统一数据管道 (读取、时序划分、冷启动过滤)
# ===================================================================
cat("\n>>> [1/6] 读取数据与严谨的时序划分...\n")

dt_ratings <- vroom(
  FILE_PATH, delim = "::",
  col_names = c("user_id", "movie_id", "rating", "timestamp"),
  col_select = c(user_id, movie_id, rating, timestamp),
  show_col_types = FALSE
)
setDT(dt_ratings)

unique_users <- unique(dt_ratings$user_id)
unique_items <- unique(dt_ratings$movie_id)
num_users <- length(unique_users)
num_items <- length(unique_items)
num_nodes <- num_users + num_items
all_items_vec <- 1:num_items

dt_ratings[, user_idx := match(user_id, unique_users)]
dt_ratings[, item_idx := match(movie_id, unique_items)]

# 严谨的时序划分 (80% Train, 20% Test)
setorder(dt_ratings, user_idx, timestamp)
dt_ratings[, `:=`(seq_num = 1:.N, total_num = .N), by = user_idx]

dt_train_raw <- dt_ratings[seq_num <= floor(0.8 * total_num)]
dt_test_raw  <- dt_ratings[seq_num > floor(0.8 * total_num)]

# 过滤测试集中的冷启动物品
valid_train_items <- unique(dt_train_raw$item_idx)
dt_test_raw <- dt_test_raw[item_idx %in% valid_train_items]

# ===================================================================
# 2. 统一构造评估集 (Leave-One-Out: 1 正 + 99 负)
# ===================================================================
cat(">>> [2/6] 构造统一的 Leave-One-Out 评估集...\n")

dt_user_history <- dt_ratings[, .(interacted = list(item_idx)), by = user_idx]

# 取测试集第一条作为目标正样本
dt_eval_pos <- dt_test_raw[, .SD[1], by = user_idx][, .(user_idx, item_idx, label = 1)]

# 仅为有效的评估用户生成 99 个负样本
valid_eval_users <- dt_eval_pos$user_idx
dt_eval_neg <- dt_user_history[user_idx %in% valid_eval_users,
                               .(item_idx = sample(
                                 setdiff(all_items_vec, unlist(interacted)),
                                 99, replace = FALSE),
                                 label = 0), by = user_idx]

dt_eval <- rbindlist(list(dt_eval_pos, dt_eval_neg), use.names = TRUE)
setorder(dt_eval, user_idx, -label) # 每个用户 100 行连续排列

cat(sprintf("    训练集交互: %d | 评估集样本数: %d\n", nrow(dt_train_raw), nrow(dt_eval)))

# ===================================================================
# 3. Baseline 1: Standard Item-CF
# ===================================================================
cat("\n>>> [3/6] 运行 Standard Item-CF...\n")

R_train <- sparseMatrix(
  i = dt_train_raw$user_idx, j = dt_train_raw$item_idx,
  x = rep(1, nrow(dt_train_raw)), dims = c(num_users, num_items)
)

item_norms <- sqrt(colSums(R_train^2))
item_norms[item_norms == 0] <- 1e-9
R_norm <- R_train %*% Diagonal(x = 1 / item_norms)
S_cosine <- crossprod(R_norm)
diag(S_cosine) <- 0 

# KNN 截断 (此处为了兼容，保留全部非零，若数据集大可调小 keep_top_k)
# S_cosine <- as(S_cosine, "dgCMatrix")
S_cosine <- as(S_cosine, "generalMatrix")
Pred_matrix <- R_train %*% S_cosine

# 提取预测分并评估
dt_eval_ibcf <- copy(dt_eval)
dt_eval_ibcf[, pred_score := as.matrix(Pred_matrix)[cbind(user_idx, item_idx)]]
dt_eval_ibcf[, rank := frank(-pred_score, ties.method = "random"), by = user_idx]
hr10_ibcf <- dt_eval_ibcf[label == 1, mean(rank <= 10)]

cat(sprintf("    Standard Item-CF HR@10: %.4f\n", hr10_ibcf))

# ===================================================================
# 4. Baseline 2: GNN-Enhanced Item-CF
# ===================================================================
cat("\n>>> [4/6] 运行 GNN-Enhanced Item-CF...\n")

A <- S_cosine 
diag(A) <- 1 # 添加自环
d_vals <- rowSums(A)
d_inv_sqrt <- 1 / sqrt(d_vals)
d_inv_sqrt[is.infinite(d_inv_sqrt)] <- 0
D_mat <- Diagonal(x = d_inv_sqrt)

A_hat <- D_mat %*% A %*% D_mat # 对称归一化

K <- 2 # 传播层数
E_layer <- R_train
Pred_matrix_gnn <- E_layer

for (k in 1:K) {
  E_layer <- E_layer %*% A_hat 
  Pred_matrix_gnn <- Pred_matrix_gnn + E_layer
}
Pred_matrix_gnn <- Pred_matrix_gnn / (K + 1)

# 提取预测分并评估
dt_eval_gnn <- copy(dt_eval)
dt_eval_gnn[, pred_score := as.matrix(Pred_matrix_gnn)[cbind(user_idx, item_idx)]]
dt_eval_gnn[, rank := frank(-pred_score, ties.method = "random"), by = user_idx]
hr10_gnn <- dt_eval_gnn[label == 1, mean(rank <= 10)]

cat(sprintf("    GNN-Enhanced CF HR@10: %.4f\n", hr10_gnn))

# ===================================================================
# 5. Model 3: LightGCN 模型与训练
# ===================================================================
cat("\n>>> [5/6] 构建与训练 LightGCN...\n")

dt_train_pos <- dt_train_raw[, .(user_node_id = user_idx, item_node_id = item_idx + num_users)]

build_sparse_adjacency <- function(dt, n_nodes) {
  u_deg <- dt[, .N, by = user_node_id]
  i_deg <- dt[, .N, by = item_node_id]
  
  deg_vec <- rep(0, n_nodes)
  deg_vec[u_deg$user_node_id] <- u_deg$N
  deg_vec[i_deg$item_node_id] <- i_deg$N
  
  deg_inv_sqrt <- 1 / sqrt(deg_vec)
  deg_inv_sqrt[is.infinite(deg_inv_sqrt)] <- 0
  
  edges_fwd <- dt[, .(src = user_node_id, dst = item_node_id)]
  edges_bwd <- dt[, .(src = item_node_id, dst = user_node_id)]
  self_loop <- data.table(src = 1:n_nodes, dst = 1:n_nodes)
  
  all_edges <- rbind(edges_fwd, edges_bwd, self_loop)
  values <- deg_inv_sqrt[all_edges$src] * deg_inv_sqrt[all_edges$dst]
  
  indices <- torch_tensor(matrix(c(all_edges$src, all_edges$dst), nrow = 2, byrow = TRUE), dtype = torch_long())
  values_t <- torch_tensor(values, dtype = torch_float())
  
  return(torch_sparse_coo_tensor(indices, values_t, size = c(n_nodes, n_nodes)))
}

adj_sparse <- build_sparse_adjacency(dt_train_pos, num_nodes)$to(device = device)
gc()

LightGCN <- nn_module(
  "LightGCN",
  initialize = function(num_nodes, emb_dim, n_layers) {
    self$n_layers <- n_layers
    self$embedding <- nn_embedding(num_nodes, emb_dim)
    nn_init_normal_(self$embedding$weight, std = 0.1)
  },
  forward = function(adj) {
    E_current <- self$embedding$weight
    E_final <- E_current
    for (i in 1:self$n_layers) {
      E_current <- torch_mm(adj, E_current)
      E_final <- E_final + E_current
    }
    return(E_final / (self$n_layers + 1))
  },
  score_pairs = function(all_embs, u_indices, i_indices) {
    torch_sum(all_embs[u_indices, ] * all_embs[i_indices, ], dim = 2)
  }
)

model <- LightGCN(num_nodes, EMB_DIM, n_layers = LAYERS)$to(device = device)
optimizer <- optim_adam(model$parameters, lr = LR)

train_u_all <- torch_tensor(dt_train_pos$user_node_id, dtype = torch_long())
train_i_all <- torch_tensor(dt_train_pos$item_node_id, dtype = torch_long())
num_edges <- nrow(dt_train_pos)
num_batches <- ceiling(num_edges / TRAIN_BATCH_SIZE)

for (epoch in 1:EPOCHS) {
  model$train() 
  total_bpr_loss <- 0
  perm_indices <- sample.int(num_edges)
  
  for (i in 1:num_batches) {
    start_idx <- (i - 1) * TRAIN_BATCH_SIZE + 1
    end_idx <- min(i * TRAIN_BATCH_SIZE, num_edges)
    batch_idx <- torch_tensor(perm_indices[start_idx:end_idx], dtype = torch_long())
    
    pos_u <- train_u_all[batch_idx]$to(device = device)
    pos_i <- train_i_all[batch_idx]$to(device = device)
    neg_i <- torch_randint(low = num_users + 1, high = num_nodes + 1, 
                           size = c(length(batch_idx)), 
                           device = device, dtype = torch_long())
    
    optimizer$zero_grad()
    all_embs <- model(adj_sparse)
    
    pos_scores <- model$score_pairs(all_embs, pos_u, pos_i)
    neg_scores <- model$score_pairs(all_embs, pos_u, neg_i)
    bpr_loss <- -torch_mean(nnf_logsigmoid(pos_scores - neg_scores))
    
    E_0 <- model$embedding$weight
    reg_loss <- (E_0[pos_u, ]$norm(p = 2)^2 +
                   E_0[pos_i, ]$norm(p = 2)^2 +
                   E_0[neg_i, ]$norm(p = 2)^2) / 2
    loss <- bpr_loss + L2_DECAY * (reg_loss / length(batch_idx))
    
    loss$backward()
    optimizer$step()
    total_bpr_loss <- total_bpr_loss + bpr_loss$item()
  }
  
  if (epoch %% 10 == 0 || epoch == 1) {
    cat(sprintf("Epoch %02d/%d | Avg BPR Loss: %.4f\n", epoch, EPOCHS, total_bpr_loss / num_batches))
  }
}

# ===================================================================
# 6. LightGCN 评估与最终对比汇总
# ===================================================================
cat("\n>>> [6/6] 评估 LightGCN 并汇总结果...\n")

model$eval()
with_no_grad({
  all_embs <- model(adj_sparse)
  u_idx <- torch_tensor(dt_eval$user_idx, dtype = torch_long(), device = device)
  i_idx <- torch_tensor(dt_eval$item_idx + num_users, dtype = torch_long(), device = device)
  
  u_emb <- all_embs[u_idx, ]
  i_emb <- all_embs[i_idx, ]
  
  scores <- torch_sum(u_emb * i_emb, dim = 2)
  
  dt_eval_lightgcn <- copy(dt_eval)
  dt_eval_lightgcn[, pred_score := as_array(scores$to(device = "cpu"))]
  dt_eval_lightgcn[, rank := frank(-pred_score, ties.method = "random"), by = user_idx]
  hr10_lightgcn <- dt_eval_lightgcn[label == 1, mean(rank <= 10)]
})

# 对比性能
cat(sprintf(" 1. Standard Item-CF   : %.4f\n", hr10_ibcf))
cat(sprintf(" 2. GNN-Enhanced CF    : %.4f\n", hr10_gnn))
cat(sprintf(" 3. LightGCN           : %.4f\n", hr10_lightgcn))

