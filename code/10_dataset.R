# ==========================================
# 封装 Luz Dataset 与 DataLoader
# ==========================================
ds_matrix_factorization <- dataset(
  name = "MatrixFactorizationDataset",
  initialize = function(dt) {
    self$users <- torch_tensor(dt$user_idx, dtype = torch_long())
    self$items <- torch_tensor(dt$item_idx, dtype = torch_long())
    self$labels <- torch_tensor(dt$label, dtype = torch_float())
  },
  
  # 抛弃 .getitem，改用 .getbatch 进行纯张量切片
  .getbatch = function(indices) {
    # indices 是一个包含 batch_size 个索引的向量
    # 这里直接利用 C++ 底层一次性切出整个 Batch
    list(list(user = self$users[indices], item = self$items[indices]), self$labels[indices])
  },
  
  .length = function() {
    self$labels$size(1)
  }
)

# 使用纯张量切片后，单线程即可喂饱 GPU，关掉 num_workers 避免进程通讯开销
dl_train <- dataloader(
  ds_matrix_factorization(dt_train),
  batch_size = 1024 * 2,
  shuffle = TRUE,
  num_workers = 0
)

dl_eval  <- dataloader(
  ds_matrix_factorization(dt_eval),
  batch_size = 2000,
  # 保持 100 的倍数
  shuffle = FALSE,
  num_workers = 0
)

# ==========================================
# 统一评估框架 (纯 Recall@K)
# ==========================================
evaluate_recall <- function(fitted_model, dl_eval, k = 10) {
  # 1. 获取模型预测结果 (luz 的 predict 默认在无梯度模式下运行)
  preds <- predict(fitted_model, dl_eval)
  
  # 确保转为 torch_tensor
  if (!inherits(preds, "torch_tensor")) {
    preds <- torch_tensor(as.numeric(preds), dtype = torch_float())
  }
  
  # 2. Reshape 为 [N_users, 100] 的矩阵
  # 依赖前置条件: batch_size 是 100 的倍数，且每个用户 1正+99负 连续排列
  num_users_eval <- preds$size(1) / 100L
  preds_matrix <- preds$view(c(num_users_eval, 100L))
  
  # 3. 提取 Top-K 的索引
  topk_indices <- torch_topk(
    preds_matrix,
    k,
    dim = -1L,
    largest = TRUE,
    sorted = TRUE
  )[[2]]
  
  # 4. 计算 Recall
  # 因为正样本永远在每个用户的第一列 (Index = 1L)
  # 直接统计每行 (每个用户) 的 Top-K 索引中是否包含 1L
  hits_tensor <- (topk_indices == 1L)$sum(dim = -1L)$to(dtype = torch_float())
  recall <- hits_tensor$mean()$item()
  
  return(recall)
}
