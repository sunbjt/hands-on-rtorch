# ==========================================
# 0. 环境准备与包加载
# ==========================================
library(vroom)
library(data.table)
library(torch)

# 使用 CPU (如果后续想换 MPS/CUDA 直接改这里即可)
device <- torch_device("cpu")
cat("当前使用设备:", as.character(device), "\n")

# ==========================================
# 1. 数据读取与基础预处理
# ==========================================
DATA_DIR <- "~/data/ml-1m"
FILE_PATH <- file.path(DATA_DIR, "ratings.dat")

cat("\n1. 正在读取数据...\n")
dt_ratings <- vroom(
  FILE_PATH,
  delim = "::",
  col_names = c("user_id", "movie_id", "rating", "timestamp"),
  col_select = c(user_id, movie_id, rating, timestamp),
  show_col_types = FALSE
)
setDT(dt_ratings)

# 提取唯一 ID 并计算维度
unique_users <- unique(dt_ratings$user_id)
unique_items <- unique(dt_ratings$movie_id)
num_users <- length(unique_users)
num_items <- length(unique_items)
all_items_vec <- 1:num_items

# 映射为从 1 开始的连续整数索引
dt_ratings[, user_idx := match(user_id, unique_users)]
dt_ratings[, item_idx := match(movie_id, unique_items)]

# 严谨的时序划分 (80% Train, 20% Test)
cat("2. 正在进行时序划分...\n")
setorder(dt_ratings, user_idx, timestamp)
dt_ratings[, `:=`(seq_num = 1:.N, total_num = .N), by = user_idx]

dt_train_raw <- dt_ratings[seq_num <= floor(0.8 * total_num)]
dt_test_raw  <- dt_ratings[seq_num > floor(0.8 * total_num)]

# 过滤测试集中的冷启动物品
valid_train_items <- unique(dt_train_raw$item_idx)
dt_test_raw <- dt_test_raw[item_idx %in% valid_train_items]

# 全局历史记录 (用于后续负采样避开正样本)
dt_user_history <- dt_ratings[, .(interacted = list(item_idx)), by = user_idx]

# ==========================================
# 2. 构建训练集 (Tensor 预处理极致优化版)
# ==========================================
cat("3. 正在构建训练序列 (Tensor 预构建)...\n")
setorder(dt_train_raw, user_idx, timestamp)
train_seqs_dt <- dt_train_raw[, .(item_seq = list(item_idx)), by = user_idx]
train_seqs_dt[, seq_len := sapply(item_seq, length)]
train_seqs_dt <- train_seqs_dt[seq_len >= 3] # 过滤过短序列
maxlen <- 50 # 最大序列长度

# 一次性把 List 裁剪/填充好
user_seqs_list <- lapply(train_seqs_dt$item_seq, function(seq) {
  seq <- seq + 1 # 真实物品 ID + 1，为 Padding 留出 1
  n <- length(seq)
  tokens <- seq[-n]
  labels <- seq[-1]
  seq_len <- length(tokens)
  
  if (seq_len > maxlen) {
    tokens <- tokens[(seq_len - maxlen + 1):seq_len]
    labels <- labels[(seq_len - maxlen + 1):seq_len]
  } else if (seq_len < maxlen) {
    pad_len <- maxlen - seq_len
    tokens <- c(tokens, rep(1, pad_len))
    labels <- c(labels, rep(1, pad_len))
  }
  list(tokens = tokens, labels = labels)
})

# 优化版 Dataset：丢弃 lapply 内部生成，直接构建大 Tensor
sasrec_dataset_optimized <- dataset(
  name = "sasrec_dataset",
  initialize = function(user_seqs, num_items, maxlen) {
    self$num_items <- num_items
    self$maxlen <- maxlen
    
    # 将 list 提前拼装并转换为整块 Tensor，避免 getitem 时的 R 语言开销
    tokens_mat <- do.call(rbind, lapply(user_seqs, `[[`, "tokens"))
    labels_mat <- do.call(rbind, lapply(user_seqs, `[[`, "labels"))
    
    self$data_tokens <- torch_tensor(tokens_mat, dtype = torch_long())
    self$data_labels <- torch_tensor(labels_mat, dtype = torch_long())
    cat("   训练集 Tensor 构建完成。样本数:", self$data_tokens$size(1), "\n")
  },
  .getitem = function(i) {
    # 只做切片，负采样放在 batch 级别做
    list(
      log_seqs = self$data_tokens[i, ],
      pos_labels = self$data_labels[i, ]
    )
  },
  .length = function() {
    self$data_tokens$size(1)
  }
)

dl_train <- dataloader(
  sasrec_dataset_optimized(user_seqs_list, num_items = num_items, maxlen = maxlen),
  batch_size = 256,
  shuffle = TRUE,
  num_workers = 0 
)

# ==========================================
# 3. 构造测试集 (1正99负 严格对齐评估)
# ==========================================
cat("4. 正在构建评估集 (1正 + 99负)...\n")
set.seed(42)

dt_eval_pos <- dt_test_raw[, .SD[1], by = user_idx][, .(user_idx, item_idx, label = 1)]
valid_eval_users <- dt_eval_pos$user_idx

dt_eval_neg <- 
  dt_user_history[user_idx %in% valid_eval_users,
                  .(item_idx = sample(
                    setdiff(all_items_vec, unlist(interacted)), 99, replace = FALSE),
                    label = 0), by = user_idx]

dt_eval <- rbindlist(list(dt_eval_pos, dt_eval_neg), use.names = TRUE)
setorder(dt_eval, user_idx, -label) 

dt_eval_candidates <- dt_eval[, .(candidates = list(item_idx)), by = user_idx]
setkey(train_seqs_dt, user_idx)
dt_eval_merged <- merge(train_seqs_dt[user_idx %in% valid_eval_users],
                        dt_eval_candidates, by = "user_idx")

eval_list <- lapply(1:nrow(dt_eval_merged), function(i) {
  seq <- dt_eval_merged$item_seq[[i]] + 1
  candidates <- dt_eval_merged$candidates[[i]] + 1
  seq_len <- length(seq)
  
  if (seq_len > maxlen) {
    seq <- seq[(seq_len - maxlen + 1):seq_len]
    seq_len <- maxlen
  }
  pad_len <- maxlen - seq_len
  if (pad_len > 0) seq <- c(seq, rep(1, pad_len))
  
  list(seq = seq, candidates = candidates, valid_len = seq_len)
})

sasrec_eval_dataset <- dataset(
  name = "sasrec_eval_dataset",
  initialize = function(eval_list) {
    self$eval_list <- eval_list
  },
  .getitem = function(i) {
    item <- self$eval_list[[i]]
    list(
      log_seqs = torch_tensor(item$seq, dtype = torch_long()),
      candidates = torch_tensor(item$candidates, dtype = torch_long()),
      valid_len = torch_tensor(item$valid_len, dtype = torch_long())
    )
  },
  .length = function() {
    length(self$eval_list)
  }
)

dl_eval <- dataloader(
  sasrec_eval_dataset(eval_list),
  batch_size = 256,
  shuffle = FALSE,
  num_workers = 0
)

# ==========================================
# 4. SASRec 模型架构
# ==========================================
sasrec_optimized <- nn_module(
  "SASRec",
  initialize = function(item_num,
                        maxlen,
                        hidden_units,
                        num_blocks = 2,
                        num_heads = 1,
                        dropout_rate = 0.2) {
    self$item_emb <- nn_embedding(item_num + 2, hidden_units, padding_idx = 1)
    self$pos_emb <- nn_embedding(maxlen, hidden_units)
    self$emb_dropout <- nn_dropout(dropout_rate)
    
    nn_init_normal_(self$item_emb$weight, mean = 0, std = 0.02)
    nn_init_normal_(self$pos_emb$weight, mean = 0, std = 0.02)
    
    self$register_buffer("pos_idx", torch_arange(1, maxlen, dtype = torch_long()))
    mask <- torch_full(c(maxlen, maxlen), -Inf)
    mask <- torch_triu(mask, diagonal = 1)
    self$register_buffer("causal_mask", mask)
    
    encoder_layer <- nn_transformer_encoder_layer(
      d_model = hidden_units,
      nhead = num_heads,
      dim_feedforward = hidden_units * 4, 
      dropout = dropout_rate,
      batch_first = TRUE,
      norm_first = TRUE # 开启 Pre-LN，这是不加 Warmup 也能收敛的关键！
    )
    self$transformer <- nn_transformer_encoder(encoder_layer, num_layers = num_blocks)
  },
  forward = function(log_seqs, pos_labels = NULL, neg_labels = NULL) {
    seq_len <- log_seqs$size(2)
    
    pos <- self$pos_idx[1:seq_len] 
    mask <- self$causal_mask[1:seq_len, 1:seq_len]
    
    seqs <- self$item_emb(log_seqs) + self$pos_emb(pos)
    seqs <- self$emb_dropout(seqs)
    
    key_padding_mask <- (log_seqs == 1)
    seqs <- self$transformer(seqs, mask = mask, src_key_padding_mask = key_padding_mask)
    
    if (!is.null(pos_labels) && !is.null(neg_labels)) {
      pos_embs <- self$item_emb(pos_labels)
      neg_embs <- self$item_emb(neg_labels)
      pos_logits <- torch_sum(seqs * pos_embs, dim = 3)
      neg_logits <- torch_sum(seqs * neg_embs, dim = 3)
      return(list(pos_logits = pos_logits, neg_logits = neg_logits))
    }
    
    return(seqs)
  }
)

# ==========================================
# 5. 评估函数定义 (保持原逻辑)
# ==========================================
evaluate_sasrec_recall <- function(model, dl_eval, k = 10, device) {
  model$eval()
  total_hits <- 0
  total_users <- 0
  
  with_no_grad({
    coro::loop(for (batch in dl_eval) {
      log_seqs <- batch$log_seqs$to(device = device)
      candidates <- batch$candidates$to(device = device) # [B, 100]
      valid_lens <- batch$valid_len$to(device = device)  # [B]
      
      seqs <- model(log_seqs) # [B, Maxlen, Hidden]
      B <- seqs$size(1)
      hidden_dim <- seqs$size(3)
      
      index <- valid_lens$view(c(B, 1, 1))$expand(c(B, 1, hidden_dim))
      last_hidden <- seqs$gather(dim = 2, index = index)$squeeze(2) # [B, Hidden]
      
      cand_embs <- model$item_emb(candidates) # [B, 100, Hidden]
      scores <- torch_sum(last_hidden$unsqueeze(2) * cand_embs, dim = 3) # [B, 100]
      
      topk_indices <- torch_topk(scores, k, dim = -1L, largest = TRUE, sorted = TRUE)[[2]]
      hits <- (topk_indices == 1L)$sum()$item() 
      
      total_hits <- total_hits + hits
      total_users <- total_users + B
    })
  })
  return(total_hits / total_users)
}

# ==========================================
# 6. 开始训练 (优化后的批量负采样)
# ==========================================
hidden_units <- 64
num_blocks <- 2
num_heads <- 1
dropout_rate <- 0.2
epochs <- 40        
eval_interval <- 10

# 实例化模型
model <- sasrec_optimized(num_items, maxlen, hidden_units, num_blocks, num_heads, dropout_rate)
model <- model$to(device = device)
optimizer <- optim_adamw(model$parameters, lr = 0.001, weight_decay = 1e-5)

# 开始训练模型
for (epoch in 1:epochs) {
  model$train()
  total_loss <- 0
  
  coro::loop(for (batch in dl_train) {
    log_seqs <- batch$log_seqs$to(dtype = torch_long(), device = device)
    pos_labels <- batch$pos_labels$to(dtype = torch_long(), device = device)
    
    neg_labels <- torch_randint(
      low = 2, 
      high = num_items + 2, 
      size = pos_labels$shape, 
      dtype = torch_long(),
      device = device
    )
    
    optimizer$zero_grad()
    logits <- model(log_seqs, pos_labels, neg_labels)
    
    pad_mask <- (pos_labels != 1)
    valid_pos_logits <- logits$pos_logits[pad_mask]
    valid_neg_logits <- logits$neg_logits[pad_mask]
    
    loss_pos <- nnf_binary_cross_entropy_with_logits(valid_pos_logits,
                                                     torch_ones_like(valid_pos_logits))
    loss_neg <- nnf_binary_cross_entropy_with_logits(valid_neg_logits,
                                                     torch_zeros_like(valid_neg_logits))
    
    loss <- loss_pos + loss_neg
    loss$backward()
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
  })
  
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %03d | Training Loss: %.4f\n", epoch, total_loss))
  }
  
  if (epoch %% eval_interval == 0) {
    recall_score <- evaluate_sasrec_recall(model, dl_eval, k = 10, device = device)
    cat(sprintf(
      "[Eval] Epoch %03d | SASRec Recall@10: %.4f \n",
      epoch,
      recall_score
    ))
  }
}

# ==========================================
# 7. 线上实时预测模拟
# ==========================================
cat("\n--- 模拟线上实时推荐开始 ---\n")
recent_clicks <- c(10, 42, 99, 205, 310)
seq_len <- length(recent_clicks)
cat(sprintf("用户近期点击了 %d 部电影...\n", seq_len))

tokens <- recent_clicks + 1
pad_len <- maxlen - seq_len
if (pad_len > 0) {
  tokens_padded <- c(tokens, rep(1, pad_len))
} else {
  tokens_padded <- tokens[(seq_len - maxlen + 1):seq_len]
}

log_seqs_tensor <- torch_tensor(tokens_padded,
                                dtype = torch_long())$unsqueeze(1)$to(device = device)

model$eval()
with_no_grad({
  seq_output <- model(log_seqs_tensor)
  intent_vector <- seq_output[1, seq_len, ] 
  cat("实时生成的意图向量 (前5维):", as.numeric(intent_vector[1:5]), "...\n")
  
  all_item_embs <- model$item_emb$weight[2:(num_items + 1), ]
  scores <- torch_matmul(all_item_embs, intent_vector) 
  scores[recent_clicks] <- -1e9
  
  topk_res <- torch_topk(scores, k = 5, largest = TRUE, sorted = TRUE)
  topk_item_ids <- topk_res[[2]]$cpu() %>% as.integer()
  topk_scores   <- topk_res[[1]]$cpu() %>% as.numeric()
})

cat("\n 最终推荐给该用户的 Top-5 电影 ID:\n")
for (i in 1:5) {
  cat(sprintf("Top %d: 电影ID %4d (匹配得分: %.4f)\n", i, topk_item_ids[i], topk_scores[i]))
}
