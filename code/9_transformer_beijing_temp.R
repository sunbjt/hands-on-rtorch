library(torch)
library(luz)
library(tidyverse)
library(lubridate)
library(tidymodels)
library(showtext)
showtext_auto()
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

torch_manual_seed(6)
set.seed(6)

# ==========================================
# 1. 数据准备与特征工程
# ==========================================
# 北京 PM2.5 数据，使用温度预测
# https://github.com/sunbjt/hands-on-rtorch/tree/main/bookdata

url <- "~/data/PRSA_data_2010.1.1-2014.12.31.csv"

# 基础清洗与特征提取
df_clean <- read.csv(url) %>%
  mutate(time = ymd_h(paste(year, month, day, hour))) %>%
  arrange(time) %>%
  mutate(
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  ) %>%
  # 仅保留后续需要用的列（加上 time 备用）
  select(time, TEMP, PRES, DEWP, hour_sin, hour_cos, month_sin, month_cos)

# ==========================================
# 2. 数据集划分与标准化 (Tidymodels 工业级流水线)
# ==========================================
seq_len <- 72

# 专门用于时序数据，严格按时间顺序截断 (前80%训练，后20%验证)
time_split <- initial_time_split(df_clean, prop = 0.8)
train_raw <- training(time_split)
valid_raw <- testing(time_split)

ts_recipe <- recipe(~ ., data = train_raw) %>%
  update_role(time, new_role = "ID") %>% 
  step_normalize(TEMP, PRES, DEWP) 

ts_prep <- prep(ts_recipe, training = train_raw)

train_data_norm <- bake(ts_prep, new_data = NULL, -time) %>% as.matrix()
valid_data_norm <- bake(ts_prep, new_data = valid_raw, -time) %>% as.matrix()

# TIDYMODELS 把计算好的统计量极其规范地存在了 prep 对象里
temp_mean <- ts_prep$steps[[1]]$means["TEMP"]
temp_sd   <- ts_prep$steps[[1]]$sds["TEMP"]

# ==========================================
# 3. 定义 Dataset 和 DataLoader
# ==========================================
time_series_dataset <- dataset(
  name = "ts_dataset",
  initialize = function(data, seq_len) {
    self$data <- torch_tensor(data, dtype = torch_float())
    self$seq_len <- seq_len
  },
  .getitem = function(i) {
    x <- self$data[i:(i + self$seq_len - 1), ]
    y <- self$data[i + self$seq_len, 1]$unsqueeze(1) # 预测目标: 下一个时刻的 TEMP
    list(x = x, y = y)
  },
  .length = function() {
    nrow(self$data) - self$seq_len
  }
)

train_dl <- dataloader(time_series_dataset(train_data_norm, seq_len), batch_size = 64, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(time_series_dataset(valid_data_norm, seq_len), batch_size = 64, shuffle = FALSE, drop_last = TRUE)


# ------------------------------------------
# 版本 A: 使用原生 Transformer Encoder (适用于不提取权重的纯预测)
# ------------------------------------------
tsTransformer_Native <- nn_module(
  "tsTransformer_Native",
  initialize = function(
    input_size, d_model, nhead, seq_len, num_layers = 1, dropout = 0.1) {
    self$proj <- nn_linear(input_size, d_model)
    self$pos_embed <- nn_embedding(
      num_embeddings = seq_len, embedding_dim = d_model)
    
    # 原生 Transformer Layer
    encoder_layer <- nn_transformer_encoder_layer(
      d_model = d_model, 
      nhead = nhead, 
      dropout = dropout, 
      batch_first = TRUE
    )
    self$transformer <- nn_transformer_encoder(
      encoder_layer, num_layers = num_layers)
    self$decoder <- nn_linear(d_model, 1)
  },
  forward = function(x) {
    seq_length <- x$shape[2]
    x_emb <- self$proj(x)
    
    positions <- torch_tensor(1:seq_length, dtype = torch_long(), device = x$device)
    pos_embeddings <- self$pos_embed(positions)
    x_combined <- x_emb + pos_embeddings
    
    # 直接由原生 Encoder 处理
    x_out <- self$transformer(x_combined)
    
    last_step_features <- x_out[, seq_length, ]
    pred <- self$decoder(last_step_features)
    return(pred)
  }
)

fitted_Native <- tsTransformer_Native %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    input_size = ncol(df_clean) - 1,
    d_model = 64,
    nhead = 4,
    seq_len = seq_len
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_dl,
    epochs = 20,
    valid_data = valid_dl,
    callbacks = list( 
      luz_callback_early_stopping( 
        patience = 5, # 容忍多少个 epoch 没有提升
        min_delta = 0.0,
        monitor = "valid_loss"
      ) 
    )
  )

plot(fitted_Native)
preds_tensor <- predict(fitted_Native, valid_dl)
preds_scaled <- as.numeric(preds_tensor)
actuals_scaled <- valid_data_norm[(seq_len + 1):nrow(valid_data_norm), 1]

# 反标准化
preds_real <- preds_scaled * temp_sd + temp_mean
actuals_real <- actuals_scaled * temp_sd + temp_mean

plot_len <- 200
n <- nrow(df_clean)
tail_idx <- (length(preds_real) - plot_len + 1):length(preds_real)

df_res <- data.frame(
  time = df_clean$time[(n - plot_len + 1):n],
  Actual = actuals_real[tail_idx],
  Predicted = preds_real[tail_idx]
) %>% pivot_longer(cols = c("Actual", "Predicted"), names_to = "Type", values_to = "Value")

p1 <- ggplot(df_res, aes(x = time, y = Value, color = Type)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "#333333", "Predicted" = "#E63946")) +
  labs(title = "", y = "温度C", x = "时间步（Hours）")

print(p1)



# ------------------------------------------
# 版本 B: 用于可视化的拆解版-embedding
# ------------------------------------------
TimeSeriesTransformer <- nn_module(
  "TimeSeriesTransformer",
  initialize = function(input_size, d_model, nhead, seq_len, dropout = 0.1) {
    self$proj <- nn_linear(input_size, d_model)
    
    self$pos_embed <- nn_embedding(
      num_embeddings = seq_len, embedding_dim = d_model)
    
    self$self_attn <- nn_multihead_attention(
      embed_dim = d_model, 
      num_heads = nhead, 
      dropout = dropout, 
      batch_first = TRUE
    )
    self$norm1 <- nn_layer_norm(d_model)
    self$norm2 <- nn_layer_norm(d_model)
    self$ffn <- nn_sequential(
      nn_linear(d_model, d_model * 4),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(d_model * 4, d_model)
    )
    self$decoder <- nn_linear(d_model, 1)
    
    self$last_attn_weights <- NULL 
  },
  forward = function(x) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    
    x_emb <- self$proj(x)
    
    positions <- torch_tensor(1:seq_length, dtype = torch_long(), device = x$device)
    pos_embeddings <- self$pos_embed(positions)
    x_combined <- x_emb + pos_embeddings
    
    attn_out <- self$self_attn(
      x_combined, x_combined, x_combined, 
      need_weights = TRUE, 
      avg_weights = FALSE 
    )
    
    self$last_attn_weights <- attn_out[[2]]$detach()$cpu()
    
    x_combined <- self$norm1(x_combined + attn_out[[1]])
    ffn_out <- self$ffn(x_combined)
    x_out <- self$norm2(x_combined + ffn_out)
    
    last_step_features <- x_out[, seq_length, ] 
    pred <- self$decoder(last_step_features)
    
    return(pred)
  }
)

# 模型训练 
fitted1 <- TimeSeriesTransformer %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    input_size = ncol(df_clean) - 1,
    d_model = 32,
    nhead = 4,
    seq_len = seq_len
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_dl,
    epochs = 20,
    valid_data = valid_dl,
    callbacks = list( 
      luz_callback_early_stopping( 
        patience = 3, # 容忍多少个 epoch 没有提升
        min_delta = 0.0,
        monitor = "valid_loss"
        ) 
      )
  )


# ==========================================
# 多头注意力 (Multi-Head) 拆解
# ==========================================
# 1. 抓取数据并触发前向传播
valid_iter <- valid_dl$.iter()
batch <- valid_iter$.next()
x_val <- batch$x

model_device <- fitted1$model$parameters[[1]]$device
x_val <- x_val$to(device = model_device)

fitted1$model$eval()
with_no_grad({
  dummy_out <- fitted1$model(x_val) 
})

# 2. 提取 4 维权重矩阵: [Batch=64, Heads=4, Output_Seq=60, Input_Seq=60]
batch_weights <- as.array(fitted1$model$last_attn_weights$cpu()) 

# 3. 在 Batch 维度 (第1维) 上求平均，保留 Heads 维度
# 结果 mean_weights_per_head 维度变为: [4, 60, 60]
mean_weights_per_head <- apply(batch_weights, MARGIN = c(2, 3, 4), FUN = mean)

# 4. 取出第 seq_len 个 Output_Pos 对所有历史步的视线
# 结果 head_weights_mat 维度为: [4, seq_len]
head_weights_mat <- mean_weights_per_head[, seq_len, ]

# 5. 格式转换，准备绘图
df_heads <- reshape2::melt(head_weights_mat) %>%
  rename(Head = Var1, Input_Pos = Var2, Weight = value) %>%
  mutate(
    Head = paste("Attention Head", Head), 
    Lag = seq_len - Input_Pos # 计算滞后时间
  )

# 6. 使用 facet_wrap 绘制四宫格解构图
p_heads <- ggplot(df_heads, aes(x = Lag, y = Weight, fill = Head)) +
  geom_line(aes(color = Head), linewidth = 1) +
  geom_area(alpha = 0.2) +
  geom_vline(xintercept = c(24, 48, 72), linetype = "dashed", color = "#E63946", linewidth = 0.6) +
  scale_x_reverse(breaks = seq(0, seq_len, 12), expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  facet_wrap(~ Head, ncol = 2) + 
  theme_minimal() +
  labs(
    x = "历史时刻距今的滞后时间 (Lag, Hours ago)",
    y = "平均注意力权重 (Mean Attention Weight)"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11, color = "#333333"),
    panel.grid.minor = element_blank(),
    panel.spacing.x = unit(1, "lines") # 控制左右间距
  )

print(p_heads)


# ------------------------------------------
# 版本 C: 用于可视化的 ALiBi 线性偏差版
# ------------------------------------------

# 辅助函数：使用 R 原生逻辑预计算 ALiBi 距离惩罚矩阵
create_alibi_matrix <- function(nhead, max_len) {
  slopes <- 2^(-8 * (1:nhead) / nhead)
  # 使用  outer 计算绝对距离矩阵: |i - j|
  dist_matrix <- abs(outer(1:max_len, 1:max_len, FUN = "-"))
  alibi_base <- array(0, dim = c(nhead, max_len, max_len))
  #  遍历每个 Head，乘以对应的负斜率
  for (h in 1:nhead) {
    alibi_base[h, , ] <- dist_matrix * (-slopes[h])
  }
  return(alibi_base)
}

TimeSeriesTransformer_ALiBi <- nn_module(
  "TimeSeriesTransformer_ALiBi",
  initialize = function(
    input_size, d_model, nhead, seq_len, dropout = 0.1, max_length = 500) {
    self$proj <- nn_linear(input_size, d_model)
    self$nhead <- nhead
    
    self$self_attn <- nn_multihead_attention(
      embed_dim = d_model, 
      num_heads = nhead, 
      dropout = dropout, 
      batch_first = TRUE
    )
    self$norm1 <- nn_layer_norm(d_model)
    self$norm2 <- nn_layer_norm(d_model)
    self$ffn <- nn_sequential(
      nn_linear(d_model, d_model * 4),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(d_model * 4, d_model)
    )
    self$decoder <- nn_linear(d_model, 1)
    
    self$last_attn_weights <- NULL 
    
    # 调用外部函数获取矩阵，并注册为 GPU 缓存
    alibi_array <- create_alibi_matrix(nhead, max_len = max_length)
    alibi_tensor <- torch_tensor(alibi_array, dtype = torch_float())
    self$register_buffer("alibi_base", alibi_tensor)
  },
  
  forward = function(x) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    
    x_combined <- self$proj(x)
    
    # 获取 mask 矩阵
    sliced_alibi <- self$alibi_base[, 1:seq_length, 1:seq_length]
    final_alibi_mask <- sliced_alibi$
      unsqueeze(1)$
      expand(c(batch_size, self$nhead, seq_length, seq_length))$
      reshape(c(batch_size * self$nhead, seq_length, seq_length))
    
    # 将切片好的矩阵注入注意力机制
    attn_out <- self$self_attn(
      x_combined, x_combined, x_combined, 
      attn_mask = final_alibi_mask, 
      need_weights = TRUE, 
      avg_weights = FALSE 
    )
    
    self$last_attn_weights <- attn_out[[2]]$detach()$cpu()
    
    x_combined <- self$norm1(x_combined + attn_out[[1]])
    ffn_out <- self$ffn(x_combined)
    x_out <- self$norm2(x_combined + ffn_out)
    
    last_step_features <- x_out[, seq_length, ] 
    pred <- self$decoder(last_step_features)
    
    return(pred)
  }
)

fitted2 <- ALiBiTransformer %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    input_size = ncol(df_clean) - 1,
    d_model = 32,
    nhead = 4,
    seq_len = seq_len
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_dl,
    epochs = 20,
    valid_data = valid_dl,
    callbacks = list( 
      luz_callback_early_stopping( 
        patience = 3, # 容忍多少个 epoch 没有提升
        min_delta = 0.0,
        monitor = "valid_loss"
      ) 
    )
  )

# 1. 抓取数据并触发前向传播
valid_iter <- valid_dl$.iter()
batch <- valid_iter$.next()
x_val <- batch$x

model_device <- fitted2$model$parameters[[1]]$device
x_val <- x_val$to(device = model_device)

fitted2$model$eval()
with_no_grad({
  dummy_out <- fitted2$model(x_val) 
})

# 2. 提取 4 维权重矩阵: [Batch=64, Heads=4, Output_Seq=60, Input_Seq=60]
batch_weights <- as.array(fitted2$model$last_attn_weights$cpu()) 

# 3. 在 Batch 维度 (第1维) 上求平均，保留 Heads 维度
# 结果 mean_weights_per_head 维度变为: [4, 60, 60]
mean_weights_per_head <- apply(batch_weights, MARGIN = c(2, 3, 4), FUN = mean)

# 4. 精准提取：取出第 seq_len 个 Output_Pos 对所有历史步的视线
# 结果 head_weights_mat 维度为: [4, seq_len]
head_weights_mat <- mean_weights_per_head[, seq_len, ]

# 5. 格式转换，准备绘图
df_heads <- reshape2::melt(head_weights_mat) %>%
  rename(Head = Var1, Input_Pos = Var2, Weight = value) %>%
  mutate(
    Head = paste("Attention Head", Head), 
    Lag = seq_len - Input_Pos # 计算滞后时间
  )

# 6. 使用 facet_wrap 绘制四宫格解构图
p_heads <- ggplot(df_heads, aes(x = Lag, y = Weight, fill = Head)) +
  geom_line(aes(color = Head), linewidth = 1) +
  geom_area(alpha = 0.2) +
  geom_vline(xintercept = c(24, 48, 72), linetype = "dashed", color = "#E63946", linewidth = 0.6) +
  scale_x_reverse(breaks = seq(0, seq_len, 12), expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  facet_wrap(~ Head, ncol = 2) + 
  theme_minimal() +
  labs(
    x = "历史时刻距今的滞后时间 (Lag, Hours ago)",
    y = "平均注意力权重 (Mean Attention Weight)"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11, color = "#333333"),
    panel.grid.minor = element_blank(),
    panel.spacing.x = unit(1, "lines") # 控制左右间距
  )

print(p_heads)

# ==========================================
# 9. 极限压力测试：ALiBi 的零次学习外推 (Zero-shot Extrapolation)
# ==========================================

# 1. 强行将序列长度翻倍！(72 -> 144)
seq_len_ext <- 144

# 2. 用新长度重新构建验证集的 DataLoader
valid_dl_ext <- dataloader(
  time_series_dataset(valid_data_norm, seq_len_ext), 
  batch_size = 64, 
  shuffle = FALSE, 
  drop_last = TRUE
)

# 3. 抓取一个 144 长度的 Batch
valid_iter_ext <- valid_dl_ext$.iter()
batch_ext <- valid_iter_ext$.next()
x_val_ext <- batch_ext$x$to(device = model_device)

# 4. ALiBi 模型前向传播 (毫无压力通过)
fitted2$model$eval()
with_no_grad({
  # 见证奇迹的时刻：不需要任何修改，直接喂给训练时只见过 72 长度的模型
  pred_ext <- fitted2$model(x_val_ext) 
})

cat("ALiBi 模型完美存活！成功输出预测结果。\n")

# ==========================================
# 10. 绘制 144 小时长序列下的 ALiBi 注意力衰减图
# ==========================================
# 提取权重
batch_weights_ext <- as.array(fitted2$model$last_attn_weights$cpu()) 
mean_weights_ext <- apply(batch_weights_ext, MARGIN = c(2, 3, 4), FUN = mean)

# 取出预测时刻 (第 144 步) 往前看 144 步的视线
head_weights_mat_ext <- mean_weights_ext[, seq_len_ext, ]

# 数据整理
df_heads_ext <- reshape2::melt(head_weights_mat_ext) %>%
  rename(Head = Var1, Input_Pos = Var2, Weight = value) %>%
  mutate(
    Head = paste("Attention Head", Head), 
    Lag = seq_len_ext - Input_Pos # 这里的 Lag 最大会到 144
  )

# 绘图
p_ext <- ggplot(df_heads_ext, aes(x = Lag, y = Weight, fill = Head)) +
  geom_line(aes(color = Head), linewidth = 1) +
  geom_area(alpha = 0.2) +
  # 画一条 72 小时的分界线
  geom_vline(xintercept = 72, linetype = "dashed", color = "#E63946", linewidth = 0.8) +
  annotate("text", x = 108, y = max(df_heads_ext$Weight)*0.8, 
           label = "从未见过的领域\n(>72h)", color = "#E63946", fontface = "bold") +
  scale_x_reverse(breaks = seq(0, seq_len_ext, 24), expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  facet_wrap(~ Head, ncol = 2) + 
  theme_minimal() +
  labs(
   x = "历史时刻距今的滞后时间 (Lag, Hours ago)",
    y = "平均注意力权重 (Mean Attention Weight)"
  ) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey30", size = 10),
    strip.text = element_text(face = "bold", size = 11, color = "#333333"),
    panel.grid.minor = element_blank()
  )

print(p_ext)

