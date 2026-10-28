# 数据地址：
# https://github.com/sunbjt/hands-on-rtorch/tree/main/bookdata

library(readr)
library(tidyverse)
library(torch)
library(luz)
library(showtext)
showtext_auto()
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================

# --- 路径设置 (请修改为你本地的路径) ---
path_train <- "~/data/ECG5000/ECG5000_TRAIN.txt"
path_test  <- "~/data/ECG5000/ECG5000_TEST.txt"

# --- 读取与合并 ---
# col_names = FALSE 表示没有标题行，readr 会自动命名为 X1, X2...
df_train <- read_table(path_train, col_names = FALSE, show_col_types = FALSE)
df_test  <- read_table(path_test, col_names = FALSE, show_col_types = FALSE)
raw_data <- bind_rows(df_train, df_test)

# ==========================================
# 0. 原始数据 EDA 可视化
# ==========================================
eda_data <- raw_data %>%
  mutate(
    ID = row_number(), # 为每条序列生成唯一 ID
    # 将 1 映射为 Normal，其他映射为 Anomaly
    Label = factor(ifelse(X1 == 1, "Normal", "Anomaly"), levels = c("Normal", "Anomaly")) 
  ) %>%
  # 剔除原始标签列，开始长宽转换
  select(-X1) %>%
  pivot_longer(
    cols = -c(ID, Label), 
    names_to = "TimeStep", 
    values_to = "Value"
  ) %>%
  # 将列名 (如 X2, X3) 转换为纯数字时间步 (1, 2...)
  mutate(TimeStep = as.numeric(str_remove(TimeStep, "X")) - 1)

trend_data <- eda_data %>%
  group_by(Label, TimeStep) %>%
  summarise(
    Mean_Value = mean(Value),
    SD_Value = sd(Value),
    .groups = "drop"
  )

ggplot(trend_data, aes(x = TimeStep, y = Mean_Value, color = Label, fill = Label)) +
  # 绘制均值加减一个标准差的置信带
  geom_ribbon(aes(ymin = Mean_Value - SD_Value, ymax = Mean_Value + SD_Value), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1) +
  labs(
    x = "时间步",
    y = "平均信号值"
  ) +
  theme_minimal() +
  theme(legend.position = "right") +
  # 直接使用 Nature 杂志的配色
  scale_color_npg() + 
  # 或者使用新英格兰医学杂志配色: 
  # scale_color_nejm() +
  # 或者使用柳叶刀配色: 
  # scale_color_lancet() + 
  theme(
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22, face = "bold"),
    strip.text = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    legend.position = "right"
  )

# ==========================================
# 1. 数据和特征处理
# ==========================================
# --- 提取特征与标签 ---
# 第一列是 Label，后面 140 列是时间序列特征
# 原始标签定义：1 = 正常 (Normal), 2/3/4/5 = 异常 (Anomaly)
all_labels   <- raw_data$X1
all_features <- raw_data %>% select(-X1) %>% as.matrix()

# --- 核心划分逻辑 ---
# 目标：训练集只包含"正常"样本（让模型学会什么是"正常"）
# 验证集包含"正常"和"异常"（用来测试模型能不能分清）

idx_normal   <- which(all_labels == 1)
idx_abnormal <- which(all_labels != 1)

# 打乱正常样本的顺序
set.seed(123)
idx_normal_shuffled <- sample(idx_normal)

# 80% 正常样本 -> 训练集
n_normal <- length(idx_normal_shuffled)
n_train  <- round(n_normal * 0.8)

train_indices <- idx_normal_shuffled[1:n_train]
valid_indices_normal <- idx_normal_shuffled[(n_train + 1):n_normal]

# 构建矩阵
x_train <- all_features[train_indices, ]

# 验证集 = 剩余的正常样本 + 所有的异常样本
x_valid_normal   <- all_features[valid_indices_normal, ]
x_valid_abnormal <- all_features[idx_abnormal, ]
x_valid <- rbind(x_valid_normal, x_valid_abnormal)

# 构建验证集标签 (0=正常, 1=异常) 用于评估
y_valid_labels <- c(
  rep(0, nrow(x_valid_normal)),
  rep(1, nrow(x_valid_abnormal))
)

# --- 数据标准化 (Z-Score) ---
# 关键点：只能使用训练集的统计量来标准化验证集 (防止数据泄露)
train_mean <- apply(x_train, 2, mean)
train_sd   <- apply(x_train, 2, sd)
train_sd[train_sd == 0] <- 1 # 避免除以 0

scale_func <- function(x, m, s) {
  # 利用 R 的广播机制：(x - m) / s
  t((t(x) - m) / s)
}

x_train_scaled <- scale_func(x_train, train_mean, train_sd)
x_valid_scaled <- scale_func(x_valid, train_mean, train_sd)

cat(sprintf("训练集 (纯正常): %d 样本 x %d 特征\n", nrow(x_train_scaled), ncol(x_train_scaled)))
cat(sprintf("验证集 (混合):   %d 样本 (正常:%d, 异常:%d)\n", 
            nrow(x_valid_scaled), sum(y_valid_labels==0), sum(y_valid_labels==1)))

# ==========================================
# 2. Dataset 与 DataLoader
# ==========================================

ecg_dataset <- dataset(
  name = "ECG_DS",
  initialize = function(x) {
    self$x <- torch_tensor(x, dtype = torch_float())
  },
  .getitem = function(i) {
    # 对于自编码器，输入是 x，目标也是 x
    list(x = self$x[i, ], y = self$x[i, ])
  },
  .length = function() {
    self$x$size(1)
  }
)

train_ds <- ecg_dataset(x_train_scaled)
valid_ds <- ecg_dataset(x_valid_scaled)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 64, shuffle = FALSE) # 验证集不打乱，方便对应标签

# ==========================================
# 3. VAE 模型定义
# ==========================================

VAE <- nn_module(
  "VAE",
  initialize = function(input_dim = 140, latent_dim = 8) {
    # Encoder: 压缩
    self$enc1    <- nn_linear(input_dim, 32)
    self$enc2    <- nn_linear(32, 16)
    self$fc_mu   <- nn_linear(16, latent_dim)
    self$fc_logvar <- nn_linear(16, latent_dim)
    
    # Decoder: 还原
    self$dec1    <- nn_linear(latent_dim, 16)
    self$dec2    <- nn_linear(16, 32)
    self$fc_out  <- nn_linear(32, input_dim)
    
    self$relu <- nn_relu()
  },
  
  reparameterize = function(mu, log_var) {
    # 重参数化技巧: z = mu + sigma * epsilon
    std <- torch_exp(0.5 * log_var)
    eps <- torch_randn_like(std)
    mu + eps * std
  },
  
  forward = function(x) {
    # 编码
    h <- self$relu(self$enc1(x))
    h <- self$relu(self$enc2(h))
    
    mu <- self$fc_mu(h)
    log_var <- self$fc_logvar(h)
    
    # 采样
    z <- self$reparameterize(mu, log_var)
    
    # 解码
    h_d <- self$relu(self$dec1(z))
    h_d <- self$relu(self$dec2(h_d))
    recon <- self$fc_out(h_d) # 输出层通常不加激活函数，或者根据数据范围用 Sigmoid/Tanh
    
    # 返回列表，包含重构值和分布参数（用于算 Loss）
    list(recon = recon, mu = mu, log_var = log_var)
  }
)

# --- 损失函数 ---
# VAE Loss = Reconstruction Loss + KL Divergence
loss_vae <- function(preds, target) {
  # 1. 重构误差 (MSE)
  recon_loss <- nnf_mse_loss(preds$recon, target, reduction = "sum")
  
  # 2. KL 散度 (解析解)
  # 衡量潜在分布与标准正态分布 N(0, 1) 的差异
  kl_loss <- -0.5 * torch_sum(1 + preds$log_var - preds$mu^2 - torch_exp(preds$log_var))
  
  # 返回平均损失 (per batch)
  (recon_loss + kl_loss) / target$size(1)
}

# ==========================================
# 4. 训练模型
# ==========================================

fitted <- VAE %>% 
  setup(
    loss = loss_vae,
    optimizer = optim_adamw
  ) %>% 
  set_hparams(input_dim = 140, latent_dim = 8) %>% 
  set_opt_hparams(lr = 1e-3) %>% 
  fit(
    train_dl, 
    epochs = 50, # 演示用 50，实际可更多
    valid_data = valid_dl,
    verbose = TRUE
  )

# 绘制 Loss 曲线
plot(fitted)

# ==========================================
# 5. 推理与异常检测 (Inference)
# ==========================================

# 1. 准备模型
trained_model <- fitted$model
trained_model$eval() # 切换到评估模式

# 2. 准备数据 Tensor
x_val_tensor <- torch_tensor(x_valid_scaled, dtype = torch_float())

# 3. 设备对齐 (非常重要，防止 M1/CUDA 报错)
device <- trained_model$parameters[[1]]$device
x_val_tensor <- x_val_tensor$to(device = device)

# 4. 执行预测
with_no_grad({
  preds_list <- trained_model(x_val_tensor)
})

# 5. 提取重构结果 (转回 CPU 和 Matrix)
reconstructions <- as.matrix(preds_list$recon$cpu())

# 6. 计算异常得分 (MAE - Mean Absolute Error)
# 逻辑：正常样本重构误差小，异常样本重构误差大
abs_diff <- abs(x_valid_scaled - reconstructions)
anomaly_scores <- rowMeans(abs_diff)

# ==========================================
# 6. 结果可视化
# ==========================================
library(ggplot2)
library(ggsci)

# --- 图 1: 异常得分分布直方图 ---
plot_df <- data.frame(
  score = anomaly_scores,
  label = factor(ifelse(y_valid_labels == 0, "Normal", "Anomaly"))
)

p1 <- ggplot(plot_df, aes(x = score, fill = label)) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity", color = "white") +
  scale_fill_manual(values = c("Normal" = "#1f77b4", "Anomaly" = "#d62728")) +
  labs(
    x = "异常得分 (MAE)",
    y = "数量"
  ) +
  scale_color_npg() + 
  theme_minimal() +
  theme(
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22, face = "bold"),
    strip.text = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    legend.position = "right"
  )

print(p1)

# --- 图 2: 个案微观对比 ---
# 挑选得分最低（最正常）和最高（最异常）的样本
idx_best  <- which.min(anomaly_scores)
idx_worst <- which.max(anomaly_scores)

# 辅助函数：提取画图数据
get_plot_data <- function(row_idx, raw_matrix, recon_matrix, title_suffix) {
  tibble(
    Time = 1:ncol(raw_matrix),
    Original = raw_matrix[row_idx, ],
    Reconstructed = recon_matrix[row_idx, ]
  ) %>% 
    pivot_longer(-Time, names_to = "Type", values_to = "Value") %>% 
    mutate(Case = paste0("Sample ", row_idx, ": ", title_suffix))
}

df_best  <- get_plot_data(idx_best, x_valid_scaled, reconstructions, "正确还原 (正常心跳)")
df_worst <- get_plot_data(idx_worst, x_valid_scaled, reconstructions, "失败还原 (异常心跳)")
df_cases <- bind_rows(df_best, df_worst)



p2 <- ggplot(df_cases, aes(x = Time, y = Value, color = Type)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~Case, scales = "free_y", ncol = 2) +
  theme_bw() +
  labs(
    y = "标准化值",
    x = "时间步"
  ) +
  # 直接使用 Nature 杂志的配色
  scale_color_npg() + 
  # 或者使用新英格兰医学杂志配色: 
  # scale_color_nejm() +
  # 或者使用柳叶刀配色: 
  # scale_color_lancet() + 
  theme(
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22, face = "bold"),
    strip.text = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    legend.position = "right"
  )

print(p2)
