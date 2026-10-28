library(torch)
library(tidyverse)
library(showtext)
library(magick)
library(ggsci)

# --- 全局设置 ---
set.seed(42)
torch_manual_seed(42)
showtext_auto()

device <- if (backends_mps_is_available()) {
  torch_device("mps") # Mac M1/M2/M3
} else if (cuda_is_available()) {
  torch_device("cuda") # NVIDIA GPU
} else {
  torch_device("cpu")
}
cat("Using device:", device$type, "\n")

# ==========================================
# 1. 准备数据 (Data Preparation)
# ==========================================
# 书中注：为了简化代码，可以跳过此函数，直接读取处理好的 r_logo.csv
make_r_logo <- function(n_samples = 40000) {
  url <- "https://www.r-project.org/logo/Rlogo.svg"
  # 读取并预处理图片
  img <- image_read(url) %>%
    image_convert(colorspace = "gray") %>%
    image_resize("200x200")
  
  raw_mat <- image_data(img)[1, , ]
  w <- dim(raw_mat)[1]
  h <- dim(raw_mat)[2]
  
  # 像素转权重：颜色越深(值越小)，权重越大
  pixel_vals <- as.integer(raw_mat)
  weights <- 255 - pixel_vals
  weights[weights < 20] <- 0 # 去噪
  
  # 采样坐标
  df_grid <- expand.grid(x = 1:w, y = 1:h)
  idx <- sample(1:length(weights),
                n_samples,
                replace = TRUE,
                prob = weights)
  sample_data <- df_grid[idx, ]
  
  # 归一化到 [-1, 1] 并翻转 Y 轴
  x <- (sample_data$x - 1) / (w - 1) * 2 - 1
  y <- -((sample_data$y - 1) / (h - 1) * 2 - 1)
  
  # 加微量抖动 (De-quantization)
  x <- x + rnorm(n_samples, sd = 0.003)
  y <- y + rnorm(n_samples, sd = 0.003)
  
  torch_tensor(matrix(c(x, y), ncol = 2), dtype = torch_float32())
}

# ==========================================
# 2. 扩散调度器 (Diffusion Schedule)
# ==========================================
n_steps <- 1000

# 线性调度 (Linear Schedule)
# 简单直观：噪声方差 beta 从 0.0001 线性增加到 0.02
betas <- torch_linspace(1e-4, 0.02, n_steps)$to(device = device)

# 计算 alpha 及其累乘 (Pre-calculate Alphas)
alphas <- 1 - betas
alphas_bar <- torch_cumprod(alphas, dim = 1)

# ==========================================
# 3. 扩散过程函数 (Functional Diffusion)
# ==========================================

# --- 前向过程 (Forward / q_sample) ---
# 公式：x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
q_sample <- function(x0, t) {
  # x0: (batch, 2)
  # t:  (batch) 这里的 t 是索引
  
  noise <- torch_randn_like(x0)
  
  # 提取对应时间步的系数，并调整形状为 (batch, 1) 以便广播
  sqrt_ab <- torch_sqrt(alphas_bar[t])$view(c(-1, 1))
  sqrt_one_minus_ab <- torch_sqrt(1 - alphas_bar[t])$view(c(-1, 1))
  
  # 混合信号与噪声
  xt <- sqrt_ab * x0 + sqrt_one_minus_ab * noise
  
  list(xt = xt, noise = noise)
}

# --- 逆向过程 (Reverse / p_sample) ---
# 这一步用于推理：给定 xt 和 t，算出 x_{t-1}
p_sample <- function(model, xt, t_idx) {
  # t_idx 是一个整数 (例如 300)
  
  # 1. 构造时间张量
  batch_size <- xt$size(1)
  t_tensor <- torch_full(c(batch_size),
                         t_idx,
                         dtype = torch_float(),
                         device = xt$device)
  
  # 2. 预测噪声 (Model Prediction)
  with_no_grad({
    pred_noise <- model(xt, t_tensor)
  })
  
  # 3. 获取系数
  beta_t <- betas[t_idx]
  alpha_t <- alphas[t_idx]
  alpha_hat_t <- alphas_bar[t_idx]
  
  # 4. 计算均值 (Mean) - DDPM 核心公式
  # mean = 1/sqrt(alpha) * (xt - beta / sqrt(1-alpha_bar) * noise)
  coef <- beta_t / torch_sqrt(1 - alpha_hat_t)
  mean <- (1 / torch_sqrt(alpha_t)) * (xt - coef * pred_noise)
  
  # 5. 加回方差 (Variance)
  if (t_idx > 1) {
    z <- torch_randn_like(xt)
    sigma <- torch_sqrt(beta_t)
    x_prev <- mean + sigma * z
  } else {
    x_prev <- mean # 最后一步不加噪声
  }
  
  x_prev
}

# ==========================================
# 4. 神经网络模型 (Neural Network)
# ==========================================
# 保持原有的 Time-Aware MLP 结构，这非常适合 2D 点云

PositionEmbeddings <- nn_module(
  "PositionEmbeddings",
  initialize = function(dim) {
    self$dim <- dim
  },
  forward = function(time) {
    device <- time$device
    half_dim <- self$dim %/% 2
    embeddings <- log(10000) / (half_dim - 1)
    embeddings <- torch_exp(
      torch_arange(0, half_dim - 1, device = device) * -embeddings)
    embeddings <- time$unsqueeze(2) * embeddings$unsqueeze(1)
    torch_cat(list(torch_sin(embeddings),
                   torch_cos(embeddings)), dim = -1)
  }
)

TimeAwareResBlock <- nn_module(
  "TimeAwareResBlock",
  initialize = function(hidden_dim, time_dim, dropout = 0.1) {
    self$mlp1 <- nn_sequential(nn_layer_norm(hidden_dim),
                               nn_gelu(),
                               nn_linear(hidden_dim, hidden_dim))
    self$time_proj <- nn_sequential(nn_gelu(),
                                    nn_linear(time_dim, hidden_dim)
                                    ) # 投影时间
    self$mlp2 <- nn_sequential(
      nn_layer_norm(hidden_dim),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden_dim, hidden_dim)
    )
  },
  forward = function(x, t_emb) {
    h <- self$mlp1(x)
    h <- h + self$time_proj(t_emb) # 注入时间
    x + self$mlp2(h) # 残差连接
  }
)

NoisePredictor <- nn_module(
  "NoisePredictor",
  initialize = function(dim_in = 2,
                        time_dim = 64,
                        hidden_dim = 256,
                        num_blocks = 3) {
    self$input_proj <- nn_linear(dim_in, hidden_dim)
    self$time_mlp <- nn_sequential(
      PositionEmbeddings(time_dim),
      nn_linear(time_dim, time_dim * 4),
      nn_gelu(),
      nn_linear(time_dim * 4, time_dim)
    )
    self$blocks <- nn_module_list()
    for (i in 1:num_blocks)
      self$blocks$append(TimeAwareResBlock(hidden_dim, time_dim))
    self$final_mlp <- nn_sequential(nn_layer_norm(hidden_dim),
                                    nn_gelu(),
                                    nn_linear(hidden_dim, dim_in))
  },
  forward = function(x, t) {
    t_emb <- self$time_mlp(t$to(dtype = torch_float()))
    x <- self$input_proj(x)
    for (i in 1:length(self$blocks))
      x <- self$blocks[[i]](x, t_emb)
    self$final_mlp(x)
  }
)

# ==========================================
# 5. 训练主循环 (Training Loop)
# ==========================================

# 初始化
dataset <- make_r_logo(10000)$to(device = device)
model <- NoisePredictor()$to(device = device)

batch_size <- 2048
lr <- 1e-3
epochs <- 5000 # 演示用，可根据需要调整

optimizer <- optim_adamw(model$parameters, lr = lr)
# OneCycle 调度器有助于快速收敛
scheduler <- lr_one_cycle(
  optimizer,
  max_lr = lr,
  epochs = epochs,
  steps_per_epoch = 1,
  pct_start = 0.2
)

cat("开始训练...\n")

for (epoch in 1:epochs) {
  model$train()
  optimizer$zero_grad()
  
  # 1. 采样 Batch
  idx <- sample(1:nrow(dataset), batch_size)
  x0 <- dataset[idx, ]
  
  # 2. 随机采样时间步 t
  t <- torch_randint(
    low = 1,
    high = n_steps + 1,
    size = c(batch_size),
    dtype = torch_long(),
    device = device
  )
  
  # 3. 前向过程 (核心函数调用)
  forward_out <- q_sample(x0, t)
  xt <- forward_out$xt
  real_noise <- forward_out$noise
  
  # 4. 预测与 Loss
  pred_noise <- model(xt, t)
  loss <- nnf_mse_loss(pred_noise, real_noise)
  
  # 5. 反向传播
  loss$backward()
  nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
  optimizer$step()
  scheduler$step()
  
  if (epoch %% 500 == 0 || epoch == 1) {
    cat(sprintf("Epoch %4d | Loss: %.5f\n", epoch, loss$item()))
  }
}

# ==========================================
# 6. 生成与可视化 (Sampling & Plotting)
# ==========================================
cat("开始生成...\n")
model$eval()

# 策略：非线性采样，捕捉后期的细节变化
steps_to_capture <- seq(sqrt(n_steps), sqrt(1), length.out = 9)^2 %>%
  round() %>% unique()

# 从纯噪声开始
current_x <- torch_randn(2000, 2, device = device)
history <- list()

# 倒计时循环 (t: n_steps -> 1)
pb <- txtProgressBar(min = 0, max = n_steps, style = 3)
for (t in seq(n_steps, 1)) {
  # 核心函数调用：一步步去噪
  current_x <- p_sample(model, current_x, t)
  
  if (t %in% steps_to_capture) {
    df <- as.data.frame(as.matrix(current_x$cpu()))
    df$step <- sprintf("Step: %03d", t)
    history[[length(history) + 1]] <- df
  }
  setTxtProgressBar(pb, n_steps - t + 1)
}
close(pb)


# 绘图
plot_data <- bind_rows(history)
# 确保绘图顺序正确
plot_data$step <- factor(plot_data$step, levels = unique(plot_data$step))

ggplot(plot_data, aes(V1, V2)) +
  geom_point(alpha = 0.5,
             size = 0.3,
             color = "#1879bf") +
  coord_fixed(xlim = c(-1.2, 1.2), ylim = c(-1.2, 1.2)) +
  facet_wrap( ~ step, nrow = 3) +
  theme_void() +
  theme(
    strip.text = element_text(
      size = 9,
      face = "bold",
      margin = margin(b = 5)
    ),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  labs(title = "Diffusion Process: Restoring the R Logo")


### 优化版  ###
### 1. 加上调度器
### 2. 增加 R6 类的定义

library(torch)
library(tidyverse)
library(showtext)

set.seed(42)
torch_manual_seed(42)
showtext_auto()

# --- 1. Beta 调度器 (Cosine Schedule) ---
# 相比 Linear，Cosine 能在中间段保留更多信号，极大提升生成质量
get_cosine_schedule <- function(n_steps, s = 0.008) {
  steps <- 0:n_steps
  # 计算 f(t)
  f_t <- torch_cos(((steps / n_steps + s) / (1 + s)) * (pi / 2))^2
  # alpha_bar = f(t) / f(0)
  alpha_bar <- f_t / f_t[1]
  
  # beta = 1 - (alpha_bar_t / alpha_bar_{t-1})
  betas <- torch_zeros(n_steps)
  for (i in 2:(n_steps + 1)) {
    betas[i - 1] <- 1 - (alpha_bar[i] / alpha_bar[i - 1])
  }
  # 截断防止数值不稳定
  torch_clamp(betas, min = 0, max = 0.999)
}

# --- 2. 增强的数据处理 ---
library(magick)

make_r_logo <- function(n_samples = 40000) {
  url <- "https://www.r-project.org/logo/Rlogo.svg"
  img <- image_read(url) %>%
    image_convert(colorspace = "gray") %>%
    image_resize("200x200")
  
  # 取第一个通道 [1,,] 得到的是一个 [width, height] 的 raw matrix
  raw_mat <- image_data(img)[1, , ]
  
  # 注意：magick 的数据矩阵通常第一维是宽(x)，第二维是高(y)
  w <- dim(raw_mat)[1]
  h <- dim(raw_mat)[2]
  
  # 转为整数 (这步会把它拉平成一维向量，但我们已经存好了 w 和 h)
  pixel_vals <- as.integer(raw_mat)
  
  # 3. 建立网格坐标
  # expand.grid 的顺序要和 pixel_vals 的排列顺序对应
  # R 的矩阵是按列优先填充 (Column Major)，即先变第一维(x)，再变第二维(y)
  df_grid <- expand.grid(x = 1:w, y = 1:h)
  
  # 4. 计算权重 (颜色越深，权重越大)
  weights <- 255 - pixel_vals
  
  # 去噪：把接近白色的背景权重设为 0
  weights[weights < 20] <- 0
  
  # 5. 采样
  idx <- sample(1:length(weights),
                n_samples,
                replace = TRUE,
                prob = weights)
  sample_data <- df_grid[idx, ]
  
  # 6. 坐标变换
  x <- sample_data$x
  y <- sample_data$y
  
  # 翻转 Y 轴 (因为图片 (1,1) 在左上角，而 plot (0,0) 在左下角)
  y <- -y
  
  # 7. 归一化到 [-1, 1]
  x <- (x - min(x)) / (max(x) - min(x)) * 2 - 1
  y <- (y - min(y)) / (max(y) - min(y)) * 2 - 1
  
  # 8. 加抖动 (De-quantization)
  x <- x + rnorm(n_samples, sd = 0.003)
  y <- y + rnorm(n_samples, sd = 0.003)
  
  data_matrix <- matrix(c(x, y), ncol = 2)
  
  # 转为 Tensor
  torch_tensor(data_matrix, dtype = torch_float32())
}

test_data <- make_r_logo(5000)
df_test <- as_tibble(as.matrix(test_data))

ggplot(df_test, aes(V1, V2)) +
  geom_point(size = 0.7,
             alpha = 0.5,
             color = "#1879bf") + # R Logo blue
  coord_fixed() +
  theme_void() +
  ggtitle("")

# --- 3. 正弦位置嵌入 (保持不变) ---
PositionEmbeddings <- nn_module(
  "PositionEmbeddings",
  initialize = function(dim) {
    self$dim <- dim
  },
  forward = function(time) {
    device <- time$device
    half_dim <- self$dim %/% 2
    embeddings <- log(10000) / (half_dim - 1)
    embeddings <- torch_exp(torch_arange(0, half_dim - 1, device = device) * -embeddings)
    embeddings <- time$unsqueeze(2) * embeddings$unsqueeze(1)
    embeddings <- torch_cat(list(torch_sin(embeddings), torch_cos(embeddings)), dim = -1)
    embeddings
  }
)

# --- 4. 时间感知残差块 (Time-Aware ResBlock) ---
# 每一层都吃进 time_emb，确保时间信号不衰减
TimeAwareResBlock <- nn_module(
  "TimeAwareResBlock",
  initialize = function(hidden_dim, time_dim, dropout_prob = 0.1) {
    self$mlp1 <- nn_sequential(nn_layer_norm(hidden_dim),
                               nn_gelu(),
                               nn_linear(hidden_dim, hidden_dim))
    
    # 专门处理时间的层，将其映射到与 hidden_dim 一致，方便相加
    self$time_mlp <- nn_sequential(nn_gelu(), nn_linear(time_dim, hidden_dim))
    
    self$mlp2 <- nn_sequential(
      nn_layer_norm(hidden_dim),
      nn_gelu(),
      nn_dropout(dropout_prob),
      # 加一点 Dropout 防止死记硬背
      nn_linear(hidden_dim, hidden_dim)
    )
  },
  forward = function(x, t_emb) {
    # x: (batch, hidden)
    # t_emb: (batch, time_dim)
    
    h <- self$mlp1(x)
    
    # 【注入时间信号】
    # 将时间特征变换后，直接加到当前特征上
    # 这让网络在每一层都知道“现在是第几步”
    time_feature <- self$time_mlp(t_emb)
    h <- h + time_feature
    
    h <- self$mlp2(h)
    
    # 残差连接
    x + h
  }
)

# --- 5. 模型主干 ---
NoisePredictor <- nn_module(
  "NoisePredictor",
  initialize = function(dim_in = 2,
                        time_dim = 64,
                        hidden_dim = 256,
                        num_blocks = 4) {
    # 输入投影
    self$input_proj <- nn_linear(dim_in, hidden_dim)
    
    # 时间嵌入
    self$time_mlp <- nn_sequential(
      PositionEmbeddings(time_dim),
      nn_linear(time_dim, time_dim * 4),
      nn_gelu(),
      nn_linear(time_dim * 4, time_dim)
    )
    
    # 堆叠 Time-Aware ResBlocks
    self$blocks <- nn_module_list()
    for (i in 1:num_blocks) {
      self$blocks$append(TimeAwareResBlock(hidden_dim, time_dim))
    }
    
    # 输出头
    self$final_mlp <- nn_sequential(nn_layer_norm(hidden_dim),
                                    nn_gelu(),
                                    nn_linear(hidden_dim, dim_in))
  },
  forward = function(x, t) {
    # 1. 获得时间嵌入 (batch, time_dim)
    t_emb <- self$time_mlp(t$to(dtype = torch_float()))
    
    # 2. 初始特征映射 (batch, hidden)
    x <- self$input_proj(x)
    
    # 3. 逐层传递，每一层都注入 t_emb
    for (i in 1:length(self$blocks)) {
      x <- self$blocks[[i]](x, t_emb)
    }
    
    # 4. 输出
    self$final_mlp(x)
  }
)

# --- 6. 扩散管理器 (适配 Cosine Schedule) ---
DiffusionManager <- R6::R6Class(
  "DiffusionManager",
  public = list(
    n_steps = NULL,
    beta = NULL,
    alpha = NULL,
    alpha_hat = NULL,
    
    initialize = function(n_steps) {
      self$n_steps <- n_steps
      self$beta <- get_cosine_schedule(n_steps)
      self$alpha <- 1 - self$beta
      self$alpha_hat <- torch_cumprod(self$alpha, dim = 1)
    },
    
    sample_forward = function(x0, t) {
      sqrt_alpha_hat <- torch_sqrt(self$alpha_hat[t])$view(c(-1, 1))
      sqrt_one_minus_alpha_hat <- torch_sqrt(1 - self$alpha_hat[t])$view(c(-1, 1))
      noise <- torch_randn_like(x0)
      xt <- sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
      list(xt = xt, noise = noise)
    },
    
    sample_backward = function(model, xt, t_idx) {
      batch_size <- xt$size(1)
      
      t_tensor <- torch_full(c(batch_size),
                             t_idx,
                             dtype = torch_float(),
                             device = xt$device)
      
      with_no_grad({
        predicted_noise <- model(xt, t_tensor)
      })
      
      beta_t <- self$beta[t_idx]
      alpha_t <- self$alpha[t_idx]
      alpha_hat_t <- self$alpha_hat[t_idx]
      
      coef <- beta_t / torch_sqrt(1 - alpha_hat_t)
      mean <- 1 / torch_sqrt(alpha_t) * (xt - coef * predicted_noise)
      
      if (t_idx > 1) {
        z <- torch_randn_like(xt)
        sigma <- torch_sqrt(beta_t)
        x_prev <- mean + sigma * z
      } else {
        x_prev <- mean
      }
      x_prev
    }
  )
)

# --- 7. 主程序 ---

# 超参数
n_steps <- 300
batch_size <- 4096    # 加大 Batch Size，让梯度更准
num_epochs <- 10000
lr <- 1e-3

# 准备
device <- if (backends_mps_is_available())
  torch_device("mps") else torch_device("cpu")
cat("Using device:", device$type, "\n")

# --- 2. 初始化 ---
dataset <- make_r_logo(10000)

diffusion <- DiffusionManager$new(n_steps = n_steps)
diffusion$beta <- diffusion$beta$to(device = device)
diffusion$alpha <- diffusion$alpha$to(device = device)
diffusion$alpha_hat <- diffusion$alpha_hat$to(device = device)

model <- NoisePredictor(
  dim_in = 2,
  time_dim = 64,
  hidden_dim = 256,
  num_blocks = 4
)
model$to(device = device)

optimizer <- optim_adamw(model$parameters, lr = lr, weight_decay = 1e-5)
scheduler <- lr_one_cycle(
  optimizer,
  max_lr = lr,
  epochs = num_epochs,
  steps_per_epoch = 1,
  pct_start = 0.1
)

cat(sprintf("开始训练 (Device: %s)...\n", device$type))

# --- 3. 训练循环 ---
for (epoch in 1:num_epochs) {
  model$train()
  optimizer$zero_grad()
  idx <- sample(1:nrow(dataset), batch_size)
  x0_cpu <- dataset[idx, ]
  x0 <- x0_cpu$to(device = device)
  t <- torch_randint(
    low = 1,
    high = n_steps + 1,
    size = c(batch_size),
    dtype = torch_long(),
    device = device
  )
  forward_out <- diffusion$sample_forward(x0, t)
  xt <- forward_out$xt
  real_noise <- forward_out$noise
  
  pred_noise <- model(xt, t)
  
  loss <- nnf_mse_loss(pred_noise, real_noise)
  loss$backward()
  nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0) # 防止梯度爆炸
  optimizer$step()
  scheduler$step()
  
  if (epoch %% 500 == 0 || epoch == 1) {
    cat(sprintf("Epoch %5d | Loss: %.6f\n", epoch, loss$item()))
  }
}

# --- 8. 生成与可视化 ---

cat("开始生成 (MPS)...\n")
model$eval() # 确保模型处于推理模式
n_gen <- 2000
# 使用非线性采样，捕捉更多后期的精彩变化
steps_to_capture <- seq(sqrt(n_steps), sqrt(1), length.out = 8)^2 %>%
  round() %>%
  unique()
history_data <- list()
current_x <- torch_randn(n_gen, 2, device = device)
pb <- txtProgressBar(min = 0, max = n_steps, style = 3)

for (t in seq(n_steps, 1)) {
  current_x <- diffusion$sample_backward(model, current_x, t)
  setTxtProgressBar(pb, n_steps - t + 1)
  
  # 捕捉逻辑
  if (t %in% steps_to_capture) {
    df <- as_tibble(as.matrix(current_x$cpu()))
    # 为了让 step 名字好看点，补齐 4 位数
    step_label <- sprintf("Step: %03d", t)
    df$step <- step_label
    history_data[[step_label]] <- df
  }
}
close(pb)

# 绘图部分也稍微改一下排序逻辑，确保 facet_wrap 顺序对
plot_data <- bind_rows(history_data)
# 按照步数倒序排列 (1000 -> 1)
plot_data$step <- factor(plot_data$step, levels = unique(plot_data$step))

p <- ggplot(plot_data, aes(V1, V2)) +
  geom_point(alpha = 0.5,
             size = 0.1,
             color = "#1879bf") +
  coord_fixed(xlim = c(-1.2, 1.2), ylim = c(-1.2, 1.2)) +
  facet_wrap( ~ step, nrow = 2) + # 自动换行
  theme_void() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_npg() + 
  theme(
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22, face = "bold"),
    strip.text = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    legend.position = "right"
  )

print(p)

