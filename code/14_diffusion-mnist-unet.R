library(torch)
library(torchvision)

# --- 1. 配置与设备 ---
device <- if (backends_mps_is_available()) {
  torch_device("mps") # Mac M1/M2/M3
} else if (cuda_is_available()) {
  torch_device("cuda") # NVIDIA GPU
} else {
  torch_device("cpu")
}

batch_size <- 128
n_steps <- 1000
image_size <- 28
channels <- 1

# --- 2. 数据准备 (MNIST) ---
# 变换：转 Tensor -> 归一化到 [-1, 1]
transform <- function(x) {
  x <- transform_to_tensor(x)
  (x - 0.5) * 2
}

# 第一次运行会自动下载 MNIST
data_root <- "~/data/mnist" 
ds <- mnist_dataset(
  root = data_root,
  train = TRUE,
  download = TRUE,
  transform = transform
)
dl <- dataloader(ds, batch_size = batch_size, shuffle = TRUE)

# --- 3. 扩散调度器 (Schedule) & 辅助函数 ---
betas <- torch_linspace(1e-4, 0.02, n_steps)$to(device = device)
alphas <- 1 - betas
alphas_bar <- torch_cumprod(alphas, dim = 1)

# 作用：把形状为 (Batch) 的时间系数，变成 (Batch, 1, 1, 1)
# 这样才能和图像 (Batch, Channel, Height, Width) 进行广播运算
extract <- function(a, t, x_shape) {
  batch_size <- t$size(1)
  
  # 1. 确保 t 是长整型 (MPS 要求)
  t_long <- t$to(dtype = torch_long())
  
  # a 是 [1000], t 是 [B], a[t] 会直接返回 [B]
  out <- a[t_long]
  
  # 3. 调整维度为 (B, 1, 1, 1) 以便与图像广播
  out$view(c(batch_size, 1, 1, 1))
}

# --- 4. 简易 U-Net 模型 ---

# 4.1 时间编码 (同上一节)
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
    torch_cat(list(torch_sin(embeddings), torch_cos(embeddings)), dim = -1)
  }
)

# 4.2 基础卷积块 (带时间注入)
# 结构：Conv -> GroupNorm -> SiLU + Time_Emb
Block <- nn_module(
  "Block",
  initialize = function(dim, time_emb_dim) {
    self$proj <- nn_conv2d(dim, dim, 3, padding = 1)
    self$norm <- nn_group_norm(32, dim)
    self$act <- nn_silu()
    # 时间投影层
    self$time_mlp <- nn_sequential(nn_silu(), nn_linear(time_emb_dim, dim))
  },
  forward = function(x, t_emb) {
    h <- self$proj(x)
    h <- self$norm(h)
    
    # 注入时间：(B, dim) -> (B, dim, 1, 1) -> 加到特征图上
    time_h <- self$time_mlp(t_emb)$unsqueeze(-1)$unsqueeze(-1)
    h <- h + time_h
    
    self$act(h)
  }
)

# 4.3 U-Net 主体
SimpleUNet <- nn_module(
  "SimpleUNet",
  initialize = function(in_channels = 1,
                        time_dim = 32,
                        base_dim = 32) {
    # 时间处理
    self$time_mlp <- nn_sequential(
      PositionEmbeddings(time_dim),
      nn_linear(time_dim, time_dim * 4),
      nn_gelu(),
      nn_linear(time_dim * 4, time_dim)
    )
    
    # 初始卷积
    self$init_conv <- nn_conv2d(in_channels, base_dim, 3, padding = 1)
    
    # --- Down (下采样) ---
    self$down1 <- Block(base_dim, time_dim)
    self$down2 <- Block(base_dim * 2, time_dim)
    self$to_down2 <- nn_conv2d(base_dim,
                               base_dim * 2,
                               4,
                               stride = 2,
                               padding = 1) # 28->14
    
    self$down3 <- Block(base_dim * 4, time_dim)
    self$to_down3 <- nn_conv2d(base_dim * 2,
                               base_dim * 4,
                               4,
                               stride = 2,
                               padding = 1) # 14->7
    
    # --- Mid (瓶颈层) ---
    self$mid1 <- Block(base_dim * 4, time_dim)
    self$mid2 <- Block(base_dim * 4, time_dim)
    
    # --- Up (上采样) ---
    # Up1: 7->14. Input: 128 + 128 (skip) = 256 -> 64
    self$up1 <- nn_conv_transpose2d(base_dim * 4,
                                    base_dim * 2,
                                    4,
                                    stride = 2,
                                    padding = 1)
    self$up_block1 <- Block(base_dim * 4, time_dim) # 接收拼接后的特征
    
    # Up2: 14->28. Input: 64 + 32 (skip) = 96 -> 32
    self$up2 <- nn_conv_transpose2d(base_dim * 4,
                                    base_dim,
                                    4,
                                    stride = 2,
                                    padding = 1)
    self$up_block2 <- Block(base_dim * 2, time_dim)
    
    # 输出
    self$out_conv <- nn_conv2d(base_dim * 2, in_channels, 3, padding = 1)
  },
  
  forward = function(x, t) {
    t_emb <- self$time_mlp(t)
    
    # Initial
    x1 <- self$init_conv(x) # (B, 32, 28, 28)
    
    # Down
    h1 <- self$down1(x1, t_emb)         # Skip 1
    h2 <- self$to_down2(h1)             # (B, 64, 14, 14)
    h2 <- self$down2(h2, t_emb)         # Skip 2
    h3 <- self$to_down3(h2)             # (B, 128, 7, 7)
    h3 <- self$down3(h3, t_emb)
    
    # Mid
    h_mid <- self$mid1(h3, t_emb)
    h_mid <- self$mid2(h_mid, t_emb)
    
    # Up (Skip Connection 这里发生)
    # R torch 的 cat dim=2 代表 Channel (N=1, C=2, H=3, W=4)
    
    # Up 1
    h_up1 <- self$up1(h_mid)            # (B, 64, 14, 14)
    h_cat1 <- torch_cat(list(h_up1, h2), dim = 2) # Concat
    h_res1 <- self$up_block1(h_cat1, t_emb) # (B, 128, ...) -> Block 内部 output channel 是 input 保持不变
    
    # Up 2
    h_up2 <- self$up2(h_res1)           # (B, 32, 28, 28)
    h_cat2 <- torch_cat(list(h_up2, h1), dim = 2)
    h_res2 <- self$up_block2(h_cat2, t_emb)
    
    self$out_conv(h_res2)
  }
)

# --- 5. 训练循环 ---

model <- SimpleUNet()$to(device = device)
optimizer <- optim_adam(model$parameters, lr = 1e-3)
epochs <- 10 # 演示用，实际建议 10+

cat("Starting training on MNIST...\n")
for (epoch in 1:epochs) {
  losses <- c()
  
  coro::loop(for (batch in dl) {
    optimizer$zero_grad()
    
    # 1. 准备数据
    x0 <- batch$x$to(device = device)
    
    # 【修复】确保维度是 (Batch, Channel, H, W)
    # 某些版本的 torchvision 加载 MNIST 可能是 (B, 28, 28)，导致广播错误
    if (length(x0$shape) == 3) {
      x0 <- x0$unsqueeze(2)
    }
    
    batch_n <- x0$size(1)
    
    # 显式指定 dtype 防止 MPS 报错
    t <- torch_randint(
      1,
      n_steps + 1,
      size = c(batch_n),
      dtype = torch_long(),
      device = device
    )
    
    # 2. 前向加噪
    noise <- torch_randn_like(x0)
    
    # 使用修正后的 extract
    sqrt_ab <- extract(torch_sqrt(alphas_bar), t, x0$shape)
    sqrt_om_ab <- extract(torch_sqrt(1 - alphas_bar), t, x0$shape)
    
    # 现在维度对齐了：[128,1,1,1] * [128,1,28,28] -> [128,1,28,28]
    xt <- sqrt_ab * x0 + sqrt_om_ab * noise
    
    # 3. 预测噪声
    pred_noise <- model(xt, t)
    
    # 4. Loss
    loss <- nnf_mse_loss(pred_noise, noise)
    loss$backward()
    optimizer$step()
    
    losses <- c(losses, loss$item())
  })
  cat(sprintf("Epoch %d | Avg Loss: %.4f\n", epoch, mean(losses)))
}

# --- 6. 采样与可视化 ---

# 逆向采样函数 (专门处理 4D 图像)
p_sample_loop <- function(model, n_samples) {
  model$eval()
  # 从纯噪声开始
  img <- torch_randn(n_samples, 1, 28, 28, device = device)
  
  for (i in seq(n_steps, 1)) {
    t <- torch_full(c(n_samples), i, device = device, dtype = torch_long())
    
    with_no_grad({
      pred_noise <- model(img, t)
    })
    
    # 获取系数并广播
    beta <- extract(betas, t, img$shape)
    sqrt_one_minus_alpha_hat <- extract(torch_sqrt(1 - alphas_bar), t, img$shape)
    sqrt_inv_alpha <- extract(torch_sqrt(1 / alphas), t, img$shape)
    
    # DDPM 采样公式
    mean <- sqrt_inv_alpha * (img - (beta / sqrt_one_minus_alpha_hat) * pred_noise)
    
    if (i > 1) {
      noise <- torch_randn_like(img)
      sigma <- torch_sqrt(beta)
      img <- mean + sigma * noise
    } else {
      img <- mean
    }
  }
  # 还原到 [0, 1] 以便绘图
  (img$clamp(-1, 1) + 1) / 2
}

## 生成 16 张图
cat("Generating digits...\n")
samples <- p_sample_loop(model, 16)

# 简单绘图
par(mfrow = c(2, 8), mar = c(0.1, 0.1, 0.1, 0.1))
samples_cpu <- samples$cpu()
for (i in 1:16) {
  # 旋转图像以便 R 的 image 函数正确显示
  img_mat <- as.matrix(samples_cpu[i, 1, , ])
  img_mat <- apply(img_mat, 2, rev)
  image(t(img_mat), col = gray.colors(256), axes = FALSE)
}


## 绘制扩散模型去噪渐变轨迹

# --- 重新定义捕获渐变过程的采样函数 ---
p_sample_loop_with_trajectory <- function(model, num_samples, save_intervals = c(1000, 800, 600, 400, 200, 1)) {
  model$eval()
  
  # 1. 从纯高斯噪声开始
  img <- torch_randn(num_samples, 1, 28, 28, device = device)
  
  # 用于存储指定步数的历史图像
  trajectory <- list()
  
  # 2. 倒序去噪大循环 (对应原本的整个 DDPM 采样公式)
  for (i in seq(n_steps, 1)) {
    t <- torch_full(c(num_samples), i, device = device, dtype = torch_long())
    
    with_no_grad({
      pred_noise <- model(img, t)
    })
    
    # 获取系数并广播
    beta <- extract(betas, t, img$shape)
    sqrt_one_minus_alpha_hat <- extract(torch_sqrt(1 - alphas_bar), t, img$shape)
    sqrt_inv_alpha <- extract(torch_sqrt(1 / alphas), t, img$shape)
    
    # DDPM 采样公式
    mean <- sqrt_inv_alpha * (img - (beta / sqrt_one_minus_alpha_hat) * pred_noise)
    
    if (i > 1) {
      noise <- torch_randn_like(img)
      sigma <- torch_sqrt(beta)
      img <- mean + sigma * noise
    } else {
      img <- mean
    }
    
    # 3. 关键的拦截器：如果当前迭代步在指定的观察节点里，立刻截图保存一份
    if (i %in% save_intervals) {
      # 还原到 [0, 1] 区间以便绘图
      img_clean <- (img$clamp(-1, 1) + 1) / 2
      # 搬回 CPU
      trajectory[[as.character(i)]] <- img_clean$cpu()
    }
  }
  
  # 返回最终生成结果和中间轨迹数据
  return(list(final = (img$clamp(-1, 1) + 1) / 2, trajectory = trajectory))
}

#' @param trajectory_list 包含不同时间步图像的列表，每个元素的形状为 (B, 1, 28, 28)
#' @param num_samples_to_show 纵向展示几个样本（默认前 4 个）
#' @param steps_to_show 横向展示哪些时间步
plot_diffusion_trajectory <- function(trajectory_list, 
                                      num_samples_to_show = 4, 
                                      steps_to_show = c("1000", "800", "600", "400", "200", "1")) {
  
  n_cols <- length(steps_to_show)
  n_rows <- num_samples_to_show
  
  # 【优化点 1】将子图内边距 mar 设为 0（仅给顶部留一点点标题空间），加大外边距 oma 留给最外层的标签
  par(mfrow = c(n_rows, n_cols), 
      mar = c(0, 0, 1.0, 0), 
      oma = c(1, 3, 2, 1))
  
  for (row in 1:n_rows) {
    for (col in 1:n_cols) {
      step_key <- steps_to_show[col]
      
      img_tensor <- trajectory_list[[step_key]][row, 1, , ]
      img_matrix <- as.matrix(img_tensor)
      img_matrix <- t(img_matrix[28:1, ]) 
      
      # 【优化点 2】xaxs="i" 和 yaxs="i" 会强行去掉 R 默认在坐标轴两端增加的 4% 空白
      plot(c(0, 1), c(0, 1), type = "n", axes = FALSE, xlab = "", ylab = "",
           xaxs = "i", yaxs = "i", asp = 1)
      
      rasterImage(img_matrix, 0, 0, 1, 1)
      
      # 只在第一行绘制顶部时间步标题
      if (row == 1) {
        title(main = sprintf("t = %s", step_key), cex.main = 1.2, font.main = 2, line = 0.3)
      }
      
      # 只在第一列的左侧绘制样本标签（利用 mtext 绘制在外边距上）
      if (col == 1) {
        mtext(sprintf("Sample %d", row), side = 2, line = 0.4, cex = 1.0, font = 2, las = 0)
      }
    }
  }
  
  # 恢复默认画布配置
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, oma = c(0, 0, 0, 0))
}

# 1. 设置想要观察的时间节点
intervals <- c(1000, 600, 400, 200, 100, 1)

# 2. 生成数字并捕获轨迹
cat("Generating digits with trajectory...\n")
res <- p_sample_loop_with_trajectory(model, num_samples = 16, save_intervals = intervals)

# 3. 调用绘图函数（展示前 4 个样本在这些时间节点的演变）
plot_diffusion_trajectory(res$trajectory, num_samples_to_show = 4,
                          steps_to_show = as.character(intervals))

