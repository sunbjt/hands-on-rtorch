library(torch)
library(torchvision)
library(coro)

# ==========================================
# 1. 配置与设备
# ==========================================
device <- if (backends_mps_is_available()) {
  torch_device("mps") # Mac M1/M2/M3
} else if (cuda_is_available()) {
  torch_device("cuda") # NVIDIA GPU
} else {
  torch_device("cpu")
}
cat("Using device:", device$type, "\n")

# 超参数
batch_size <- 128
n_steps <- 400
image_size <- 28
channels <- 1
base_dim <- 32 
num_classes <- 10 # 0-9 共10个数字
dropout_prob <- 0.1 # 10% 的概率把标签扔掉

# ==========================================
# 2. 数据准备
# ==========================================
transform <- function(x) {
  x <- transform_to_tensor(x)
  (x - 0.5) * 2
}

data_root <- "/Users/liusizhe/data/mnist" 
ds <- mnist_dataset(
  root = data_root,
  train = TRUE,
  download = TRUE,
  transform = transform
)
dl <- dataloader(ds, batch_size = batch_size, shuffle = TRUE)

# ==========================================
# 3. 扩散调度器
# ==========================================
betas <- torch_linspace(1e-4, 0.02, n_steps)$to(device = device)
alphas <- 1 - betas
alphas_bar <- torch_cumprod(alphas, dim = 1)

extract <- function(a, t, x_shape) {
  batch_size <- t$size(1)
  t_long <- t$to(dtype = torch_long())
  out <- a[t_long]
  out$view(c(batch_size, 1, 1, 1))
}

# ==========================================
# 4. 模型定义 (ContextUnet)
# ==========================================

# 4.1 位置编码
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

# 4.2 自注意力 
SelfAttention <- nn_module(
  "SelfAttention",
  initialize = function(channels) {
    self$channels <- channels
    self$norm <- nn_group_norm(32, channels)
    self$qkv <- nn_conv2d(channels, channels * 3, kernel_size = 1)
    self$proj <- nn_conv2d(channels, channels, kernel_size = 1)
  },
  forward = function(x) {
    B <- x$size(1); C <- x$size(2); H <- x$size(3); W <- x$size(4)
    h <- self$norm(x)
    qkv <- self$qkv(h)
    chunks <- torch_chunk(qkv, 3, dim = 2)
    q <- chunks[[1]]; k <- chunks[[2]]; v <- chunks[[3]]
    q <- q$view(c(B, C, -1)); k <- k$view(c(B, C, -1)); v <- v$view(c(B, C, -1))
    
    attn <- torch_bmm(q$permute(c(1, 3, 2)), k) * (C^-0.5)
    attn <- nnf_softmax(attn, dim = -1) # Fix: 使用 nnf_softmax
    out <- torch_bmm(v, attn$permute(c(1, 3, 2)))
    
    out <- out$view(c(B, C, H, W))
    out <- self$proj(out)
    x + out
  }
)

# 4.3 基础块
Block <- nn_module(
  "Block",
  initialize = function(dim, time_emb_dim) {
    self$proj <- nn_conv2d(dim, dim, 3, padding = 1)
    self$norm <- nn_group_norm(32, dim)
    self$act <- nn_silu()
    self$time_mlp <- nn_sequential(nn_silu(), nn_linear(time_emb_dim, dim))
  },
  forward = function(x, t_emb) {
    h <- self$proj(x)
    h <- self$norm(h)
    time_h <- self$time_mlp(t_emb)$unsqueeze(-1)$unsqueeze(-1)
    h <- h + time_h
    self$act(h)
  }
)

# 4.4 带条件的 U-Net
ContextUnet <- nn_module(
  "ContextUnet",
  initialize = function(in_channels = 1, time_dim = 32, base_dim = 64, num_classes = 10) {
    
    # 【新增】类别 Embedding
    # 10个数字 + 1个空标签(用于无条件生成) = 11
    self$label_emb <- nn_embedding(num_classes + 1, time_dim)
    
    self$time_mlp <- nn_sequential(
      PositionEmbeddings(time_dim),
      nn_linear(time_dim, time_dim * 4),
      nn_gelu(),
      nn_linear(time_dim * 4, time_dim)
    )
    
    self$init_conv <- nn_conv2d(in_channels, base_dim, 3, padding = 1)
    self$down1 <- Block(base_dim, time_dim)
    self$to_down2 <- nn_conv2d(base_dim, base_dim * 2, 4, 2, 1)
    self$down2 <- Block(base_dim * 2, time_dim)
    self$to_down3 <- nn_conv2d(base_dim * 2, base_dim * 4, 4, 2, 1)
    self$down3 <- Block(base_dim * 4, time_dim)
#    self$attn_down <- SelfAttention(base_dim * 4)
    
    self$mid1 <- Block(base_dim * 4, time_dim)
#    self$attn_mid <- SelfAttention(base_dim * 4)
    self$mid2 <- Block(base_dim * 4, time_dim)
    
    self$up1 <- nn_conv_transpose2d(base_dim * 4, base_dim * 2, 4, 2, 1)
    self$up_block1 <- Block(base_dim * 4, time_dim)
    self$up2 <- nn_conv_transpose2d(base_dim * 4, base_dim, 4, 2, 1)
    self$up_block2 <- Block(base_dim * 2, time_dim)
    self$out_conv <- nn_conv2d(base_dim * 2, in_channels, 3, padding = 1)
  },
  
  # Forward 接收 x, t, 和 y (labels)
  forward = function(x, t, y) {
    t_emb <- self$time_mlp(t)
    y_emb <- self$label_emb(y)
    
    # 融合：简单相加 (Embedding Add)
    ctx_emb <- t_emb + y_emb
    
    x1 <- self$init_conv(x)
    h1 <- self$down1(x1, ctx_emb)
    h2 <- self$to_down2(h1)
    h2 <- self$down2(h2, ctx_emb)
    h3 <- self$to_down3(h2)
    h3 <- self$down3(h3, ctx_emb)
#    h3 <- self$attn_down(h3)
    
    h_mid <- self$mid1(h3, ctx_emb)
#    h_mid <- self$attn_mid(h_mid)
    h_mid <- self$mid2(h_mid, ctx_emb)
    
    h_up1 <- self$up1(h_mid)
    h_cat1 <- torch_cat(list(h_up1, h2), dim = 2)
    h_res1 <- self$up_block1(h_cat1, ctx_emb)
    
    h_up2 <- self$up2(h_res1)
    h_cat2 <- torch_cat(list(h_up2, h1), dim = 2)
    h_res2 <- self$up_block2(h_cat2, ctx_emb)
    
    self$out_conv(h_res2)
  }
)

# ==========================================
# 5. 训练循环 (含 Label Dropout)
# ==========================================
model <- ContextUnet(num_classes = num_classes, base_dim = base_dim)$to(device = device)
optimizer <- optim_adam(model$parameters, lr = 1e-3)
epochs <- 5 # 建议设为 10 或 20 以获得更好效果

cat("Starting Conditional Training with Label Dropout...\n")

for (epoch in 1:epochs) {
  losses <- c()
  
  coro::loop(for (batch in dl) {
    optimizer$zero_grad()
    x0 <- batch$x$to(device = device)
    # MNIST labels 原始是 1-10 (R torch 特性)，我们直接用
    labels <- batch$y$to(device = device, dtype = torch_long())
    
    if (length(x0$shape) == 3) x0 <- x0$unsqueeze(2)
    batch_n <- x0$size(1)
    
    t <- torch_randint(1, n_steps + 1, size = c(batch_n), dtype = torch_long(), device = device)
    
    # 【核心技巧】Label Dropout
    # 随机生成一个 mask，概率为 0.1
    mask <- torch_rand(batch_n, device = device) < dropout_prob
    # 将被选中的标签设为 11 (空标签)
    labels[mask] <- 11
    
    noise <- torch_randn_like(x0)
    sqrt_ab <- extract(torch_sqrt(alphas_bar), t, x0$shape)
    sqrt_om_ab <- extract(torch_sqrt(1 - alphas_bar), t, x0$shape)
    xt <- sqrt_ab * x0 + sqrt_om_ab * noise
    
    pred_noise <- model(xt, t, labels)
    loss <- nnf_mse_loss(pred_noise, noise)
    
    loss$backward()
    optimizer$step()
    losses <- c(losses, loss$item())
  })
  cat(sprintf("Epoch %d | Avg Loss: %.4f\n", epoch, mean(losses)))
}

# ==========================================
# 6. Classifier-Free Guidance 采样
# ==========================================
sample_conditional <- function(model, n_samples, target_digit, guide_w = 2.0) {
  model$eval()
  
  # target_digit: 0-9. 我们要在 dataset 索引 (1-10) 中找到对应值
  # R mnist dataset: 1->0, 2->1 ... 10->9 ? 通常 R 是 1-based index 对应 class label.
  # 实际上 torch mnist_dataset$classes 顺序是 "0", "1", ...
  # 所以 target_digit 0 对应的 label 是 1。 target_digit 7 对应 label 8。
  label_idx <- target_digit + 1
  
  # 构造条件标签
  y_cond <- torch_full(c(n_samples), label_idx, dtype = torch_long(), device = device)
  # 构造空标签 (11)
  y_uncond <- torch_full(c(n_samples), 11, dtype = torch_long(), device = device)
  
  img <- torch_randn(n_samples, 1, 28, 28, device = device)
  
  cat(sprintf("Sampling digit '%d' with guidance scale %.1f...\n", target_digit, guide_w))
  
  for (i in seq(n_steps, 1)) {
    t <- torch_full(c(n_samples), i, dtype = torch_long(), device = device)
    
    with_no_grad({
      # 双重预测
      noise_cond <- model(img, t, y_cond)
      noise_uncond <- model(img, t, y_uncond)
      
      # CFG 公式: Uncond + w * (Cond - Uncond)
      pred_noise <- noise_uncond + guide_w * (noise_cond - noise_uncond)
    })
    
    beta <- extract(betas, t, img$shape)
    sqrt_one_minus_alpha_hat <- extract(torch_sqrt(1 - alphas_bar), t, img$shape)
    sqrt_inv_alpha <- extract(torch_sqrt(1 / alphas), t, img$shape)
    
    mean <- sqrt_inv_alpha * (img - (beta / sqrt_one_minus_alpha_hat) * pred_noise)
    
    if (i > 1) {
      noise <- torch_randn_like(img)
      sigma <- torch_sqrt(beta)
      img <- mean + sigma * noise
    } else {
      img <- mean
    }
  }
  (img$clamp(-1, 1) + 1) / 2
}

# ==========================================
# 7. 结果展示
# ==========================================
# 尝试生成 16 个数字 "7"
# guide_w = 0 是无条件(瞎画)，2.0 是强力引导，4.0 可能过拟合/失真
samples <- sample_conditional(model, n_samples = 16, target_digit = 7, guide_w = 2.0)

par(mfrow = c(2, 8), mar = c(0.1, 0.1, 0.1, 0.1))
samples_cpu <- samples$cpu()
for (i in 1:16) {
  img_mat <- as.matrix(samples_cpu[i, 1, , ])
  img_mat <- apply(img_mat, 2, rev)
  image(t(img_mat), col = gray.colors(256), axes = FALSE)
}

