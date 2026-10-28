library(torch)

set.seed(123)
torch_manual_seed(123)

get_real_data <- function(n) { torch_randn(n, 1) * 1 + 2 }

# --- 1. 模型定义 ---
Generator <- nn_module(
  initialize = function() {
    self$net <- nn_sequential(nn_linear(5, 64), nn_leaky_relu(0.2), nn_linear(64, 1))
  },
  forward = function(x) { self$net(x) }
)

Critic <- nn_module(
  initialize = function() {
    self$net <- nn_sequential(nn_linear(1, 64), nn_leaky_relu(0.2), nn_linear(64, 1))
  },
  forward = function(x) { self$net(x) }
)

netG <- Generator()
netC <- Critic()

batch_size <- 512

# [注意] WGAN-GP 论文推荐使用 Adam (betas=0, 0.9)，但 RMSprop 也可以跑
lr <- 0.0005 
optG <- optim_rmsprop(netG$parameters, lr = lr)
optC <- optim_rmsprop(netC$parameters, lr = lr)

# 梯度惩罚计算函数
calc_gradient_penalty <- function(netC, real_data, fake_data) {
  # 1. 随机插值
  alpha_dims <- c(batch_size, rep(1, length(real_data$shape) - 1))
  alpha <- torch_rand(alpha_dims, device = real_data$device)
  interpolates <- (alpha * real_data + (1 - alpha) * fake_data)$requires_grad_(TRUE)
  
  # 2. 判别插值结果
  d_interpolates <- netC(interpolates)
  fake <- torch_ones_like(d_interpolates)
  
  # 3. 计算对插值输入的梯度
  gradients <- autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE, 
    retain_graph = TRUE
  )[[1]]
  
  # 4. 计算 L2 范数惩罚项 (||grad||_2 - 1)^2
  gradients <- gradients$view(c(batch_size, -1))
  gradients_norm <- gradients$norm(2, dim = 2)
  gradient_penalty <- ((gradients_norm - 1)^2)$mean()
  
  return(gradient_penalty)
}

# --- 2. 训练循环 ---
total_steps <- 10000
n_critic <- 5
capture_steps <- c(1, 1000, 2500, 5000, 7500, 10000)

snapshots_data <- list()
snapshots_wdist <- list()

cat("Start Training Loop (WGAN-GP)...\n")

for (step in 1:total_steps) {
  
  # === 训练 Critic ===
  for (i in 1:n_critic) {
    optC$zero_grad()
    
    real_data <- get_real_data(batch_size)
    fake_data <- netG(torch_rand(batch_size, 5))$detach()
    
    d_loss_original <- torch_mean(netC(fake_data)) - torch_mean(netC(real_data)) 
    
    # 引入梯度惩罚项
    lambda_gp <- 0.1 # 论文默认推荐值通常是 10
    gradient_penalty <- calc_gradient_penalty(netC, real_data, fake_data)
    
    d_loss <- d_loss_original + (lambda_gp * gradient_penalty)
    d_loss$backward()
    optC$step()
  }
  
  # === 训练 Generator ===
  optG$zero_grad()
  fake_data <- netG(torch_rand(batch_size, 5))
  loss_g <- -torch_mean(netC(fake_data))
  loss_g$backward()
  optG$step()
  
  # --- 记录与打印 ---
  if (step %in% capture_steps) {
    with_no_grad({
      fake_vals <- netG(torch_rand(5000, 5))
      real_vals <- get_real_data(5000)
      
      snapshots_data[[as.character(step)]] <- as.numeric(fake_vals)
      w_dist_est <- torch_mean(netC(real_vals)) - torch_mean(netC(fake_vals))
      snapshots_wdist[[as.character(step)]] <- w_dist_est$item()
    })
    cat(sprintf("Step %d | C Loss: %.4f | Est W-Dist: %.4f\n", 
                step, d_loss$item(), snapshots_wdist[[as.character(step)]]))
  }
}

# --- 3. 绘图 ---
par(mfrow = c(2, 3), mar = c(2, 1, 2, 0)) 
real_density <- density(as.numeric(get_real_data(5000))) 

for (step in capture_steps) {
  fake_vals <- snapshots_data[[as.character(step)]]
  w_dist_val <- snapshots_wdist[[as.character(step)]]
  
  plot(density(fake_vals), col = "red", lwd = 2, 
       main = sprintf("Step %d (W-Dist: %.4f)", step, w_dist_val),
       xlim = c(-5, 8), ylim = c(0, 0.6), xlab = "", ylab = "")
  lines(real_density, col = "blue", lwd = 2, lty = 2)
  polygon(real_density, col = rgb(0, 0, 1, 0.1), border = NA)
}
par(mfrow = c(1, 1))