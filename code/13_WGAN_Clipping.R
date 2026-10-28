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

# [核心] WGAN 必须使用无动量的优化器，如 RMSprop
lr <- 0.00005 
optG <- optim_rmsprop(netG$parameters, lr = lr)
optC <- optim_rmsprop(netC$parameters, lr = lr)

clip_value <- 0.01 
batch_size <- 512
total_steps <- 10000
n_critic <- 5        
capture_steps <- c(1, 1000, 2500, 5000, 7500, 10000)

snapshots_data <- list()
snapshots_wdist <- list()

# --- 2. 训练循环 ---
cat("Start Training Loop (WGAN-Clipping)...\n")

for (step in 1:total_steps) {
  
  # === 训练 Critic ===
  for (i in 1:n_critic) {
    optC$zero_grad()
    
    real_data <- get_real_data(batch_size)
    fake_data <- netG(torch_rand(batch_size, 5))$detach()
    
    # Loss = D(fake) - D(real) (等价于最小化 - (D(real) - D(fake)))
    d_loss <- torch_mean(netC(fake_data)) - torch_mean(netC(real_data))
    d_loss$backward()
    optC$step()
    
    # Weight Clipping：使用 with_no_grad 更加规范安全
    with_no_grad({
      for (p in netC$parameters) {
        p$clamp_(-clip_value, clip_value)
      }
    })
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
      test_noise <- torch_rand(5000, 5)
      fake_vals <- netG(test_noise)
      real_vals <- get_real_data(5000)
      
      # 真实的 Wasserstein 估计值公式：E[C(real)] - E[C(fake)]
      estimated_w_dist <- torch_mean(netC(real_vals)) - torch_mean(netC(fake_vals))
      
      snapshots_data[[as.character(step)]] <- as.numeric(fake_vals)
      snapshots_wdist[[as.character(step)]] <- estimated_w_dist$item()
    })
    cat(sprintf("Step %d | W-Dist Est: %.4f\n", step, snapshots_wdist[[as.character(step)]]))
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