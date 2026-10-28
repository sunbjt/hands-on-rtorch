library(torch)

# 设置随机种子
set.seed(123)
torch_manual_seed(123)

# --- 1. 数据生成函数 ---
get_real_data <- function(n) {
  # 真实数据：均值 2，标准差 1
  torch_randn(n, 1) * 1 + 2
}

# --- 2. 模型定义 ---
Generator <- nn_module(
  initialize = function() {
    self$net <- nn_sequential(
      nn_linear(5, 64),      
      nn_leaky_relu(0.2), 
      nn_linear(64, 1) 
    )
  },
  forward = function(x) { self$net(x) }
)

Discriminator <- nn_module(
  initialize = function() {
    self$net <- nn_sequential(
      nn_linear(1, 64),
      nn_leaky_relu(0.2),
      nn_linear(64, 1)
      # [注意] LSGAN 不使用 Sigmoid
    )
  },
  forward = function(x) { self$net(x) }
)

netG <- Generator()
netD <- Discriminator()

# RMSprop 或 Adam 均可，这里保留极小学习率
lr <- 0.00005 
optG <- optim_rmsprop(netG$parameters, lr = lr)
optD <- optim_rmsprop(netD$parameters, lr = lr)

# [核心] 使用 MSE Loss 代替 BCE
criterion <- nn_mse_loss()

# --- 3. 训练循环 ---
cat("Training 10,000 steps with LSGAN...\n")

batch_size <- 512
total_steps <- 10000 
capture_steps <- c(1, 1000, 2500, 5000, 7500, 10000)
snapshots <- list()

for (step in 1:total_steps) {
  
  real_data <- get_real_data(batch_size)
  fake_data <- netG(torch_rand(batch_size, 5))
  
  # === 训练 Discriminator ===
  # LSGAN D Loss: 最小化 D(real) 与 1 的 MSE，以及 D(fake) 与 0 的 MSE
  loss_d <- criterion(netD(real_data), torch_ones(batch_size, 1)) +
    criterion(netD(fake_data$detach()), torch_zeros(batch_size, 1))
  
  optD$zero_grad()
  loss_d$backward()
  optD$step()
  
  # === 训练 Generator ===
  # LSGAN G Loss: 最小化 D(fake) 与 1 的 MSE
  loss_g <- criterion(netD(fake_data), torch_ones(batch_size, 1))
  
  optG$zero_grad()
  loss_g$backward()
  optG$step()
  
  # --- 记录数据 ---
  if (step %in% capture_steps) {
    with_no_grad({
      snapshots[[as.character(step)]] <- as.numeric(netG(torch_rand(5000, 5)))
    })
    if(step %% 2000 == 0) cat(sprintf("Step %d completed...\n", step))
  }
}

# --- 4. 绘图 ---
par(mfrow = c(2, 3), mar = c(2, 1, 2, 0)) 
real_density <- density(as.numeric(get_real_data(5000))) 

for (step in capture_steps) {
  fake_vals <- snapshots[[as.character(step)]]
  plot(density(fake_vals), col = "red", lwd = 2, 
       main = paste("Step", step),
       xlim = c(-3, 7), ylim = c(0, 0.6), xlab = "", ylab = "")
  lines(real_density, col = "blue", lwd = 2, lty = 2)
  polygon(real_density, col = rgb(0, 0, 1, 0.1), border = NA)
}
par(mfrow = c(1, 1))