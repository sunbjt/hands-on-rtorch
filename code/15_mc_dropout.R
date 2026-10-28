library(torch)
library(ggplot2)

# 1. 构造带有噪声的非线性训练数据 (范围在 -4 到 4 之间)
set.seed(42)
torch_manual_seed(42)
x_train <- torch_tensor(seq(-4, 4, length.out = 150))$unsqueeze(2)
y_train <- x_train * torch_sin(x_train) + torch_randn(150, 1) * 0.3

# 2. 定义带 Dropout 的多层感知机
MCDropoutNet <- nn_module(
  "MCDropoutNet",
  initialize = function() {
    self$fc1 <- nn_linear(1, 64)
    self$fc2 <- nn_linear(64, 64)
    self$fc3 <- nn_linear(64, 1)
    # 关键：引入 Dropout 层
    self$drop <- nn_dropout(p = 0.2) 
  },
  forward = function(x) {
    x <- nnf_relu(self$fc1(x))
    x <- self$drop(x)
    x <- nnf_relu(self$fc2(x))
    x <- self$drop(x)
    self$fc3(x)
  }
)

net <- MCDropoutNet()
optimizer <- optim_adam(net$parameters, lr = 0.01, weight_decay = 1e-4)

# 3. 基础训练循环
for (epoch in 1:1000) {
  optimizer$zero_grad()
  loss <- nnf_mse_loss(net(x_train), y_train)
  loss$backward()
  optimizer$step()
}

# 4. MC Dropout 推理过程
# 构造范围更广的测试数据 (扩展到 -7 到 7)
x_test <- torch_tensor(seq(-7, 7, length.out = 300))$unsqueeze(2)

# 关键工程技巧：通常这里会调用 net$eval()
# 但为了进行 MC Dropout，我们保持网络在 train() 模式，让 Dropout 持续生效
net$train() 

n_simulations <- 100
predictions <- matrix(0, nrow = 300, ncol = n_simulations)

# 强制禁用梯度计算以节省显存，进行多次蒙特卡洛采样
with_no_grad({
  for (i in 1:n_simulations) {
    predictions[, i] <- as.numeric(net(x_test))
  }
})

# 5. 计算预测均值与不确定性 (标准差)
pred_mean <- apply(predictions, 1, mean)
pred_sd <- apply(predictions, 1, sd)

# 绘图可视化
test_df <- data.frame(
  x = as.numeric(x_test),
  y_mean = pred_mean,
  y_lower = pred_mean - 2 * pred_sd, # 95% 置信区间近似
  y_upper = pred_mean + 2 * pred_sd
)
train_df <- data.frame(x = as.numeric(x_train), y = as.numeric(y_train))

ggplot() +
  geom_ribbon(data = test_df, aes(x = x, ymin = y_lower, ymax = y_upper), 
              fill = "steelblue", alpha = 0.3) +
  geom_line(data = test_df, aes(x = x, y = y_mean), color = "blue", linewidth = 1) +
  geom_point(data = train_df, aes(x = x, y = y), color = "red", alpha = 0.5) +
  geom_vline(xintercept = c(-4, 4), linetype = "dashed", color = "black") +
  theme_minimal()






