library(tidymodels)
library(torch)
library(tidyverse)
library(showtext)
showtext_auto()
data(ames)

set.seed(123)
torch_manual_seed(123)

ames_split <- initial_split(ames, prop = 0.80)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_normalize(all_outcomes()) %>%
  step_rm(matches("Id|Longitude|Latitude")) %>%
  step_log(Gr_Liv_Area, Lot_Area, base = 10) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

prep_rec <- prep(rec)
train_df <- bake(prep_rec, new_data = NULL)
test_df <- bake(prep_rec, new_data = ames_test)

# 提取 x 和 y，转为 tensor
x_train <- torch_tensor(
  as.matrix(train_df %>% select(-Sale_Price)), dtype = torch_float())
y_train <- torch_tensor(
  matrix(train_df$Sale_Price, ncol=1), dtype = torch_float())

x_test <- torch_tensor(
  as.matrix(test_df %>% select(-Sale_Price)), dtype = torch_float())
y_test <- torch_tensor(
  matrix(test_df$Sale_Price, ncol=1), dtype = torch_float())

# --- 4. 定义模型 ---
model <- nn_sequential(
  nn_linear(ncol(x_train), 128), # 输入特征 -> 128个神经元
  nn_relu(),                     # 激活函数
  nn_linear(128, 64),            # 128 -> 64
  nn_relu(),                     # 激活函数
  nn_linear(64, 1)               # 64 -> 输出1个预测值
)

learning_rate <- 0.002
optimizer <- optim_rprop(model$parameters, lr = learning_rate)
loss_fn <- nn_mse_loss()

# --- 训练循环 ---

epochs <- 1000
# 用于记录 loss 变化以便画图
train_losses <- numeric(epochs)
for (t in 1:epochs) {
  
  # 前向传播
  y_pred <- model(x_train)
  loss <- loss_fn(y_pred, y_train)
  
  # 梯度求解三部曲
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
  
  train_losses[t] <- loss$item() # 用于记录与打印
  
  if (t %% 100 == 0) {
    cat(sprintf("Epoch %d: Loss = %.4f\n", t, loss$item()))
  }
}

plot(train_losses, type = "l", col = "blue", lwd = 2,
     main = "训练 Loss 下降曲线", xlab = "Epoch", ylab = "MSE Loss")

# --- 评估 (Evaluation) ---
# 切换到评估模式 (虽然本例没用 Dropout，但养成好习惯)
model$eval()

# 提取预测值 (Tensor -> R Vector)
with_no_grad({
  # 注意：y_pred 是 [N, 1] 的矩阵，as.numeric 会把它拉平为向量
  preds_nn <- as.numeric(model(x_test))
})

# 计算 RMSE
truth <- as.numeric(y_test) # 真实标签
rmse_nn <- sqrt(mean((truth - preds_nn)^2))
cat(sprintf("Neural Network RMSE: %.4f\n", rmse_nn))

# 测试 lm 使用与神经网络完全相同的数据集
fit_lm <- lm(Sale_Price ~ ., data = train_df)
preds_lm <- predict(fit_lm, newdata = test_df)

# 计算 RMSE (均方根误差)
rmse_lm <- sqrt(mean((test_df$Sale_Price - preds_lm)^2))
cat(sprintf("Linear Regression RMSE: %.4f\n", rmse_lm))

# 整理数据用于绘图
results <- tibble(
  Truth = truth,
  Linear_Model = preds_lm,
  Neural_Network = preds_nn
) %>% 
  pivot_longer(cols = c(Linear_Model, Neural_Network), 
               names_to = "Model", values_to = "Prediction")

library(ggplot2)
ggplot(results, aes(x = Truth, y = Prediction, color = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 0.5, size = 2) +
  scale_color_manual(values = c("Linear_Model" = "black", "Neural_Network" = "red")) +
  labs(
    title = "线性回归 vs 神经网络 (Ames 房价预测)",
    x = "真实房价",
    y = "预测房价"
  ) +
  theme_minimal() +
  coord_obs_pred() # 保证 x 和 y 轴比例一致


# ==========================================
# 参数与优化器初始化 (Lasso 设置)
# ==========================================
n_in <- ncol(x_train)
feature_names <- colnames(train_df %>% select(-Sale_Price))

w <- (torch_randn(n_in, 1) * 0.1)$requires_grad_(TRUE)
b <- torch_zeros(1, requires_grad = TRUE)

lambda <- 0.5
alpha <- 0.95
learning_rate <- 0.002
optimizer <- optim_adam(list(w, b), lr = learning_rate)

# 5. 带有软阈值的训练循环
epochs <- 1000
train_losses <- numeric(epochs)
w_history <- matrix(NA, nrow = epochs, ncol = n_in)
colnames(w_history) <- feature_names

for (t in 1:epochs) {
  optimizer$zero_grad()
  
  y_pred <- torch_mm(x_train, w) + b # 向前传播
  
  mse_loss <- loss_fn(y_pred, y_train)
  l2_penalty <- torch_square(w)$sum()
  loss <- mse_loss + lambda * (1 - alpha) * 0.5 * l2_penalty # 计算 loss
  
  # 梯度反向传播并走一步
  loss$backward()
  optimizer$step()
  
  # 近端梯度下降 (Proximal Step) 截断 L1 惩罚
  with_no_grad({
    # 计算截断阈值 (学习率 * 惩罚力度 * L1比例)
    tau <- learning_rate * lambda * alpha
    
    # 应用软阈值公式进行截断，强行抹零微小权重
    w_new <- torch_sign(w) * torch_relu(torch_abs(w) - tau)
    w$copy_(w_new) # 就地更新权重
  })
  
  # E. 记录指标
  train_losses[t] <- loss$item()
  w_history[t, ] <- as.numeric(w$detach())
  
  if (t %% 100 == 0) cat(sprintf("Epoch %d: Loss = %.4f\n", t, loss$item()))
}

# ==========================================
# 6. 优化的 Coefficient Path 绘图
# ==========================================
# 获取最终 epoch 的绝对权重，找出 Top 5
final_w <- abs(w_history[epochs, ])
top_5_idx <- order(final_w, decreasing = TRUE)[1:5]
top_5_features <- feature_names[top_5_idx]

# 转换为 ggplot 格式
w_df <- as_tibble(w_history) %>%
  mutate(Epoch = 1:epochs) %>%
  pivot_longer(cols = -Epoch, names_to = "Feature", values_to = "Weight") %>%
  mutate(Highlight = if_else(Feature %in% top_5_features, Feature, "Other"))

# 绘制路径图
ggplot(w_df, aes(x = Epoch, y = Weight, group = Feature)) +
  # 添加一条显眼的零位虚线
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.8
  ) +
  geom_line(aes(
    color = Highlight,
    alpha = Highlight == "Other",
    linewidth = Highlight == "Other"
  )) +
  scale_color_manual(values = c(
    setNames(RColorBrewer::brewer.pal(5, "Set1"), top_5_features),
    "Other" = "grey70"
  )) +
  scale_alpha_manual(values = c("TRUE" = 0.4, "FALSE" = 1), guide = "none") +
  scale_linewidth_manual(values = c("TRUE" = 0.5, "FALSE" = 1.2), guide = "none") +
  theme_minimal(base_size = 14) +
  labs(x = "Epoch", y = "Coefficient Weight", color = "Top 5") +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

# ==========================================
# 7. 评估与零权重统计
# ==========================================
with_no_grad({
  preds_nn <- torch_mm(x_test, w) + b
  rmse_nn <- as.numeric(torch_sqrt(torch_mean(torch_square(preds_nn - y_test))))
})

cat(sprintf("\n=== 模型总结 ===\n"))
cat(sprintf("Elastic Net RMSE: %.4f\n", rmse_nn))
cat(sprintf("初始特征总数: %d\n", n_in))
cat(sprintf("被完全清零的特征数: %d\n", sum(final_w == 0)))
cat(sprintf("最终保留的特征数: %d\n", sum(final_w != 0)))


#####
library(embed)
rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_rm(matches("Id|Longitude|Latitude")) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_log(Gr_Liv_Area, Lot_Area, base = 10) %>%
  step_mutate(across(c(where(is.numeric), -Sale_Price), ~ ., .names = "{.col}_bin")) %>%
  step_discretize_cart(
    ends_with("_bin"), 
    outcome = "Sale_Price", 
    tree_depth = 3,             
    cost_complexity = 0.001      
  ) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_outcomes()) %>%
  step_normalize(all_predictors())

# 训练并应用
prep_rec <- prep(rec)
train_df <- bake(prep_rec, new_data = NULL)
test_df <- bake(prep_rec, new_data = ames_test)

# 提取 x 和 y，转为 tensor
x_train <- torch_tensor(
  as.matrix(train_df %>% select(-Sale_Price)), dtype = torch_float())
y_train <- torch_tensor(
  matrix(train_df$Sale_Price, ncol=1), dtype = torch_float())

x_test <- torch_tensor(
  as.matrix(test_df %>% select(-Sale_Price)), dtype = torch_float())
y_test <- torch_tensor(
  matrix(test_df$Sale_Price, ncol=1), dtype = torch_float())

# --- 4. 定义模型 ---
model <- nn_sequential(
  nn_linear(ncol(x_train), 128), # 输入特征 -> 128个神经元
  nn_relu(),                     # 激活函数
  nn_linear(128, 64),            # 128 -> 64
  nn_relu(),                     # 激活函数
  nn_linear(64, 1)               # 64 -> 输出1个预测值
)

epochs <- 1000
train_losses <- numeric(epochs)
w_history <- matrix(NA, nrow = epochs, ncol = n_in)
colnames(w_history) <- feature_names

for (t in 1:epochs) {
  optimizer$zero_grad()
  
  y_pred <- torch_mm(x_train, w) + b # 向前传播
  
  mse_loss <- loss_fn(y_pred, y_train)
  l2_penalty <- torch_square(w)$sum()
  loss <- mse_loss + lambda * (1 - alpha) * 0.5 * l2_penalty # 计算 loss
  
  # 梯度反向传播并走一步
  loss$backward()
  optimizer$step()
  
  # 近端梯度下降 (Proximal Step) 截断 L1 惩罚
  with_no_grad({
    # 计算截断阈值 (学习率 * 惩罚力度 * L1比例)
    tau <- learning_rate * lambda * alpha
    
    # 应用软阈值公式进行截断，强行抹零微小权重
    w_new <- torch_sign(w) * torch_relu(torch_abs(w) - tau)
    w$copy_(w_new) # 就地更新权重
  })
  
  # E. 记录指标
  train_losses[t] <- loss$item()
  w_history[t, ] <- as.numeric(w$detach())
  
  if (t %% 100 == 0) cat(sprintf("Epoch %d: Loss = %.4f\n", t, loss$item()))
}

with_no_grad({
  preds_nn <- torch_mm(x_test, w) + b
  rmse_nn <- as.numeric(torch_sqrt(torch_mean(torch_square(preds_nn - y_test))))
})

cat(sprintf("\n=== 模型总结 ===\n"))
cat(sprintf("Elastic Net RMSE: %.4f\n", rmse_nn))
cat(sprintf("初始特征总数: %d\n", n_in))
cat(sprintf("被完全清零的特征数: %d\n", sum(final_w == 0)))
cat(sprintf("最终保留的特征数: %d\n", sum(final_w != 0)))


