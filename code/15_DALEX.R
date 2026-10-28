library(tidymodels)
library(torch)
library(tidyverse)
library(showtext)
library(luz)
library(DALEX)
library(ggplot2)

options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条
showtext_auto()

# =========================================================
# 1. 数据准备 & 2. 基础 Recipe
# =========================================================
data(ames)

set.seed(123)
torch_manual_seed(123)

ames_split <- initial_split(ames, prop = 0.80)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# 使用 Recipe 预处理数据
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

# 转换为张量 (Matrix -> Tensor)
prep_rec <- prep(rec)
train_df <- bake(prep_rec, new_data = NULL)
test_df  <- bake(prep_rec, new_data = ames_test)

# 提取 x 和 y，转为 tensor
x_train <- torch_tensor(
  as.matrix(train_df %>% select(-Sale_Price)), dtype = torch_float())
y_train <- torch_tensor(
  matrix(train_df$Sale_Price, ncol = 1), dtype = torch_float())

x_test <- torch_tensor(
  as.matrix(test_df %>% select(-Sale_Price)), dtype = torch_float())
y_test <- torch_tensor(
  matrix(test_df$Sale_Price, ncol = 1), dtype = torch_float())

# =========================================================
# 3. 使用 luz 定义并训练模型
# =========================================================

# 封装为 Dataset
ames_dataset <- dataset(
  name = "ames_ds",
  initialize = function(x, y) {
    self$x <- torch_tensor(x, dtype = torch_float32())
    self$y <- torch_tensor(y, dtype = torch_float32())
  },
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i, ])
  },
  .length = function() {
    self$x$size(1)
  }
)

train_ds <- ames_dataset(x_train, y_train)
test_ds  <- ames_dataset(x_test, y_test)

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
test_dl  <- dataloader(test_ds, batch_size = 128)

# 定义网络
AmesNet <- nn_module(
  "AmesNet",
  initialize = function(input_dim) {
    self$net <- nn_sequential(
      nn_linear(input_dim, 128),
      nn_relu(),
      nn_linear(128, 64),
      nn_relu(),
      nn_linear(64, 1)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

# 建立与训练
AmesNet_fitted <- AmesNet %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_rmse())
  ) %>%
  set_hparams(input_dim = ncol(x_train)) %>%
  set_opt_hparams(lr = 0.002) %>%
  fit(
    train_dl,
    epochs = 50,
    valid_data = test_dl,
    verbose = TRUE
  )

# =========================================================
# 4. 接入 DALEX 进行模型解释
# =========================================================

# --- A. 准备解释器需要的数据 ---
# 剔除 Target 列，保留预测用到的纯特征
df_test_x <- test_df %>% select(-Sale_Price) %>% as.data.frame()
y_test_vec <- test_df$Sale_Price

# --- B. 定义自定义预测函数 ---
custom_predict <- function(model, newdata) {
  if (is.data.frame(newdata)) {
    x_mat <- as.matrix(newdata)
  } else {
    x_mat <- newdata
  }
  x_t <- torch_tensor(x_mat, dtype = torch_float())
  # luz model 的 predict 会直接执行 forward 拿到张量
  pred_t <- predict(model, x_t)
  return(as.numeric(pred_t))
}

# --- C. 创建解释器 ---
explainer <- DALEX::explain(
  model = AmesNet_fitted,
  data = df_test_x,
  y = y_test_vec,
  predict_function = custom_predict,
  label = "Ames MLP",
  type = "regression",
  verbose = FALSE
)

# --- D. 绘图验证  ---

# 1. 全局特征重要性
vi <- model_parts(explainer, B = 50) # 打乱 50 次
# 使用 max_vars = 15 截断，只显示最重要的 15 个特征
plot(vi, max_vars = 15) + 
  labs(
    title = "Ames 基于排列重要性的特征 (Top 15)", 
    subtitle = ""
  ) +
  theme(
    # 调整主标题
    plot.title = element_text(size = 40, face = "bold"),
    # 调整副标题
    plot.subtitle = element_text(size = 35, color = "grey30"),
    # 调整 X 和 Y 轴的标题 (label)
    axis.title.x = element_text(size = 28, face = "bold"),
    axis.title.y = element_text(size = 28, face = "bold"),
    # 调整 X 和 Y 轴上的刻度文字 (变量名和数值)
    axis.text.x = element_text(size = 28),
    axis.text.y = element_text(size = 28, color = "black")
  )

# 2. 局部依赖 (以 Year_Built 为例)
ale_year <- model_profile(explainer, variables = "Year_Built", type = "accumulated")
plot(ale_year) +
  labs(
    title = "ALE: Year_Built vs Normalized Sale_Price", 
    subtitle = "展示神经网络捕捉到的非线性趋势",
    x = "Year_Built (Standardized)",
    y = "Average Prediction (Z-Score)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 14)
  )


# 3. 单样本归因 (Shapley & Break Down)
new_house <- df_test_x[2, , drop = FALSE]
bd <- predict_parts(explainer, new_observation = new_house, type = "break_down")
plot(bd) + ggtitle("单样本归因 (Break Down - DALEX内置)")

sv <- predict_parts(explainer, new_observation = new_house, type = "shap")
plot(sv) + ggtitle("Shapley Value (SHAP)", subtitle = "平均了多种顺序后的贡献值")


# =========================================================
# 5. 手写实现 Break-down (Greedy Strategy)
# =========================================================

predict_wrapper <- function(model, data_df) {
  x_mat <- as.matrix(data_df)
  x_t <- torch_tensor(x_mat, dtype = torch_float())
  preds <- predict(model, x_t)
  return(as.numeric(preds))
}

manual_break_down <- function(model, new_observation, background_data, predict_fun) {
  features <- colnames(background_data)
  n_features <- length(features)
  
  fixed_features <- c()
  
  # Step 0: Baseline
  baseline_preds <- predict_fun(model, background_data)
  current_average_prediction <- mean(baseline_preds)
  
  results <- data.frame(
    step = 0,
    variable = "Intercept",
    contribution = current_average_prediction,
    cumulative = current_average_prediction,
    variable_value = ""
  )
  
  cat("Step 0: Intercept (Baseline) =", round(current_average_prediction, 4), "\n")
  
  for (i in 1:n_features) {
    remaining_features <- setdiff(features, fixed_features)
    
    best_feature <- NULL
    best_contribution <- 0
    best_new_average <- 0
    max_abs_diff <- -1
    
    # Greedy Search
    for (feature in remaining_features) {
      temp_data <- background_data
      
      if (length(fixed_features) > 0) {
        for (f in fixed_features) {
          temp_data[[f]] <- new_observation[[f]]
        }
      }
      
      temp_data[[feature]] <- new_observation[[feature]]
      preds <- predict_fun(model, temp_data)
      new_average <- mean(preds)
      diff <- abs(new_average - current_average_prediction)
      
      if (diff > max_abs_diff) {
        max_abs_diff <- diff
        best_feature <- feature
        best_new_average <- new_average
      }
    }
    
    fixed_features <- c(fixed_features, best_feature)
    contribution <- best_new_average - current_average_prediction
    
    # 减少冗长打印：只打印贡献绝对值 > 0.01 的特征
    if (abs(contribution) > 0.01) {
      cat(sprintf("Step %2d: 固定变量 %-25s | 均值变动: %+.4f\n", 
                  i, best_feature, contribution))
    }
    
    current_average_prediction <- best_new_average
    
    results <- rbind(
      results,
      data.frame(
        step = i,
        variable = best_feature,
        contribution = contribution,
        cumulative = current_average_prediction,
        variable_value = as.character(round(new_observation[[best_feature]], 2))
      )
    )
  }
  
  return(results)
}

# =========================================================
# 6. 运行并在第 2 个样本上测试 & 画瀑布图
# =========================================================

sample_idx <- 2
obs_1 <- df_test_x[sample_idx, , drop = FALSE]

cat("\n=== 开始计算样本 #2 的 Break-down (仅打印贡献显著的步骤) ===\n")
bd_results <- manual_break_down(
  model = AmesNet_fitted,
  new_observation = obs_1,
  background_data = df_test_x,
  predict_fun = predict_wrapper
)

# 数据预处理：由于 Ames 哑变量化后特征极多（几十个），全部画图会非常拥挤！
# 策略：我们把贡献绝对值极小（< 0.02）的特征合并为 "Other Variables"

plot_data <- bd_results %>%
  mutate(
    is_major = abs(contribution) >= 0.02 | variable == "Intercept",
    display_var = ifelse(is_major, variable, "Other Variables")
  ) %>%
  group_by(display_var) %>%
  summarise(
    contribution = sum(contribution),
    # 取分组内最靠前的那一行的 step 作为排序依据
    min_step = min(step),
    variable_value = ifelse(n() == 1, first(variable_value), "...")
  ) %>%
  ungroup() %>%
  arrange(min_step) %>%
  mutate(
    cumulative = cumsum(contribution),
    ymin = lag(cumulative, default = 0),
    ymax = cumulative,
    type = case_when(
      display_var == "Intercept" ~ "Intercept",
      contribution > 0 ~ "Positive",
      TRUE ~ "Negative"
    ),
    label = ifelse(
      display_var %in% c("Intercept", "Other Variables"), 
      display_var, 
      paste0(display_var, " = ", variable_value)
    ),
    id = row_number()
  )

# 修正 Intercept 的 ymin 逻辑
plot_data$ymin[1] <- 0

# 绘图
ggplot(plot_data, aes(
  x = reorder(label, -id),
  ymin = ymin,
  ymax = ymax,
  fill = type
)) +
  geom_rect(aes(xmin = id - 0.4, xmax = id + 0.4)) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Intercept" = "#377eb8",
    "Positive" = "#4daf4a",
    "Negative" = "#e41a1c"
  )) +
  geom_text(aes(
    y = pmax(ymin, ymax),
    label = round(contribution, 3),
    x = id
  ),
  hjust = -0.2,
  size = 3.5) +
  labs(
    title = "Manual Break-down Plot (Ames Sample #2)",
    subtitle = "微小贡献特征已折叠为 Other Variables (目标：归一化房价)",
    y = "Prediction Value (Z-Score)",
    x = "Variable Attribution"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
