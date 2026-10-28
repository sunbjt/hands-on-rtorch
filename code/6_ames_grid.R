
## 第 4 章内容
library(tidymodels)
library(torch)
library(tidyverse)
library(showtext)
showtext_auto()

# --- 1. 数据准备 ---
data(ames)

set.seed(123)
torch_manual_seed(123)

ames_split <- initial_split(ames, prop = 0.80)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# --- 2. 基础 Recipe ---


# 修改 Recipe：使用更多变量
rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  # 0. 标准化 outcomes
  step_normalize(all_outcomes()) %>%
  # 1. 剔除无用的空间坐标/ID 等无预测力的列
  step_rm(matches("Id|Longitude|Latitude")) %>% 
  # 2. 处理偏态分布 (让数值更像正态分布)
  step_log(Gr_Liv_Area, Lot_Area, base = 10) %>% 
  # 3. 处理高基数类别 (比如 Neighborhood 有几十个，这就够 MLP 喝一壶的)
  step_other(all_nominal_predictors(), threshold = 0.01) %>% 
  # 4. 哑变量化
  step_dummy(all_nominal_predictors()) %>% 
  # 5. 零方差剔除 (剔除那些所有样本都一样的无用特征)
  step_nzv(all_predictors()) %>%
  # 6. 标准化 (MLP 的生命线)
  step_normalize(all_predictors()) %>%
  # 7. 缺失值填充
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

# --- 3. 转换为张量 (Matrix -> Tensor) ---
prep_rec <- prep(rec)
train_df <- bake(prep_rec, new_data = NULL)
test_df  <- bake(prep_rec, new_data = ames_test)

# 提取 x 和 y，转为 tensor
x_train <- torch_tensor(
  as.matrix(train_df %>% select(-Sale_Price)), dtype = torch_float())
y_train <- torch_tensor(
  matrix(train_df$Sale_Price, ncol=1), dtype = torch_float())

x_test <- torch_tensor(
  as.matrix(test_df %>% select(-Sale_Price)), dtype = torch_float())
y_test <- torch_tensor(
  matrix(test_df$Sale_Price, ncol=1), dtype = torch_float())



#  --- 第 6 章内容 ---

library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

# 简单封装为 Dataset (为了后续使用 Dataloader/Luz)
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

# 1. --- 利用原始的网络来看预测效果 ---
# 演示过拟合的状态

# 定义一个模型类（这是一张图纸，而不是实例）
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

AmesNet_fitted <- AmesNet %>%
  # 1. 建立 (Setup): 指定损失函数和优化器
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw, # AdamW 通常比 Adam 更稳健
    metrics = list(
      luz_metric_rmse() # 监控 RMSE
    )
  ) %>%
  set_hparams(input_dim = ncol(x_train)) %>%
  set_opt_hparams(lr = 0.002) %>% # 学习率与权重衰减
  # 3. 训练 (Fit)
  fit(
    train_dl,
    epochs = 50,
    valid_data = test_dl,
    verbose = TRUE # 显示进度条
  )

plot(AmesNet_fitted)

AmesNet_fitted |> evaluate(test_dl) |> get_metrics()


# 2. --- 设定一个可以传递参数的网络，方便我们修改参数 ---

FlexibleMLP <- nn_module(
  "FlexibleMLP",
  initialize = function(in_features, h1 = 128, dr = 0.1) {
    self$fc1 <- nn_linear(in_features, h1)
    self$bn1 <- nn_batch_norm1d(h1)
    # 第二层神经元数量设为第一层的一半，体现漏斗结构
    self$fc2 <- nn_linear(h1, floor(h1 / 2)) 
    self$bn2 <- nn_batch_norm1d(floor(h1 / 2))
    self$fc3 <- nn_linear(floor(h1 / 2), 1)
    
    self$act <- nn_relu()
    self$dropout <- nn_dropout(dr)
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% self$bn1() %>% self$act() %>% self$dropout() %>%
      self$fc2() %>% self$bn2() %>% self$act() %>% self$dropout() %>%
      self$fc3()
  }
)

# 获取输入特征的数量 (用于初始化模型第一层)
input_dim <- ncol(x_train)


# 正式训练模型
FlexibleMLP_fitted <- FlexibleMLP %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw, # AdamW 通常比 Adam 更稳健
    metrics = list(
      luz_metric_rmse() # 监控 RMSE
    )
  ) %>%
  set_hparams(in_features = input_dim, h1 = 128, dr = 0.3) %>%
  set_opt_hparams(lr = 0.002, weight_decay = 1e-5) %>% # 学习率与权重衰减
  fit(
    train_dl,
    epochs = 50,
    valid_data = test_dl,
    verbose = TRUE # 显示进度条
  )

plot(FlexibleMLP_fitted)

# 在测试集上评估
FlexibleMLP_fitted |> evaluate(test_dl) |> get_metrics()



# 3. --- Callbacks ---
# 演示调用逻辑

early_stop_cb <- luz_callback_early_stopping(
  monitor = "valid_loss", # 监控验证集损失
  patience = 5,           # 如果连续 5 轮 Loss 都没有改善
  min_delta = 0.001,      # 且改善幅度小于 0.001
  mode = "min"            # 对于 Loss，我们要越小越好
)

checkpoint_cb <- luz_callback_model_checkpoint(
  path = "checkpoints/ames_model_{epoch:02d}.pt", # 保存路径模板
  save_best_only = TRUE, # 只保存表现最好的模型
  monitor = "valid_loss",
  mode = "min"
)

lr_scheduler_cb <- luz_callback_lr_scheduler(
  lr_reduce_on_plateau, 
  patience = 3,           # 耐心比早停法稍短，先降学习率，实在不行再早停
  factor = 0.5,
  mode = "min", verbose = TRUE
)

final_fit <- FlexibleMLP %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_rmse())
  ) %>%
  set_hparams(in_features = input_dim, h1 = 128, dr = 0.3) %>%
  set_opt_hparams(lr = 0.002, weight_decay = 1e-5) %>%
  fit(
    data = train_dl,
    valid_data = test_dl,
    epochs = 100,
    callbacks = list(
      lr_scheduler_cb, # 1. 遇到瓶颈先降速
      early_stop_cb,   # 2. 降速依然无效则早停
      checkpoint_cb    # 3. 始终默默保存最佳状态
    )
  )

# 训练完成后，可视化训练过程
plot(final_fit)

final_fit |> evaluate(test_dl) |> get_metrics()


## ------------------------------------------------------ ##
## 超参数调优
## ------------------------------------------------------ ##

# 尝试找到最佳的 Learning Rate
trained_lr <- FlexibleMLP %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw
  ) %>%
  set_hparams(in_features = input_dim) %>%
  lr_finder(data = train_dl, verbose = FALSE)

# 绘制 Loss 与 Learning Rate 的关系曲线
plot(trained_lr)


# search 方法
fit_one_trial <- function(hparams, train_dl, valid_dl, e_num=20) {
  
  # 打印当前尝试的参数（可选）
  message(sprintf("Testing: Units=%d, Drop=%.2f, LR=%.4f, Decay=%.4f", 
                  hparams$units, hparams$drop, hparams$lr, hparams$decay))
  
  fitted <- FlexibleMLP %>%
    setup(
      loss = nn_mse_loss(),
      optimizer = optim_adamw,
      metrics = list(luz_metric_rmse())
    ) %>%
    set_hparams(
      in_features = input_dim,
      h1 = hparams$units,
      dr = hparams$drop
    ) %>%
    set_opt_hparams(
      lr = hparams$lr,
      weight_decay = hparams$decay
    ) %>%
    fit(
      train_dl,
      epochs = e_num, # 搜索阶段 Epoch 可以设少一点以节省时间
      valid_data = test_dl,
      verbose = FALSE, # 关闭进度条，避免刷屏
      callbacks = list(
        luz_callback_early_stopping(patience = 3) # 表现不好尽早掐断
      )
    )
  
  # 返回验证集上最好的 RMSE
  valid_rmse <- get_metrics(fitted) |> filter(
    metric == 'rmse' & set == 'valid'
  )
  best_metric <- min(valid_rmse$value)
  return(best_metric)
}


n_trials <- 24
search_grid <- tibble(
  trial_id = 1:n_trials,
  # 随机采样
  units = sample(c(64, 128, 256, 512), n_trials, replace = TRUE),
  drop  = runif(n_trials, min = 0.0, max = 0.5),
  lr    = runif(n_trials, min = 0.001, max = 0.006),
  decay = 10^runif(n_trials, min = -5, max = -3)
)

# 初始化一个列来存储结果
search_grid$val_rmse <- numeric(n_trials)
head(search_grid)


# 开始炼丹
# 注意：在实际项目中，这一步可能非常耗时
for (i in 1:nrow(search_grid)) {
  
  # 提取当前行的参数
  current_params <- list(
    units = search_grid$units[i],
    drop  = search_grid$drop[i],
    lr    = search_grid$lr[i],
    decay = search_grid$decay[i]
  )
  
  # 捕获可能的错误（防止某组参数导致报错中断循环）
  try({
    score <- fit_one_trial(current_params, train_dl, test_dl, e_num=20)
    search_grid$val_rmse[i] <- score
  })
  
  # 垃圾回收，释放显存
  gc() 
}

search_grid %>% 
  arrange(val_rmse) %>% 
  slice(1)

# 观察特征切在哪里
library(rpart)
library(rpart.plot)
fit_tree <- rpart(
  val_rmse ~ units + drop + lr + decay,
  data = search_grid,
  control = rpart.control(cp = 0.05, minsplit = 8)
)
rpart.plot(fit_tree, type = 2, extra = 101)


# 网格搜索设计
fine_grid <- expand.grid(
  # 既然 512 表现最好，我们集中火力在高端区，甚至尝试更大一点
  units = c(256, 512, 1024), 
  # 决策树显示 Decay 很重要，我们在最佳区间附近细搜
  decay = c(1e-4, 5e-4, 1e-3), 
  # Dropout 在决策树中不是主分裂点，但为了保险，保留两个常用值
  drop = c(0.2, 0.4),
  # 学习率已由 lr_finder 解决，不再浪费算力搜索
  lr = 0.003 
)

fine_grid$val_rmse <- numeric(nrow(fine_grid))
# 执行网格搜索
for (i in 1:nrow(fine_grid)) {
  
  # 提取当前行的参数
  current_params <- list(
    units = fine_grid$units[i],
    drop  = fine_grid$drop[i],
    lr    = fine_grid$lr[i],
    decay = fine_grid$decay[i]
  )
  
  # 捕获可能的错误（防止某组参数导致报错中断循环）
  try({
    score <- fit_one_trial(current_params, train_dl, test_dl, e_num=50)
    fine_grid$val_rmse[i] <- score
  })
  
  # 垃圾回收，释放显存
  gc() 
}


fine_grid %>% 
  arrange(val_rmse) %>% 
  slice(1)

fitted_ffnn <- FlexibleMLP %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_rmse())
  ) %>%
  set_hparams(
    in_features = input_dim,
    h1 = 256,
    dr = 0.2
  ) %>%
  set_opt_hparams(
    lr = 0.003,
    weight_decay = 5e-04
  ) %>%
  fit(
    train_dl,
    epochs = 50, # 最终测试
    valid_data = test_dl,
    verbose = FALSE, # 关闭进度条，避免刷屏
    callbacks = list(
      luz_callback_early_stopping(patience = 3) # 表现不好尽早掐断
    )
  )
metrics <- fitted_ffnn %>% evaluate(test_dl)
print(metrics)




library(mlr3mbo)
library(bbotk)
library(paradox)
library(mlr3learners)
library(rgenoud)

# ==========================================
# 封装评估引擎：接入你的 4 个核心参数
# ==========================================
torch_objective <- ObjectiveRFun$new(
  
  fun = function(xs) {
    # 1. 将 mbo 传过来的参数，打包成 fit_one_trial 的 hparams
    current_hparams <- list(
      units = as.integer(xs$units),
      drop  = xs$drop,
      lr    = xs$lr,
      decay = 10^xs$log_decay
    )
    
    # 2. 将整个列表作为单一参数传给底层训练逻辑
    final_val_loss <- fit_one_trial(
      current_hparams, 
      train_dl = train_dl,   # 传入训练集 DataLoader
      valid_dl = test_dl    # 传入验证集 DataLoader
    )
    
    # 3. 返回目标值
    return(list(val_loss = final_val_loss))
  },
  
  # 定义 4 维搜索空间
  domain = ps(
    units     = p_int(lower = 64, upper = 512),   
    drop      = p_dbl(lower = 0.0, upper = 0.5),  
    lr        = p_dbl(lower = 0.001, upper = 0.006), 
    log_decay = p_dbl(lower = -5, upper = -3)
  ),
  
  # 定义优化目标：最小化验证集误差
  codomain = ps(val_loss = p_dbl(tags = "minimize"))
)


# ==========================================
# 启动 4 维空间的贝叶斯优化
# ==========================================
terminator <- trm("evals", n_evals = 30)

instance <- OptimInstanceBatchSingleCrit$new(
  objective = torch_objective,
  terminator = terminator
)

# 执行优化
set.seed(42)
opt("mbo")$optimize(instance)

# 查看并还原最优参数
best <- instance$result_x_domain

cat(sprintf("
- Units (隐藏节点数) : %d
- Drop  (失活比例)   : %.4f
- LR    (学习率)     : %.4f
- Decay (权重衰减)   : %.4f
    最小验证集 Loss  : %.4f
", 
best$units, best$drop, best$lr, 
10^best$log_decay, instance$result_y))


# ----------------------------------------------------- #
# 对比 xgboost 作为基线，RMSE 是多少？
# ----------------------------------------------------- #


library(xgboost)
train_x <- as.matrix(train_df %>% select(-Sale_Price))
train_y <- train_df$Sale_Price
test_x <- as.matrix(test_df %>% select(-Sale_Price))
test_y <- test_df$Sale_Price

# 构建 DMatrix 对象
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = test_x, label = test_y)

# 模型参数与训练
params <- list(
  booster          = "gbtree",             # 使用树模型
  objective        = "reg:squarederror",   # 回归任务：均方误差
  eta              = 0.1,                  # 学习率 (步长)
  max_depth        = 6,                    # 树的最大深度
  subsample        = 0.8,                  # 随机采样比例 (防止过拟合)
  colsample_bytree = 0.8                   # 特征采样比例
)

# 设定监控列表 (Watchlist)
watchlist <- list(train = dtrain, eval = dtest)
# 训练模型
bst_model <- xgb.train(
  params          = params,
  data            = dtrain,
  nrounds         = 1000,                  # 最大迭代轮数
  evals           = watchlist,
  print_every_n   = 100,                   # 每100轮打印一次日志
  early_stopping_rounds = 50,              # 如果50轮内测试集误差没下降，就停止 (防止过拟合)
  verbose         = 1
)

# 预测与评估

pred_y <- predict(bst_model, dtest)
# 计算 RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((test_y - pred_y)^2))

# 计算 R-squared
rss <- sum((pred_y - test_y) ^ 2)  # Residual Sum of Squares
tss <- sum((test_y - mean(test_y)) ^ 2)  # Total Sum of Squares
r2 <- 1 - (rss / tss)

# 评估结果 (标准化尺度) 
cat(paste("最佳迭代轮数:", bst_model$best_iteration, "\n"))
cat(paste("测试集 RMSE :", round(rmse, 4), "\n"))
cat(paste("测试集 R2   :", round(r2, 4), "\n"))

