library(torch)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条
library(tidyverse)
library(lubridate)
library(tidymodels) # 引入 tidymodels 生态
library(showtext)
showtext_auto()

torch_manual_seed(123)
set.seed(123)

# --- 1. 数据加载 ---
url <- "~/data/PRSA_data_2010.1.1-2014.12.31.csv"

# 基础清洗与特征提取
df_clean <- read.csv(url) %>%
  mutate(time = ymd_h(paste(year, month, day, hour))) %>%
  arrange(time) %>%
  mutate(
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  ) %>%
  # 仅保留后续需要用的列（加上 time 备用）
  select(time, TEMP, PRES, DEWP, hour_sin, hour_cos, month_sin, month_cos)

# ==========================================
# 2. 数据集划分与标准化 (Tidymodels 工业级流水线)
# ==========================================
seq_len <- 72

# 专门用于时序数据，严格按时间顺序截断 (前80%训练，后20%验证)
time_split <- initial_time_split(df_clean, prop = 0.8)
train_raw <- training(time_split)
valid_raw <- testing(time_split)

ts_recipe <- recipe(~ ., data = train_raw) %>%
  update_role(time, new_role = "ID") %>% 
  step_normalize(TEMP, PRES, DEWP) 

ts_prep <- prep(ts_recipe, training = train_raw)

train_data_norm <- bake(ts_prep, new_data = NULL, -time) %>% as.matrix()
valid_data_norm <- bake(ts_prep, new_data = valid_raw, -time) %>% as.matrix()

# TIDYMODELS 把计算好的统计量极其规范地存在了 prep 对象里
temp_mean <- ts_prep$steps[[1]]$means["TEMP"]
temp_sd   <- ts_prep$steps[[1]]$sds["TEMP"]

# --- 4. 定义 Dataset (支持多变量) ---

time_series_dataset <- dataset(
  name = "ts_dataset",
  
  initialize = function(data, seq_len) {
    self$data <- torch_tensor(data, dtype = torch_float())
    self$seq_len <- seq_len
  },
  
  .getitem = function(i) {
    x <- self$data[i:(i + self$seq_len - 1), ]
    
    # 这会返回一个 (1) 的张量，DataLoader 拼接后变成 (64, 1)
    y <- self$data[i + self$seq_len, 1]$unsqueeze(1)
    
    list(x = x, y = y)
  },
  
  .length = function() {
    nrow(self$data) - self$seq_len
  }
)

train_ds <- time_series_dataset(train_data_norm, seq_len)
valid_ds <- time_series_dataset(valid_data_norm, seq_len)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 64, shuffle = FALSE, drop_last = TRUE)

# --- 5. 定义 LSTM 模型 (适配多变量输入) ---

lstm_regressor <- nn_module(
  "lstm_regressor",
  
  initialize = function(input_size, hidden_size, output_size, num_layers) {
    self$lstm <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      batch_first = TRUE,
      dropout = 0.2 # 增加一点 dropout 防止过拟合
    )
    self$linear <- nn_linear(hidden_size, output_size)
  },
  
  forward = function(x) {
    # x shape: (batch, seq_len, input_size)
    out_list <- self$lstm(x)
    
    # 取序列最后一个时间步的输出
    output <- out_list[[1]] 
    last_step_out <- output[, dim(output)[2], ]
    
    out <- self$linear(last_step_out)
    return(out)
  }
)

# ---  训练 ---
n_features <- ncol(train_data_norm) # 7 (3个物理量 + 4个时间分量)

fitted <- lstm_regressor %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    input_size = n_features, 
    hidden_size = 128,
    output_size = 1,
    num_layers = 2 # 稍微加深网络
  ) %>%
  set_opt_hparams(
    lr = 0.001
  ) %>%
  fit(
    train_dl, 
    epochs = 20, 
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_early_stopping(monitor = "valid_loss", patience = 5)
    ),
    verbose = TRUE
  )

# --- 7. 预测与反标准化 ---
plot(fitted)

# 预测
preds_tensor <- predict(fitted, valid_dl)
preds_scaled <- as.numeric(preds_tensor)

# 获取真实值：对应的时间点是从 seq_len + 1 开始的，第1列 TEMP
actuals_scaled <- valid_data_norm[(seq_len + 1):nrow(valid_data_norm), 1]

# 反标准化 (Reverse Z-Score)
# 公式: x = z * sd + mean
preds_real <- preds_scaled * temp_sd + temp_mean
actuals_real <- actuals_scaled * temp_sd + temp_mean

# --- 8. 可视化 ---

plot_len <- 200
n <- nrow(df_clean)
tail_idx <- (length(preds_real) - plot_len + 1):length(preds_real)

df_res <- data.frame(
  time = df_clean$time[(n - plot_len + 1):n],
  Actual = actuals_real[tail_idx],
  Predicted = preds_real[tail_idx]
) %>%
  pivot_longer(cols = c("Actual", "Predicted"), names_to = "Type", values_to = "Value")

ggplot(df_res, aes(x = time, y = Value, color = Type)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "#333333", "Predicted" = "#E63946")) +
  labs(
    title = "", 
    subtitle = "",
    y = "温度 C",
    x = "时间步 (Hours)"
  )

metrics <- get_metrics(fitted)
print(metrics)

