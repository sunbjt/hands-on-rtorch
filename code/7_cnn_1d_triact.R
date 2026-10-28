library(tidyverse)
library(torch)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条
library(triact)
library(ggsci)

# --- 1. 获取数据路径 ---
input_dir <- system.file("extdata", package = "triact")
cow_tracker <- Triact$new()

# 逻辑：
# 1. sep = ";": 分隔符是分号
# 2. skip = "*DATA": 直接跳过所有元数据，找到 "*DATA" 这一行，从下一行开始读
# 3. header = FALSE: 因为真正的表头 "TIME..." 被我们跳过了，数据块的第一行就是数据，不是标题
cow_tracker$load_files(
  input = input_dir, 
  id_substring = c(1, 5),
  timeFwdUpRight_cols = c(1, 2, 3, 4),
  
  # 传递给 fread 的参数
  sep = ";",
  skip = "*DATA", 
  header = FALSE
)

# --- 4. 后续步骤 ---
cow_tracker$check_orientation() 
cow_tracker$add_lying()

cow_data <- cow_tracker$data

head(cow_data)

# 提取特征 (V2, V3, V4 对应 Fwd, Up, Right)
# 注意：因为 header=FALSE，列名可能是 V1, V2... 
# 或者 triact 会自动重命名它们。我们先看一下 names
colnames(cow_data)

# 无论列名是什么，triact 的 data 对象通常已经标准化了列名
# 我们可以放心地使用 select
features <- cow_data %>% 
  select(acc_fwd, acc_up, acc_right) %>% 
  as.matrix()

labels <- as.numeric(cow_data$lying)

# 标准化
features_scaled <- apply(features, 2, scale)

# 定义滑动窗口函数
create_windows <- function(data, labels, window_size, step_size) {
  n_samples <- nrow(data)
  n_windows <- floor((n_samples - window_size) / step_size) + 1
  
  # 预分配 3D 数组: (样本数, 通道数, 长度)
  # 注意：torch 1D CNN 需要 (Batch, Channel, Length)
  x_array <- array(0, dim = c(n_windows, 3, window_size))
  y_vec   <- numeric(n_windows)
  
  for (i in 1:n_windows) {
    start_idx <- (i - 1) * step_size + 1
    end_idx   <- start_idx + window_size - 1
    
    # 提取窗口数据并转置 (Length, 3) -> (3, Length)
    window_data <- data[start_idx:end_idx, ]
    x_array[i, , ] <- t(window_data)
    
    # 提取标签：这里采用"多数投票"或"取最后一个点"
    # 这里我们取窗口中间点的状态作为该窗口的标签
    mid_idx <- floor((start_idx + end_idx) / 2)
    y_vec[i] <- labels[mid_idx]
  }
  
  list(x = x_array, y = y_vec)
}

# --- 执行切片 ---
# 假设采样率是 10Hz (triact 默认)，我们取 5 秒作为一个窗口 (L=50)
WINDOW_SIZE <- 50 
STEP_SIZE   <- 25  # 50% 重叠

windows <- create_windows(features_scaled, labels, WINDOW_SIZE, STEP_SIZE)

# 划分训练集和测试集 (按时间顺序划分，避免数据泄露)
n_total <- length(windows$y)
split_idx <- floor(n_total * 0.8)

train_x <- windows$x[1:split_idx, , ]
train_y <- windows$y[1:split_idx]

test_x  <- windows$x[(split_idx + 1):n_total, , ]
test_y  <- windows$y[(split_idx + 1):n_total]

cat("训练集形状 (N, C, L):", paste(dim(train_x), collapse = " x "), "\n")
# 应输出: (样本数, 3, 50)


CowNet <- nn_module(
  "CowNet",
  initialize = function(num_classes = 2) {
    # Block 1: 捕捉初级波形特征 (32个通道)
    self$conv1 <- nn_sequential(
      nn_conv1d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2),
      nn_batch_norm1d(32),
      nn_relu(),
      nn_max_pool1d(kernel_size = 2) # 长度从 50 压缩至 25
    )
    # Block 2: 提取深层语义特征 (64个通道)
    self$conv2 <- nn_sequential(
      nn_conv1d(32, 64, kernel_size = 3, padding = 1),
      nn_batch_norm1d(64),
      nn_relu(),
      nn_adaptive_avg_pool1d(1) # 全局平均池化，压缩时间维
    )
    # 分类头
    self$fc <- nn_sequential(
      nn_flatten(),
      nn_linear(64, 32),
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(32, num_classes)
    )
  },
  forward = function(x) {
    x %>% self$conv1() %>% self$conv2() %>% self$fc()
  }
)

# 准备 Dataset 和 DataLoader
CowDataset <- dataset(
  name = "CowDataset",
  initialize = function(x, y) {
    self$x <- torch_tensor(x, dtype = torch_float())
    # 标签从 0/1 转换为 1/2 (为了配合 R 的 CrossEntropy)
    self$y <- torch_tensor(y + 1, dtype = torch_long())
  },
  .getitem = function(i) { list(x = self$x[i,..], y = self$y[i]) },
  .length = function() { dim(self$x)[1] }
)

train_dl <- dataloader(CowDataset(train_x, train_y), batch_size = 64, shuffle = TRUE)
test_dl  <- dataloader(CowDataset(test_x, test_y), batch_size = 64)

# 训练
fitted <- CowNet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams(num_classes = 2) %>%
  fit(train_dl, epochs = 5, valid_data = test_dl, verbose = TRUE)

# 评估
print(fitted)
plot(fitted)


# --- 可视化分析 ---

library(showtext)
showtext_auto()
library(ggplot2)
library(tidyverse)

# --- 1. 准备数据提取函数 (DRY原则) ---
get_sample_df <- function(x_data, y_data, target_class) {
  # 找到第一个符合该类别的样本索引
  # target_class: 0=Standing, 1=Lying
  idx <- which(y_data == target_class)[1] 
  
  # 提取数据并转置: (3, 50) -> (50, 3)
  sample_data <- t(x_data[idx, , ]) 
  
  # 构建数据框
  df <- as.data.frame(sample_data)
  colnames(df) <- c("Fwd", "Up", "Right") # 对应通道顺序
  
  df %>% 
    mutate(
      Time = 1:n(),
      Label = ifelse(target_class == 1, "Label: 1 (Lying/躺卧)", "Label: 0 (Standing/站立)"),
      Class_Id = target_class
    ) %>%
    pivot_longer(cols = c("Fwd", "Up", "Right"), names_to = "Axis", values_to = "Value")
}

# --- 2. 提取两个典型样本 ---
# 提取一个站立样本 (Class 0)
df_standing <- get_sample_df(test_x, test_y, 0)

# 提取一个躺卧样本 (Class 1)
df_lying <- get_sample_df(test_x, test_y, 1)

# 合并数据
plot_data <- bind_rows(df_standing, df_lying)

# --- 3. 绘制对比图 ---
p_contrast <- ggplot(plot_data, aes(x = Time, y = Value, color = Axis)) +
  # 画线
  geom_line(linewidth = 1.2) +
  # 画点 (可选，增加细节感)
  geom_point(size = 1.5, alpha = 0.6) +
  
  # --- 关键：使用分面 (Facet) 进行对比 ---
  facet_wrap(~Label, ncol = 1, scales = "fixed") + 
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 14, face = "bold", hjust = 0), # 标签字体放大
    strip.background = element_rect(fill = "#f0f0f0", color = NA), # 标签背景
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "right"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "",
    x = "时间步 (50个采样点 = 5秒)",
    y = "标准化加速度 (Z-Score)",
    color = "传感器轴"
  )

p_contrast

# --- 核的可视化 ---
get_feature_map_data <- function(model, x_data, y_data, target_class, label_text) {
  
  # 1. 提取对应类别的样本
  idx <- which(y_data == target_class)[1]
  feature_extractor <- model$conv1
  feature_extractor$eval()
  device <- feature_extractor$parameters[[1]]$device
  
  input_tensor <- torch_tensor(x_data[idx, , ])$
    unsqueeze(1)$
    to(device = device)
  
  with_no_grad({
    feature_map <- feature_extractor(input_tensor)
  })
  
  # (32, 25) -> 32个通道, 25个时间步
  fmap_matrix <- as.matrix(feature_map$squeeze(1)$cpu())
  
  # 转成 data.frame
  df <- as.data.frame(fmap_matrix)
  
  # 添加 Channel 列 (转为 Factor 以便画图时离散显示)
  df$Channel <- factor(seq_len(nrow(df)))
  
  # 2. 宽变长 (Pivot)
  fmap_melt <- df %>%
    pivot_longer(
      cols = -Channel,   # 除了 Channel 列，其他都变长
      names_to = "Time_Raw", # 暂存为原始列名 (V1, V2...)
      values_to = "Activation"
    ) %>%
    # --- [关键修正] ---
    # 去掉 "V"，转为数字。这样 x 轴就是 1, 2, 3... 25
    mutate(
      Time = as.numeric(gsub("V", "", Time_Raw)),
      Status = label_text
    )
  
  return(fmap_melt)
}

# --- 下面是绘图代码 (无需大改，只需运行) ---

model <- fitted$model

# 获取两种状态的数据
df_stand <- get_feature_map_data(model, test_x, test_y, 0, "模式 A: 站立 (Standing)")
df_lie   <- get_feature_map_data(model, test_x, test_y, 1, "模式 B: 躺卧 (Lying)")

plot_data <- bind_rows(df_stand, df_lie)

p_facet <- ggplot(plot_data, aes(x = Time, y = Channel, fill = Activation)) +
  geom_tile() +
  facet_wrap(~Status, ncol = 1) +
  scale_fill_viridis_c(option = "magma", name = "激活强度") +
  labs(
    title = "",
    x = "时间步 (Time Steps 1-25)", 
    y = "特征通道 (Channel 1-32)"
  ) +
  theme_minimal() +
  # 直接使用 Nature 杂志的配色
  scale_color_npg() + 
  # 或者使用新英格兰医学杂志配色: 
  # scale_color_nejm() +
  # 或者使用柳叶刀配色: 
  # scale_color_lancet() + 
  theme(
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22, face = "bold"),
    strip.text = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    legend.position = "right"
  )

print(p_facet)


