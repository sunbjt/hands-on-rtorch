# 数据下载地址：
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset

library(tidyverse)
library(ggplot2)
library(torch)
library(patchwork)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

# --- 1. 数据加载函数 (保持不变) ---
load_image_file <- function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x    = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  list(x = x, n = n, nrow = nrow, ncol = ncol)
}

load_label_file <- function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

setwd('~/data') 
train_raw <- load_image_file('archive/train-images.idx3-ubyte')
test_raw  <- load_image_file('archive/t10k-images.idx3-ubyte')
train_y   <- load_label_file('archive/train-labels.idx1-ubyte')
test_y    <- load_label_file('archive/t10k-labels.idx1-ubyte')

# --- 2. 数据预处理 (核心修复) ---
preprocess_data <- function(raw_x, raw_y, n, h, w) {
  images <- raw_x / 255
  
  # === 关键修正 ===
  # 1. 先按单张图填充：c(28, 28, n)。
  #    R 是列优先填充，这样能确保一张图的像素聚在一起，不会跨样本混淆。
  images_array <- array(images, dim = c(h, w, n))
  
  # 2. 调整维度顺序：[Row, Col, N] -> [N, Row, Col]
  images_array <- aperm(images_array, c(3, 2, 1))
  
  # 3. 转为 Tensor: [N, 1, 28, 28]
  images_tensor <- torch_tensor(images_array, dtype = torch_float32())$unsqueeze(2)
  
  # 4. 标签: 0-9 -> 1-10
  labels_tensor <- torch_tensor(raw_y + 1, dtype = torch_long())
  
  list(images = images_tensor, labels = labels_tensor)
}

train_data <- preprocess_data(train_raw$x, train_y, train_raw$n, 28, 28)
test_data  <- preprocess_data(test_raw$x, test_y, test_raw$n, 28, 28)

# --- 3. 可视化工具 ---
visualize_image <- function(img_tensor) {
  arr <- as_array(img_tensor)
  if (length(dim(arr)) > 2) arr <- arr[1, 1, , ] 
  
  # 转置以适配 ggplot
  arr <- t(arr)
  
  df <- expand.grid(x = 1:ncol(arr), y = nrow(arr):1)
  df$val <- as.vector(arr)
  
  ggplot(df, aes(x, y, fill = val)) +
    geom_tile() + 
    scale_fill_gradient(low = "white", high = "black") +
    theme_void() + 
    theme(legend.position = "none") +
    coord_fixed()
}

visualize_feature_maps <- function(fmaps) {
  if (length(dim(fmaps)) == 4) fmaps <- fmaps[1, , , ]
  fmaps_arr <- as_array(fmaps)
  
  n_ch <- min(dim(fmaps_arr)[1], 16)
  plots <- list()
  
  for (i in 1:n_ch) {
    fm <- t(fmaps_arr[i, , ])
    df <- expand.grid(x = 1:ncol(fm), y = nrow(fm):1)
    df$val <- as.vector(fm)
    plots[[i]] <- ggplot(df, aes(x, y, fill = val)) +
      geom_tile() + scale_fill_gradient(low = "black", high = "white") +
      theme_void() + theme(legend.position = "none") + coord_fixed()
  }
  wrap_plots(plots, ncol = 4) 
}

# --- 卷积核(权重)的可视化 ---
visualize_kernels <- function(model_layer) {
  # 提取权重: [Out_Channels, In_Channels, H, W] -> [16, 1, 3, 3]
  weights <- as_array(model_layer$weight$cpu())
  
  # 归一化到 0-1 之间以便绘图 (因为权重有正有负)
  weights <- (weights - min(weights)) / (max(weights) - min(weights))
  
  plots <- list()
  for (i in 1:dim(weights)[1]) {
    # 取出第 i 个核 (3x3)
    k <- weights[i, 1, , ]
    # 转置以适配 ggplot
    k <- t(k)
    
    df <- expand.grid(x = 1:3, y = 3:1)
    df$val <- as.vector(k)
    
    plots[[i]] <- ggplot(df, aes(x, y, fill = val)) +
      geom_tile(color="gray") + # 加网格线
      scale_fill_gradient(low = "white", high = "black") +
      theme_void() + 
      theme(legend.position = "none", plot.title = element_text(size=8, hjust=0.5)) +
      coord_fixed() +
      ggtitle(paste("Ker", i))
  }
  wrap_plots(plots, ncol = 8)
}

cat("验证第一张图片 (应为数字 5):\n")
print(visualize_image(train_data$images[1, , , , drop=FALSE]))


# --- 4. 定义模型 ---
cnn_module <- nn_module(
  "MNIST_CNN",
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 16, kernel_size = 3, padding = 1)
    self$pool  <- nn_max_pool2d(kernel_size = 2)
    self$conv2 <- nn_conv2d(16, 32, kernel_size = 3, padding = 1)
    self$fc1   <- nn_linear(32 * 7 * 7, 128)
    self$fc2   <- nn_linear(128, 10)
    self$dropout <- nn_dropout(p = 0.5)
  },
  forward = function(x) {
    # 第一层块: Conv -> ReLU -> Pool
    x <- self$conv1(x) %>% nnf_relu()   # -> [N, 16, 28, 28] (padding=1 保持尺寸)
    x <- self$pool(x)                   # -> [N, 16, 14, 14] (尺寸减半)
    # 第二层块: Conv -> ReLU -> Pool
    x <- self$conv2(x) %>% nnf_relu()   # -> [N, 32, 14, 14] (通道翻倍)
    x <- self$pool(x)                   # -> [N, 32, 7, 7]   (再次减半)
    # 展平: 将三维特征图拉直为一维向量
    x <- torch_flatten(x, start_dim = 2)# -> [N, 1568]
    # 全连接分类
    x <- self$fc1(x) %>% nnf_relu()     # -> [N, 128]
    x <- self$dropout(x)
    x <- self$fc2(x)                    # -> [N, 10]
    return(x)
  },
  forward_to_layer = function(x, layer_name) {
    out <- self$conv1(x)
    if (layer_name == "conv1") return(nnf_relu(out))
    out <- self$pool(nnf_relu(out))
    if (layer_name == "pool1") return(out)
    out <- self$conv2(out)
    if (layer_name == "conv2") return(nnf_relu(out))
    out <- self$pool(nnf_relu(out))
    if (layer_name == "pool2") return(out)
    NULL
  }
)

# --- 5. 训练 ---
device <- if (cuda_is_available()) "cuda" else if (backends_mps_is_available()) "mps" else "cpu"
cat(sprintf("\n使用计算设备: %s\n", device))

train_dl <- dataloader(
  tensor_dataset(x = train_data$images, y = train_data$labels),
  batch_size = 64, shuffle = TRUE
)
test_dl <- dataloader(
  tensor_dataset(x = test_data$images, y = test_data$labels),
  batch_size = 64
)

cat("开始训练 (3 Epochs)...\n")
fitted <- cnn_module %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams() %>%
  fit(
    train_dl, 
    epochs = 3, # 3 轮演示
    valid_data = test_dl
  )

plot(fitted)

# --- 6. 卷积的可视化 ---

# 提取样本并移动到正确的设备
sample_img <- train_data$images[5, , , , drop=FALSE]$to(device = device)

# 初始化的
initial_model <- cnn_module()
initial_model$to(device = device)
initial_model$eval() # 评估模式

# 训练后的
trained_model <- fitted$model
trained_model$to(device = device)
trained_model$eval() # 评估模式

# 顺便看下各层的参数量
sapply(initial_model$parameters, function(x) x$numel())

# 可视化初始化的 16 个卷积核 (滤波器)
visualize_kernels(initial_model$conv1)
# 更新后的 Conv1 核可视化
visualize_kernels(trained_model$conv1)

# 可视化初始化情况的特征图
with_no_grad({
  c1_initial <- initial_model$forward_to_layer(sample_img, "conv1")
})
visualize_feature_maps(c1_initial$cpu())
# 可视化训练完模型后的特征图
with_no_grad({
  c1_trained <- trained_model$forward_to_layer(sample_img, "conv1")
})
visualize_feature_maps(c1_trained$cpu())



# --- 7. 更深层的变化 (Conv2) ---

# A. 初始状态的 Conv2 (随机噪声)
with_no_grad({
  # 使用之前定义的 initial_model
  c2_initial <- initial_model$forward_to_layer(sample_img, "conv2")
})

cat("初始 Conv2 (未训练): \n")
print(visualize_feature_maps(c2_initial$cpu()))


# B. 训练后的 Conv2 (有效特征)
with_no_grad({
  # 使用训练好的 trained_model
  c2_trained <- trained_model$forward_to_layer(sample_img, "conv2")
})

cat("\n训练后 Conv2 (已训练): \n")
print(visualize_feature_maps(c2_trained$cpu()))


# --- 8. 混淆矩阵 ---
cat("\n计算最终混淆矩阵...\n")

# 获取预测
preds_prob <- predict(fitted, test_dl)

# 转换: Tensor (GPU/MPS) -> CPU -> R Array -> R Vector
# 必须先转 array 再做 table，否则会报错或非常慢
preds_class <- as_array(torch_argmax(preds_prob, dim = 2)$cpu()) - 1
actual_class <- as_array(test_data$labels$cpu()) - 1

cat("混淆矩阵:\n")
print(table(Predicted = preds_class, Actual = actual_class))


# --- 9.VGG 网络 ---

# A. 定义 VGG Block 生成函数
# 这是一个"工厂函数"，根据参数动态生成包含 卷积+ReLU+池化 的序列
vgg_block <- function(num_convs, in_channels, out_channels) {
  layers <- list()
  
  for (i in 1:num_convs) {
    layers[[length(layers) + 1]] <- nn_conv2d(
      in_channels = in_channels, 
      out_channels = out_channels, 
      kernel_size = 3, 
      padding = 1
    )
    layers[[length(layers) + 1]] <- nn_relu()
    
    # 如果块内有多层卷积，后续层的输入通道 = 当前层的输出通道
    in_channels <- out_channels
  }
  
  # 块的末尾添加最大池化层 (尺寸减半)
  layers[[length(layers) + 1]] <- nn_max_pool2d(kernel_size = 2, stride = 2)
  
  # 将列表转换为 nn_sequential 容器
  do.call(nn_sequential, layers)
}

# B. 定义 Mini-VGG 网络结构
MiniVGG <- nn_module(
  "MiniVGG",
  initialize = function() {
    # === 特征提取部分 (Features) ===
    # 使用 nn_sequential 串联两个 VGG Block
    self$features <- nn_sequential(
      # Block 1: 1个卷积, 1->64通道. 图片尺寸: 28x28 -> 14x14
      vgg_block(num_convs = 2, in_channels = 1, out_channels = 64),
      
      # Block 2: 1个卷积, 64->128通道. 图片尺寸: 14x14 -> 7x7
      vgg_block(num_convs = 2, in_channels = 64, out_channels = 128)
    )
    
    # === 分类器部分 (Classifier) ===
    self$classifier <- nn_sequential(
      nn_flatten(),
      # 输入维度计算: 128 (通道) * 7 (高) * 7 (宽)
      nn_linear(128 * 7 * 7, 256), 
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(256, 10)
    )
  },
  forward = function(x) {
    x %>% 
      self$features() %>% 
      self$classifier()
  }
)

cat("开始训练 MiniVGG (3 Epochs)...\n")
fitted_vgg <- MiniVGG %>%  # <--- 这里改为 MiniVGG
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams() %>%
  fit(
    train_dl, 
    epochs = 3, 
    valid_data = test_dl,
    verbose = TRUE # 显示进度条
  )

plot(fitted_vgg)

# --- 6. (可选) 查看 VGG 的结构 ---
model_instance <- MiniVGG()
print(model_instance)

