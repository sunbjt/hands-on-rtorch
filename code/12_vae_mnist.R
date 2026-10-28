# 数据下载地址：
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset

library(tidyverse)
library(ggplot2)
library(torch)
library(patchwork)
library(luz)

setwd('~/data/')

load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

train <- load_image_file('archive/train-images.idx3-ubyte')
test <- load_image_file('archive/t10k-images.idx3-ubyte')

train$y <- load_label_file('archive/train-labels.idx1-ubyte')
test$y <- load_label_file('archive/t10k-labels.idx1-ubyte')  

set.seed(123)
torch_manual_seed(123)

preprocess_data <- function(data_list) {
  images <- data_list$x / 255
  n_samples <- nrow(images)
  # 重塑逻辑优化：直接构建 [N, 28, 28]
  images_array <- array(images, dim = c(n_samples, 28, 28))
  # 转换为 Tensor: [N, 28, 28] -> [N, 1, 28, 28]
  images_tensor <- torch_tensor(images_array, dtype = torch_float32())
  images_tensor <- images_tensor$unsqueeze(2) 
  labels_tensor <- torch_tensor(data_list$y + 1, dtype = torch_long()) 
  list(images = images_tensor, labels = labels_tensor)
}

train_data <- preprocess_data(train)
test_data <- preprocess_data(test)

# 定义 VAE 专用的 Dataset
# VAE 不需要 label，它的目标是重建输入，所以 Target = Input
vae_dataset <- dataset(
  name = "mnist_vae_dataset",
  initialize = function(images) {
    self$data <- images
  },
  .getitem = function(i) {
    # 返回 list(input, target)
    img <- self$data[i, , , , drop = FALSE] 
    list(x = img, y = img)
  },
  .length = function() {
    self$data$size(1)
  }
)

# 实例化 Dataset
train_ds <- vae_dataset(train_data$images)
test_ds <- vae_dataset(test_data$images)

# 实例化 Dataloader
train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 128, shuffle = FALSE)

vae_module <- nn_module(
  "VAE",
  
  initialize = function(input_dim = 784, hidden_dim = 400, latent_dim = 20) {
    
    # Encoder (编码器)
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    
    # 输出均值 mu 和 对数方差 logvar
    self$fc2_mu <- nn_linear(hidden_dim, latent_dim)
    self$fc2_logvar <- nn_linear(hidden_dim, latent_dim)
    
    # Decoder (解码器)
    self$fc3 <- nn_linear(latent_dim, hidden_dim)
    self$fc4 <- nn_linear(hidden_dim, input_dim)
  },
  
  # 重参数化技巧：z = mu + epsilon * sigma
  reparameterize = function(mu, logvar) {
    std <- torch_exp(0.5 * logvar)
    eps <- torch_randn_like(std)
    z <- mu + eps * std
    return(z)
  },
  
  forward = function(x) {
    # 1. 展平输入: [Batch, 1, 28, 28] -> [Batch, 784]
    x_flat <- x$view(c(-1, 784))
    
    # 2. 编码
    h <- x_flat %>% self$fc1() %>% torch_relu()
    mu <- self$fc2_mu(h)
    logvar <- self$fc2_logvar(h)
    
    # 3. 采样 (训练时采样，预测时为了稳定通常也采样，或者直接用 mu)
    z <- self$reparameterize(mu, logvar)
    
    # 4. 解码
    recon <- z %>% self$fc3() %>% torch_relu() %>%
      self$fc4() %>% torch_sigmoid()
    
    # 5. 返回 list: [重建图像, 均值, 对数方差]
    # 我们需要 mu 和 logvar 来计算 KL 散度损失
    list(recon, mu, logvar)
  }
)

vae_loss <- function(predictions, targets) {
  
  recon_x <- predictions[[1]]
  mu      <- predictions[[2]]
  logvar  <- predictions[[3]]
  
  # 展平 targets 以匹配 recon_x
  targets_flat <- targets$view(c(-1, 784))
  
  # 1. 重建损失 (Reconstruction Loss)
  # 使用二元交叉熵 (BCE)，因为我们将像素归一化到了 [0, 1]
  # reduction = 'sum' 表示对一个 batch 的所有误差求和
  BCE <- nnf_binary_cross_entropy(recon_x, targets_flat, reduction = 'sum')
  
  # 2. KL 散度 (Kullback-Leibler Divergence)
  # 衡量潜在分布与标准正态分布的差异
  # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD <- -0.5 * torch_sum(1 + logvar - mu$pow(2) - logvar$exp())
  
  # 返回总损失
  BCE + KLD
}

fitted <- vae_module %>%
  setup(
    loss = vae_loss,
    optimizer = optim_adamw
  ) %>%
  set_hparams(input_dim = 784, hidden_dim = 400, latent_dim = 20) %>%
  fit(
    train_dl,
    epochs = 20,
    valid_data = test_dl,
    verbose = TRUE
  )

plot(fitted)


generate_digits <- function(luz_fitted_obj, n = 16) {
  # 提取原始模型
  raw_model <- luz_fitted_obj$model
  raw_model$eval()

  device <- raw_model$parameters[[1]]$device
  z <- torch_randn(n, 20, device = device)
  
  # 解码
  with_no_grad({
    samples <- z %>% 
      raw_model$fc3() %>% 
      torch_relu() %>% 
      raw_model$fc4() %>% 
      torch_sigmoid()
  })
  
  samples_cpu <- samples$to(device = "cpu")
  samples_matrix <- as_array(samples_cpu$view(c(n, 28, 28)))
  
  plot_data <- map_dfr(1:n, function(i) {
    expand.grid(row = 1:28, col = 1:28) %>%
      mutate(
        val = as.vector(t(samples_matrix[i,,])), 
        img_id = i
      )
  })
  
  ggplot(plot_data, aes(x = col, y = -row, fill = val)) +
    geom_raster() +
    scale_fill_gradient(low = "black", high = "white") +
    facet_wrap(~img_id, ncol = 8) +
    coord_fixed() +
    theme_void(base_family = 'Noto Sans SC') +
    theme(legend.position = "none")
}

# 再次尝试调用
p1 <- generate_digits(fitted, n = 16)
p1

ggsave(
  filename = 'vae_generate.png',
  plot = p1,
  device = 'png',
  path = '~/github/rtorchbook/part-04-frontiers/images/',
  width = 15,
  height = 10,
  units = "cm"
)

# --- 绘制潜在空间分布图 ---
# 先是传统方法
library(Rtsne)
library(umap)
test_imgs <- test_data$images[1:2000, ..]$view(c(-1, 784)) |>
  as.matrix()
test_lbls <- test_data$labels[1:2000] |> as.matrix() |> as.character()

# PCA
pca_result <- prcomp(test_imgs) 
pca_2d <- data.frame(PC1 = pca_result$x[,1], 
                     PC2 = pca_result$x[,2], 
                     Label = test_lbls)
# t-SNE
tsne_out <- Rtsne(test_imgs, dims = 2, perplexity = 30, verbose = TRUE, check_duplicates = FALSE)

tsne_2d <- data.frame(Dim1 = tsne_out$Y[,1], 
                      Dim2 = tsne_out$Y[,2], 
                      Label = test_lbls)
# UMAP
umap_out <- umap(test_imgs)
umap_2d <- data.frame(Dim1 = umap_out$layout[,1], 
                      Dim2 = umap_out$layout[,2], 
                      Label = test_lbls)

# LDA
X_pca <- pca_result$x[, 1:50]
lda_model <- MASS::lda(x = as.data.frame(X_pca), grouping = test_lbls)

lda_out <- predict(lda_model, as.data.frame(X_pca))
lda_2d <- data.frame(Dim1 = lda_out$x[,1], 
                            Dim2 = lda_out$x[,2], 
                            Label = test_lbls)

p1 <- ggplot(pca_2d, aes(x=PC1, y=PC2, color=Label)) +
  geom_point(alpha=0.6, size=1.5) +
  theme_minimal() +
  labs(title = "PCA Visualization", subtitle = "Linear Projection")

p2 <- ggplot(tsne_2d, aes(x=Dim1, y=Dim2, color=Label)) +
  geom_point(alpha=0.6, size=1.5) +
  theme_minimal() +
  labs(title = "t-SNE Visualization", subtitle = "Focus on Local Clusters")

p3 <- ggplot(umap_2d, aes(x=Dim1, y=Dim2, color=Label)) +
  geom_point(alpha=0.6, size=1.5) +
  theme_minimal() +
  labs(title = "UMAP Visualization", subtitle = "Balances Local & Global Structure")

p4 <- ggplot(lda_2d, aes(x=Dim1, y=Dim2, color=Label)) +
  geom_point(alpha=0.6, size=1.5) +
  theme_minimal() +
  labs(title = "LDA Visualization", subtitle = "Fixed 'constant within groups' error")


# 显示图形
print(p1)
print(p2)
print(p3)
print(p4)

# 用 VAE 降维
# 1. 重新训练一个 2D VAE
fitted_2d <- vae_module %>%
  setup(loss = vae_loss, optimizer = optim_adam) %>%
  set_hparams(input_dim = 784, hidden_dim = 400, latent_dim = 2) %>%
  fit(train_dl, epochs = 20, valid_data = test_dl) # 多跑几轮

# 2. 直接获取 2D 坐标
visualize_2d_direct <- function(fitted_obj, test_data) {
  raw_model <- fitted_obj$model
  raw_model$eval()
  device <- raw_model$parameters[[1]]$device
  
  # 取部分数据
  test_imgs <- test_data$images[1:2000, ..]$to(device = device)
  test_lbls <- test_data$labels[1:2000]$to(device = "cpu")
  
  with_no_grad({
    h <- test_imgs$view(c(-1, 784)) %>% raw_model$fc1() %>% torch_relu()
    mu <- raw_model$fc2_mu(h) # 输出已经是 [2000, 2]
  })
  
  mu_cpu <- as_array(mu$to(device = "cpu"))
  
  df_plot <- data.frame(
    Dim1 = mu_cpu[,1],
    Dim2 = mu_cpu[,2],
    Label = factor(as_array(test_lbls) - 1)
  )
  
  ggplot(df_plot, aes(x = Dim1, y = Dim2, color = Label)) +
    geom_point(alpha = 0.6, size = 2) +
#    scale_color_viridis_d() +
    theme_minimal(base_family = 'Noto Sans SC') +
    labs(title = "VAE 2D 潜在空间直接映射",
         subtitle = "Latent Dim = 2")
}

p_2d <- visualize_2d_direct(fitted_2d, test_data)
p_2d
ggsave(
  filename = 'vae_2d.png',
  plot = p_2d,
  device = 'png',
  path = '~/github/rtorchbook/part-04-frontiers/images/',
  width = 20,
  height = 15,
  units = "cm"
)



plot_manifold <- function(luz_fitted_obj, n = 20, canvas_size = 28) {
  # 提取模型与设备信息
  raw_model <- luz_fitted_obj$model
  raw_model$eval()
  device <- raw_model$parameters[[1]]$device
  
  # 1. 创建网格坐标 (Grid Coordinates)
  # 我们在 [-3, 3] 之间生成 n 个点，这覆盖了正态分布 99% 的区域
  grid_x <- seq(-3, 3, length.out = n)
  grid_y <- seq(3, -3, length.out = n) # y轴从上到下，所以是从正到负
  
  # 生成网格对
  grid <- expand.grid(x = grid_x, y = grid_y)
  
  # 转换为 Tensor 放入设备
  z_grid <- torch_tensor(matrix(c(grid$x, grid$y), ncol = 2), device = device)
  
  # 2. 批量解码 (Batch Decode)
  with_no_grad({
    # z -> fc3 -> relu -> fc4 -> sigmoid
    # 注意：这里我们绕过了 forward，直接用解码部分
    recons <- z_grid %>% 
      raw_model$fc3() %>% 
      torch_relu() %>% 
      raw_model$fc4() %>% 
      torch_sigmoid()
  })
  
  # 转回 CPU: [400, 784]
  recons_cpu <- as_array(recons$to(device = "cpu"))
  
  # 3. 拼图魔法 (Stitching Images)
  # 我们不使用 facet，而是计算每个像素在“大画布”上的全局坐标
  # 这是一个纯粹的数据处理技巧
  
  plot_data <- map_dfr(1:nrow(grid), function(i) {
    # 当前是第几行第几列的图片?
    grid_col_idx <- (i - 1) %% n + 1
    grid_row_idx <- ceiling(i / n)
    
    # 提取当前图片的像素值 [784]
    img_vals <- recons_cpu[i, ]
    
    # 构建 28x28 的局部坐标
    expand.grid(r = 1:canvas_size, c = 1:canvas_size) %>% 
      mutate(
        val = img_vals, # 填充像素
        # 计算全局坐标:
        # Global X = (当前图片列号 - 1) * 28 + 局部列号
        global_x = (grid_col_idx - 1) * canvas_size + c,
        # Global Y = (当前图片行号 - 1) * 28 + 局部行号
        global_y = (grid_row_idx - 1) * canvas_size + r 
      )
  })
  
  # 4. 绘图
  ggplot(plot_data, aes(x = global_x, y = -global_y, fill = val)) +
    geom_raster() +
    scale_fill_gradient(low = "black", high = "white") +
    coord_fixed() + # 保证不拉伸
    theme_void(base_family = 'Noto Sans SC') +  # 去掉坐标轴
    theme(legend.position = "none")
}

p_nf <- plot_manifold(fitted_2d, n = 20)

ggsave(
  filename = 'vae_manifold.png',
  plot = p_nf,
  device = 'png',
  path = '~/github/rtorchbook/part-04-frontiers/images/',
  width = 20,
  height = 15,
  units = "cm"
)

