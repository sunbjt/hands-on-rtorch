# 数据下载地址
# https://huggingface.co/datasets/huggan/anime-faces
# 请注意解压后文件夹内还有一个重复的 data 文件夹，需删除

library(torch)
library(torchvision)
setwd('~/lsgan_luz_results/')
set.seed(123)
torch_manual_seed(123)


# --- 1. 配置参数 (WGAN-GP 专用配置) ---
opt <- list(
  dataroot = "~/data/AnimeFaces/",
  workers = 0,
  batch_size = 64,       # WGAN 建议 Batch 稍小一点，或者保持 64
  image_size = 64,
  nz = 100,
  ngf = 64,              # 64 (如果不关心极度细节，可改为 32 加速)
  ndf = 64,              # 同上
  num_epochs = 60,       # WGAN 收敛较快，不需要特别多轮
  lr = 0.0001,           # WGAN 学习率通常较低
  beta1 = 0.0,           # WGAN 的 Adam beta1 通常设为 0
  beta2 = 0.9,
  n_critic = 2,          # 判别器训练次数 / 生成器训练次数
  lambda_gp = 10         # 梯度惩罚系数
)

device <- if (torch_is_installed()) {
  if (torch_device("mps")$type == "mps") torch_device("mps") else if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
} else {
  torch_device("cpu")
}
cat("Using device:", device$type, "\n")

# --- 2. 数据加载 ---
transforms <- function(img) {
  img %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(opt$image_size) %>%
    torchvision::transform_center_crop(opt$image_size) %>%
    torchvision::transform_normalize(mean = c(0.5, 0.5, 0.5), std = c(0.5, 0.5, 0.5))
}

dataset <- image_folder_dataset(root = opt$dataroot, transform = transforms)
dataloader <- dataloader(dataset, batch_size = opt$batch_size, shuffle = TRUE, num_workers = opt$workers, drop_last = TRUE)

# --- 3. 模型定义 ---

# 权重初始化
weights_init <- function(m) {
  classname <- class(m)[1]
  if (grepl("conv", classname, ignore.case = TRUE)) {
    nn_init_normal_(m$weight, mean = 0.0, std = 0.02)
  } else if (grepl("batch_norm", classname, ignore.case = TRUE)) {
    nn_init_normal_(m$weight, mean = 1.0, std = 0.02)
    nn_init_constant_(m$bias, val = 0)
  }
}

# 生成器 (保持不变)
Generator <- nn_module(
  "Generator",
  initialize = function(nz, ngf) {
    self$main <- nn_sequential(
      nn_conv_transpose2d(nz, ngf * 8, 4, 1, 0, bias = FALSE),
      nn_batch_norm2d(ngf * 8),
      nn_relu(TRUE),
      
      nn_conv_transpose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = FALSE),
      nn_batch_norm2d(ngf * 4),
      nn_relu(TRUE),
      
      nn_conv_transpose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = FALSE),
      nn_batch_norm2d(ngf * 2),
      nn_relu(TRUE),
      
      nn_conv_transpose2d(ngf * 2, ngf, 4, 2, 1, bias = FALSE),
      nn_batch_norm2d(ngf),
      nn_relu(TRUE),
      
      nn_conv_transpose2d(ngf, 3, 4, 2, 1, bias = FALSE),
      nn_tanh()
    )
  },
  forward = function(input) {
    self$main(input)
  }
)

# 判别器 (Critic)
# 移除了所有 Norm 层，彻底解决 MPS 兼容性问题，速度最快
Discriminator <- nn_module(
  "Discriminator",
  initialize = function(ndf) {
    self$main <- nn_sequential(
      # 第 1 层
      nn_conv2d(3, ndf, 4, 2, 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE),
      
      # 第 2 层 (移除 GroupNorm)
      nn_conv2d(ndf, ndf * 2, 4, 2, 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE),
      
      # 第 3 层 (移除 GroupNorm)
      nn_conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE),
      
      # 第 4 层 (移除 GroupNorm)
      nn_conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE),
      
      # 输出层
      nn_conv2d(ndf * 8, 1, 4, 1, 0, bias = FALSE)
    )
  },
  forward = function(input) {
    self$main(input)
  }
)

# --- 4. 辅助函数：梯度惩罚 (Gradient Penalty) ---
compute_gradient_penalty <- function(netD, real_samples, fake_samples, device) {
  
  # 1. 随机权重 alpha
  alpha <- torch_rand(c(real_samples$size(1), 1, 1, 1), device = device)
  
  # 2. 计算插值图像 (Interpolates)
  interpolates <- (alpha * real_samples + (1 - alpha) * fake_samples)$requires_grad_(TRUE)
  
  # 3. 将插值图像送入判别器
  d_interpolates <- netD(interpolates)
  
  # 4. 创建一个全 1 的张量，用于计算梯度
  fake_grad <- torch_ones(d_interpolates$shape, device = device)
  
  # 5. 计算梯度
  # [修正点] 移除了 only_inputs = TRUE
  gradients <- autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake_grad,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  # 6. 展平梯度
  gradients <- gradients$view(c(gradients$size(1), -1))
  
  # 7. 计算梯度的范数 (L2 Norm)
  # [说明] WGAN-GP 要求梯度范数接近 1
  gradient_penalty <- ((gradients$norm(2, dim = 2) - 1) ^ 2)$mean() * opt$lambda_gp
  
  return(gradient_penalty)
}

# --- 5. 初始化模型与优化器 ---
netG <- Generator(opt$nz, opt$ngf)$to(device = device)
netD <- Discriminator(opt$ndf)$to(device = device)

netG$apply(weights_init)
netD$apply(weights_init)

fixed_noise <- torch_randn(8, opt$nz, 1, 1, device = device)

# [注意] Adam 的 beta1 设为 0
optimizerD <- optim_adam(netD$parameters, lr = opt$lr, betas = c(opt$beta1, opt$beta2))
optimizerG <- optim_adam(netG$parameters, lr = opt$lr, betas = c(opt$beta1, opt$beta2))

# --- 6. 训练循环 (WGAN-GP 修正版) ---
cat("Starting WGAN-GP Training...\n")


G_losses <- c()
D_losses <- c()

for (epoch in 1:opt$num_epochs) {
  
  i <- 0
  # 创建迭代器
  iter_loader <- dataloader$.iter() 
  
  while(TRUE) {
    i <- i + 1
    
    ############################
    # (1) 更新判别器 (Critic) - 循环 n_critic 次
    ############################
    critic_loop_break <- FALSE
    
    for (nc in 1:opt$n_critic) {
      
      batch <- iter_loader$.next()
      
      # [修正点] 使用 coro::is_exhausted 判断是否结束
      if (coro::is_exhausted(batch)) {
        critic_loop_break <- TRUE
        break 
      }
      
      real_imgs <- batch[[1]]$to(device = device)
      b_size <- real_imgs$size(1)
      
      netD$zero_grad()
      
      # --- 1.1 真实图片得分 ---
      output_real <- netD(real_imgs)$view(-1)
      loss_real <- -torch_mean(output_real)
      
      # --- 1.2 假图片得分 ---
      noise <- torch_randn(b_size, opt$nz, 1, 1, device = device)
      fake_imgs <- netG(noise)$detach()
      output_fake <- netD(fake_imgs)$view(-1)
      loss_fake <- torch_mean(output_fake)
      
      # --- 1.3 梯度惩罚 ---
      gradient_penalty <- compute_gradient_penalty(netD, real_imgs, fake_imgs, device)
      
      # --- 1.4 总 Loss ---
      loss_D <- loss_real + loss_fake + gradient_penalty
      loss_D$backward()
      optimizerD$step()
    }
    
    # 如果数据跑完了，跳出外层 while 循环，进入下一个 Epoch
    if (critic_loop_break) break 
    
    ############################
    # (2) 更新生成器 (Generator)
    ############################
    netG$zero_grad()
    
    # 注意：这里需要重新生成噪音，不要复用上面的
    # 因为上面的 fake_imgs 是 detach() 过的，没有梯度信息
    noise <- torch_randn(b_size, opt$nz, 1, 1, device = device)
    fake_imgs_g <- netG(noise)
    output <- netD(fake_imgs_g)$view(-1)
    
    loss_G <- -torch_mean(output)
    loss_G$backward()
    optimizerG$step()
    
    G_losses <- c(G_losses, loss_G$item())
    D_losses <- c(D_losses, loss_D$item())
    
    if (i %% 20 == 0) {
      cat(sprintf("[%d/%d][%d] Loss_D: %.4f Loss_G: %.4f\n",
                  epoch, opt$num_epochs, i, 
                  loss_D$item(), loss_G$item()))
    }
  }
  
  # --- 保存图片 ---
  cat(sprintf("End of Epoch %d. Saving image...\n", epoch))
  with_no_grad({
    fake_imgs <- netG(fixed_noise)
    grid <- torchvision::vision_make_grid(fake_imgs, scale = TRUE, num_rows = 8, padding = 2)
    grid <- grid$permute(c(2, 3, 1))$cpu()
    grid_array <- as.array(grid)
    
    if (requireNamespace("png", quietly = TRUE)) {
      png::writePNG(grid_array, sprintf("wgan_epoch_%03d.png", epoch))
    }
  })
}


# 我们保存一个列表，包含生成器、判别器、优化器和当前轮数
checkpoint <- list(
  epoch = epoch,
  netG_state_dict = netG$state_dict(),
  netD_state_dict = netD$state_dict(),
  optimizerG_state_dict = optimizerG$state_dict(),
  optimizerD_state_dict = optimizerD$state_dict(),
  losses = list(G = G_losses, D = D_losses) # 顺便保存 Loss 记录
)

# 保存为 .pt 文件
torch_save(checkpoint, "/Users/liusizhe/data/wgan_gp_checkpoint.pt")
cat(sprintf("Checkpoint saved at epoch %d\n", epoch))

