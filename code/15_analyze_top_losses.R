# 使用 图片 10 分类的案例说明
# 目的是找到 top Loss 的样本
# 首先执行 7_cnn_CIFAR-10.R 文件 Fine-tuning 模块之前代码，第 86 行。
# 将训练过程的 freeze_backbone 参数改为 FALSE，解冻所有参数

analyze_top_losses <- function(model, dataloader, device, top_k = 5) {
  
  model$eval() # 务必切换到评估模式 (关闭 Dropout 等)
  
  # 用于收集结果的列表 (比在循环中不断 c() 向量效率更高)
  results_list <- list()
  batch_counter <- 1
  
  # 定义不进行平均的损失函数
  criterion_no_reduce <- nn_cross_entropy_loss(reduction = "none")
  
  coro::loop(for (batch in dataloader) {
    # 1. 数据迁移 (cifar10_dataset 返回的是 list(x, y))
    inputs <- batch[[1]]$to(device = device)
    targets <- batch[[2]]$to(device = device)
    
    # 2. 前向传播
    with_no_grad({
      outputs <- model(inputs)
      # 3. 计算逐样本损失 (关键步骤)
      # 结果是一个形状为 (batch_size) 的向量，而不是标量
      individual_losses <- criterion_no_reduce(outputs, targets)
    })
    
    # 4. 收集结果
    # 我们将 tensor 转为数值存储
    results_list[[batch_counter]] <- as.numeric(individual_losses$cpu())
    batch_counter <- batch_counter + 1
  })
  
  # 5. 整合所有 Batch 的 Loss
  all_losses <- unlist(results_list)
  
  # 6. 找出 Loss 最高的 Top-N 索引
  # order 默认升序，decreasing = TRUE 获取最大的 Loss
  sorted_indices <- order(all_losses, decreasing = TRUE)
  top_n_indices <- head(sorted_indices, top_k)
  top_n_values <- head(all_losses[sorted_indices], top_k)
  
  return(list(indices = top_n_indices, losses = top_n_values))
}

# 1. 从 luz fitted 对象中提取原始 torch 模型
# fitted$model 是我们定义的 MobileNetV2 实例
raw_model <- fitted$model

# 2. 运行诊断分析
# 注意：这里使用验证集/测试集 test_dl
diagnosis <- analyze_top_losses(
  model = raw_model, 
  dataloader = test_dl, 
  top_k = 6,
  device = if (cuda_is_available()) "cuda" else if (backends_mps_is_available()) "mps" else "cpu"
)

print(paste("Top 6 Losses:", paste(round(diagnosis$losses, 4), collapse = ", ")))
print(paste("对应样本索引:", paste(diagnosis$indices, collapse = ", ")))

# 定义标准的反归一化参数 (ImageNet)
mean_vals <- c(0.485, 0.456, 0.406)
std_vals  <- c(0.229, 0.224, 0.225)

# 定义一个辅助函数来显示图像
plot_bad_case <- function(dataset, index, loss_val, model, device = "cpu") {
  
  # 1. 确保模型也在指定的设备上 (修复报错的关键)
  model$to(device = device)
  model$eval()
  
  # 2. 获取原始数据
  sample <- dataset[index] # list(x, y)
  img_tensor <- sample[[1]]
  true_label_idx <- as.numeric(sample[[2]])
  
  # 3. 准备输入：增加 batch 维度并移动到设备
  input <- img_tensor$unsqueeze(1)$to(device = device) 
  
  # 4. 模型推理
  logits <- model(input)
  pred_label_idx <- as.numeric(torch_argmax(logits, dim = 2)$cpu())
  
  # 5. 图像处理优化：标准的“反归一化” (Un-normalization)
  # 逻辑：Pixel = (Tensor * Std) + Mean
  img_tensor <- img_tensor$permute(c(2, 3, 1))$cpu() # C,H,W -> H,W,C
  img_array <- as.array(img_tensor)
  
  # 手动反归一化
  for (c in 1:3) {
    img_array[,,c] <- img_array[,,c] * std_vals[c] + mean_vals[c]
  }
  
  # 截断超出 [0,1] 范围的值 (Clip)
  img_array[img_array > 1] <- 1
  img_array[img_array < 0] <- 0
  # 6. 绘图
  plot(as.raster(img_array))
  true_name <- dataset$classes[true_label_idx]
  pred_name <- dataset$classes[pred_label_idx]
  title(main = sprintf("Loss: %.2f\nTrue: %s | Pred: %s", 
                       loss_val, true_name, pred_name),
        col.main = "red", cex.main = 1.2)
}

# --- 调用部分 ---

# 绘制 Loss 最高的图片
raw_model$to(device = "cpu")
num_plots <- min(6, length(diagnosis$indices))


setwd('~/github/rtorchbook/part-05-engineering/image/')
png(filename = 'badcase.png', 
    width = 1800,  # 双倍宽度
    height = 1200,  # 双倍高度
    res = 300,     # 高分辨率
    bg = "white",
    type = "cairo")  # 更好的渲染引擎

par(mfrow = c(2, 3), mar = c(0, 0, 3, 0)) # 增加顶部边距 (第3个值) 留给标题
for (i in 1:num_plots) {
  idx <- diagnosis$indices[i]
  loss <- diagnosis$losses[i]
  plot_bad_case(test_ds, idx, loss, raw_model, device = "cpu")
}
dev.off()

