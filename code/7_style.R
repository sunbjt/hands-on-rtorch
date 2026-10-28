library(torch)
library(torchvision)
library(magick)

# TODO: 替换为你本地的图片路径
content_path <- "/Users/liusizhe/Pictures/精选/3B4A9433.JPG" 
style_path <- "https://m.media-amazon.com/images/I/81rJ1aypRiL._AC_SL1500_.jpg"
#style_path <- "https://pic3.zhimg.com/v2-d494347f9dcced0ef68ec28cf86db693_r.jpg"
#style_path <- "https://www.budarts.com/upload/art_mid/91ada153f76f4ae7b407c92e78bbb37b.jpg"
#style_path <- "https://www.budarts.com/upload/art_mid/3f960a3e14bf401e94b972a0964b7ba6.jpg"
# ---------------------------------------------------------
# 1. 基础设置与设备选择
# ---------------------------------------------------------
device <- torch_device("mps")

# 定义 ImageNet 的标准化参数 (VGG 训练时的参数)
img_mean <- c(0.485, 0.456, 0.406)
img_std  <- c(0.229, 0.224, 0.225)

# ---------------------------------------------------------
# 2. 图像加载与预处理函数
# ---------------------------------------------------------
load_image <- function(path, max_size = 400) {
  img <- image_read(path)
  info <- image_info(img)
  
  # 缩放图像以加快计算速度
  scale <- max_size / max(info$width, info$height)
  new_width <- as.integer(info$width * scale)
  img <- image_scale(img, geometry_size_pixels(width = new_width))
  tensor <- transform_to_tensor(img)$unsqueeze(1)
  tensor <- transform_normalize(tensor, mean = img_mean, std = img_std)
  
  return(tensor$to(device = device))
}

# 图像反处理函数 (用于最终保存/显示图像)
deprocess_image <- function(tensor) {
  img <- tensor$cpu()$clone()$squeeze(1)
  
  # 逆标准化
  for (i in 1:3) {
    img[i] <- img[i] * img_std[i] + img_mean[i]
  }
  
  img <- img$clamp(0, 1) # 限制在 0-1 范围内
  return(img)
}

# ---------------------------------------------------------
# 3. 加载 VGG19 模型并提取特征
# ---------------------------------------------------------
# 加载预训练的 VGG19 模型，仅保留特征提取部分 (features)
vgg <- model_vgg19(pretrained = TRUE)$features$to(device = device)$eval()

# 冻结模型参数 (我们不需要更新模型，只需更新输入图像)
for (p in vgg$parameters) {
  p$requires_grad_(FALSE)
}

# VGG19 特定层的索引映射 (R 语言中是 1-based index)
get_features <- function(image, model) {
  layers <- list(
    "1"  = "conv1_1",
    "6"  = "conv2_1",
    "11" = "conv3_1",
    "20" = "conv4_1",
    "22" = "conv4_2", # 通常作为内容提取层
    "29" = "conv5_1"
  )
  
  features <- list()
  x <- image
  
  # 遍历模型层
  for (i in 1:length(model)) {
    x <- model[[i]](x)
    idx <- as.character(i)
    if (idx %in% names(layers)) {
      features[[layers[[idx]]]] <- x
    }
  }
  return(features)
}

# ---------------------------------------------------------
# 4. 定义格拉姆矩阵 (Gram Matrix) 
# ---------------------------------------------------------
gram_matrix <- function(tensor) {
  b <- tensor$size(1)
  c <- tensor$size(2)
  h <- tensor$size(3)
  w <- tensor$size(4)
  
  # 将特征图展平为 (channels, height * width)
  features <- tensor$view(c(b * c, h * w))
  
  # 计算矩阵乘法
  G <- torch_mm(features, features$t())
  
  # 归一化，防止值过大导致梯度爆炸
  return(G / (b * c * h * w))
}

# ---------------------------------------------------------
# 5. 执行风格迁移
# ---------------------------------------------------------
content_img <- load_image(content_path)
style_img <- load_image(style_path)

# 初始化目标图像为内容图像（这比从随机噪声开始收敛得更快）
target_img <- content_img$clone()
target_img$requires_grad_(TRUE)

# 提取静态的特征和风格矩阵
content_features <- get_features(content_img, vgg)
style_features <- get_features(style_img, vgg)
style_grams <- lapply(style_features, gram_matrix)

# 风格层权重 (早期层捕捉细节纹理，晚期层捕捉宏观风格)
style_weights <- list(
  "conv1_1" = 1.0,  # 极其重要：捕捉细线条
  "conv2_1" = 0.8,  # 重要：捕捉稍微粗一点的轮廓
  "conv3_1" = 0.3,
  "conv4_1" = 0.1,  # 降低：简笔画没有复杂的宏观结构
  "conv5_1" = 0.0   # 甚至可以直接设为0
)

# 超参数
content_weight <- 0.1
style_weight <- 1e8
epochs <- 500
learning_rate <- 0.02

# 使用 Adam 优化器，注意我们优化的参数是 `target_img`
optimizer <- optim_adam(list(target_img), lr = learning_rate)

cat("Starting training...\n")

for (epoch in 1:epochs) {
  
  # 前向传播提取目标图像特征
  target_features <- get_features(target_img, vgg)
  
  # 1. 计算内容损失 (MSE Loss)
  content_loss <- nnf_mse_loss(
    target_features[["conv4_2"]], 
    content_features[["conv4_2"]]
  )
  
  # 2. 计算风格损失
  style_loss <- 0
  for (layer in names(style_weights)) {
    target_feature <- target_features[[layer]]
    target_gram <- gram_matrix(target_feature)
    style_gram <- style_grams[[layer]]
    
    layer_style_loss <- style_weights[[layer]] * nnf_mse_loss(target_gram, style_gram)
    style_loss <- style_loss + layer_style_loss
  }
  
  # 3. 总损失
  total_loss <- content_weight * content_loss + style_weight * style_loss
  
  # 4. 反向传播与优化
  optimizer$zero_grad()
  total_loss$backward()
  optimizer$step()
  
  # 每 50 步打印一次日志
  if (epoch %% 50 == 0) {
    cat(sprintf("Epoch [%d/%d], Total Loss: %.4f, Content: %.4f, Style: %.4f\n", 
                epoch, epochs, total_loss$item(), content_loss$item(), 10000*style_loss$item()))
  }
}

# ---------------------------------------------------------
# 6. 保存结果并生成三图对比 (内容 + 风格 + 结果)
# ---------------------------------------------------------
# 定义输出文件名
output_filename <- "output_styled_image.jpg"

# --- A. 将生成的 Tensor 转换为 Magick 图像对象 ---
final_tensor <- deprocess_image(target_img)
final_array <- as_array(final_tensor)

img_array <- aperm(final_array, c(2, 3, 1))
img_raster <- as.raster(img_array)
img_styled_magick <- image_read(img_raster)

# --- B. 保存单张风格化结果 ---
image_write(img_styled_magick, path = output_filename)
cat(sprintf("Done! Styled image saved as '%s'\n", output_filename))

# --- C. 生成并显示三联对比图 ---
# 1. 读取原始内容图像和风格图像 (路径在前面已经定义过)
img_content_magick <- image_read(content_path)
img_style_magick <- image_read(style_path)

# 2. 统一高度：为了拼接美观，将原图和风格图缩放到与生成图高度一致
gen_height <- image_info(img_styled_magick)$height
img_content_resized <- image_scale(img_content_magick, geometry_size_pixels(height = gen_height))
img_style_resized <- image_scale(img_style_magick, geometry_size_pixels(height = gen_height))

# 3. 添加标签 (字号设为 20，黑底白字，放在左上角)
img_content_labeled <- image_annotate(img_content_resized, " Content ", 
                                      size = 20, color = "white", boxcolor = "black", location = "+10+10")
img_styled_labeled <- image_annotate(img_styled_magick, " Result ", 
                                     size = 20, color = "white", boxcolor = "black", location = "+10+10")

# 4. 水平拼接图像：内容图 -> 风格图 -> 生成结果
combined_img <- image_append(c(img_content_labeled, img_styled_labeled), stack = FALSE)

# 5. 在 RStudio 显示结果
print(combined_img)

# 6. (可选) 将这张完美的三联图保存下来，直接用于书稿排版！
# image_write(combined_img, "comparison_three_panels.jpg")

