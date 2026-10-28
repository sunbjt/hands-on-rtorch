# ==============================================================================
# 第 13 章实战：Grad-CAM 可视化
# ==============================================================================

# 1. 加载必要的库
# ------------------------------------------------------------------------------
library(torch)
library(torchvision)
library(magick)

device <- if (backends_mps_is_available()) {
  torch_device("mps") # Mac M1/M2/M3
} else if (cuda_is_available()) {
  torch_device("cuda") # NVIDIA GPU
} else {
  torch_device("cpu")
}
# 2. 准备数据：下载一张测试图片
# ------------------------------------------------------------------------------
# 为了演示方便，我们从网络下载一张金毛猎犬的图片到临时目录
img_url <- "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Golden_Retriever_with_tennis_ball.jpg/320px-Golden_Retriever_with_tennis_ball.jpg"
original_img <- image_read(img_url)

transform <- function(img) {
  img %>%
    transform_to_tensor() %>%
    transform_resize(c(224, 224)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406),
                        std = c(0.229, 0.224, 0.225)) %>%
    torch_unsqueeze(1)
}
input_tensor <- transform(original_img)$to(device = device)

# ------------------------------------------------------------------------------
# 2. 定义“间谍模块” (Spy Module) —— 本书核心技巧
# ------------------------------------------------------------------------------
# 这个模块像一个“中间人”，在传递数据的同时，偷偷把数据复印一份存起来。

SpyModule <- nn_module(
  "SpyModule",
  initialize = function(original_layer) {
    self$layer <- original_layer
    self$stored_features <- NULL # 用于存储特征图
  },
  forward = function(x) {
    # 1. 运行原始层
    out <- self$layer(x)
    # 2. 偷偷保存输出 (特征图)
    self$stored_features <- out
    return(out)
  }
)

# ------------------------------------------------------------------------------
# 3. 实施手术：替换模型中的层
# ------------------------------------------------------------------------------
model <- model_resnet18(pretrained = TRUE)
model$eval()
model$to(device = device)

# 定位到目标卷积层：ResNet18 -> Layer4 -> Block 2 -> Conv2
# 注意：我们这里使用 children$conv2 来确保获取对象
target_conv <- model$layer4[[2]]$children$conv2


# 创建间谍层，把原始卷积层包进去
spy_layer <- SpyModule(target_conv)

# 将间谍层塞回模型，替换掉原来的 conv2
# add_module 会用新名字覆盖旧模块，实现“狸猫换太子”
model$layer4[[2]]$add_module("conv2", spy_layer)

cat("模型中的 conv2 已被替换为 SpyModule。\n")

# ------------------------------------------------------------------------------
# 4. 前向与反向传播 (捕捉梯度)
# ------------------------------------------------------------------------------
# (1) 前向传播
output <- model(input_tensor)
pred_idx <- torch_argmax(output, dim = 2)$item()
cat("预测类别索引:", pred_idx, "\n")

spy_layer$stored_features$retain_grad() # 开启录像
model$zero_grad()
one_hot_output <- torch_zeros_like(output)
one_hot_output[1, pred_idx] <- 1
output$backward(gradient = one_hot_output)

features <- spy_layer$stored_features
gradients <- spy_layer$stored_features$grad

# (2) 计算 Grad-CAM (优化归一化逻辑)
weights <- torch_mean(gradients, dim = c(3, 4), keepdim = TRUE)
cam <- torch_sum(weights * features, dim = 2, keepdim = TRUE)

# 先转为 R 的 numeric 向量，再求 range
cam_vec <- as.numeric(cam$detach()$cpu())
raw_range <- range(cam_vec)

# 应用 ReLU：负贡献（抑制）清零
cam <- nnf_relu(cam)

# 这样保留了绝对强度的概念。如果整体响应弱，图就会淡，而不是强行变红。
cam <- cam / (torch_max(cam) + 1e-8)

# ------------------------------------------------------------------------------
# 6. 可视化
# ------------------------------------------------------------------------------
original_resized <- image_resize(original_img, "224x224!")
cam_smooth <- nnf_interpolate(
  cam,
  size = c(224, 224),
  mode = "bicubic",
  align_corners = FALSE
)

# 2. 提取数据
heatmap_matrix <- as.matrix(cam_smooth[1, 1, , ]$detach()$cpu())

# 3. 再次应用阈值过滤 (去除底噪)
heatmap_matrix[heatmap_matrix < 0.1] <- 0

# 4. 绘图准备
par(mar = c(0, 0, 2, 0))
plot.new()
plot.window(
  xlim = c(0, 1),
  ylim = c(0, 1),
  xaxs = "i",
  yaxs = "i"
)

# (A) 画底图
rasterImage(as.raster(original_resized), 0, 0, 1, 1)

# (B) 制作“彩虹热力图”调色板 (Blue -> Cyan -> Yellow -> Red)
# alpha 通道：从完全透明(00)过渡到半透明(99/AA)
jet_colors <- colorRampPalette(c(
  "#0000FF00",
  "#00FFFF44",
  "#FFFF0088",
  "#FF0000AA"  # 红 (较明显) - 高关注
), alpha = TRUE)(100)

# (C) 叠加绘制
image(t(heatmap_matrix[nrow(heatmap_matrix):1, ]),
      col = jet_colors,
      axes = FALSE,
      add = TRUE)

title(main = "ResNet18 Focus Area (Smoothed)-Layer4")

