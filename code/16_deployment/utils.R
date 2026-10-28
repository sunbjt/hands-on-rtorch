# utils.R
library(torch)
library(torchvision)
library(magick)
library(jsonlite)
library(base64enc) # 需要这个包来处理图片传输

# 1. 定义 SpyModule (保持你的原代码不变)
SpyModule <- nn_module(
  "SpyModule",
  initialize = function(original_layer) {
    self$layer <- original_layer
    self$stored_features <- NULL
  },
  forward = function(x) {
    out <- self$layer(x)
    self$stored_features <- out
    return(out)
  }
)

labels <- jsonlite::fromJSON('label_list.json')
# 2. 图像预处理函数
transform_img <- function(magick_img) {
  magick_img %>%
    transform_to_tensor() %>%
    transform_resize(c(224, 224)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406),
                        std = c(0.229, 0.224, 0.225)) %>%
    torch_unsqueeze(1)
}

# 3. 核心 Grad-CAM 函数
generate_gradcam <- function(model, spy_layer, image_path, device) {
  
  # 读取图片
  original_img <- image_read(image_path)
  input_tensor <- transform_img(original_img)$to(device = device)
  
  # --- 前向传播 ---
  output <- model(input_tensor)
  pred_idx <- torch_argmax(output, dim = 2)$item()
  
  # --- 反向传播 ---
  spy_layer$stored_features$retain_grad()
  model$zero_grad()
  one_hot_output <- torch_zeros_like(output)
  one_hot_output[1, pred_idx] <- 1
  output$backward(gradient = one_hot_output)
  
  features <- spy_layer$stored_features
  gradients <- spy_layer$stored_features$grad
  
  # --- 计算 CAM (你的逻辑) ---
  weights <- torch_mean(gradients, dim = c(3, 4), keepdim = TRUE)
  cam <- torch_sum(weights * features, dim = 2, keepdim = TRUE)
  cam <- nnf_relu(cam)
  cam <- cam / (torch_max(cam) + 1e-8)
  
  # --- 图像处理准备 ---
  original_resized <- image_resize(original_img, "224x224!")
  cam_smooth <- nnf_interpolate(
    cam, size = c(224, 224), mode = "bicubic", align_corners = FALSE
  )
  heatmap_matrix <- as.matrix(cam_smooth[1, 1, , ]$detach()$cpu())
  heatmap_matrix[heatmap_matrix < 0.1] <- 0 # 阈值过滤
  
  # --- 关键修改：绘图并保存到内存/临时文件 ---
  # 创建一个临时文件来保存绘图结果
  tmp_file <- tempfile(fileext = ".png")
  png(tmp_file, width = 224, height = 224) # 开启绘图设备
  
  par(mar = c(0, 0, 0, 0)) # 移除边距
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")
  
  # 画底图
  rasterImage(as.raster(original_resized), 0, 0, 1, 1)
  
  # 画热力图
  jet_colors <- colorRampPalette(c("#0000FF00", "#00FFFF44", "#FFFF0088", "#FF0000AA"), alpha = TRUE)(100)
  image(t(heatmap_matrix[nrow(heatmap_matrix):1, ]), 
        col = jet_colors, axes = FALSE, add = TRUE)
  
  dev.off() # 关闭设备，此时图片已写入 tmp_file
  
  # 读取二进制文件并转为 Base64
  # Base64 可以在 JSON 中直接传输图片
  img_base64 <- base64encode(tmp_file)
  unlink(tmp_file) # 删除临时文件
  
  # 返回预测结果和图片字符串
  list(
    class_idx = labels[pred_idx],
    image_base64 = img_base64 
  )
}
