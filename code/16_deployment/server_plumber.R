#!/usr/bin/env Rscript
library(plumber)
source("utils.R")

device <- torch_device("cpu")
model <- model_resnet18(pretrained = TRUE)
model$eval() # 记得设为 eval 模式

target_conv <- model$layer4[[2]]$children$conv2
spy_layer <- SpyModule(target_conv)
model$layer4[[2]]$add_module("conv2", spy_layer)

# 辅助函数 (获取路径)
get_image_path <- function(img_obj) {
  if (!is.null(img_obj$datapath)) return(img_obj$datapath)
  tmp <- tempfile(fileext = ".jpg")
  writeBin(img_obj[[1]], tmp)
  return(tmp)
}

# --- 路由定义 (带注释) ---

#* @apiTitle Grad-CAM Explainer
#* @apiDescription 上传图片，返回 JSON 结果

#* 上传图片并解释
#* @param image:file 上传图片文件
#* @post /explain
#* @serializer json
function(req, image) {
  img_path <- get_image_path(image)
  tryCatch({
    generate_gradcam(model, spy_layer, img_path, device)
  }, error = function(e) list(error = e$message))
}