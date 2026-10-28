#!/usr/bin/env Rscript
Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")

library(RestRserve)
library(torch)
library(torchvision)

source("utils.R")
device <- torch_device("cpu")
model <- torchvision::model_resnet18(pretrained = TRUE)
model$eval()

target_conv <- model$layer4[[2]]$conv2
spy_layer <- SpyModule(target_conv)
model$layer4[[2]]$add_module("conv2", spy_layer)

app <- Application$new(content_type = "application/json")

get_image_path <- function(request) {
  file_info <- request$files$image
  file_content <- NULL
  
  # 情况 A: contents 字段直接存在
  if (!is.null(file_info$contents)) {
    file_content <- file_info$contents
  } 
  # 情况 B: 从 body 切片
  else if (!is.null(file_info$offset) && !is.null(file_info$length)) {
    if (!is.raw(request$body)) {
      stop("File has offset pointers but request$body is not raw bytes.")
    }
    
    # 日志证明 offset 已经是 1-based 的起始位置了，不需要 +1
    start_pos <- file_info$offset
    end_pos <- file_info$offset + file_info$length - 1
    
    file_content <- request$body[start_pos:end_pos]
  } 
  else {
    stop("Cannot extract file content.")
  }
  
  # 保存到临时文件
  ext <- tools::file_ext(file_info$filename)
  if (ext == "") ext <- "jpg"
  tmp <- tempfile(fileext = paste0(".", ext))
  
  writeBin(file_content, tmp)
  return(tmp)
}

app$add_post(
  path = "/explain",
  FUN = function(request, response) {
    tryCatch({
      img_path <- get_image_path(request)
      result <- generate_gradcam(model, spy_layer, img_path, device)
      response$set_body(result)
    }, error = function(e) {
      message("[Request Error] ", e$message)
      response$set_body(list(error = e$message))
      response$set_status_code(400)
    })
  }
)

message("--- Starting Server on 0.0.0.0:8080 ---")
backend <- BackendRserve$new(workers = 1)
backend$start(app, http_port = 8080)