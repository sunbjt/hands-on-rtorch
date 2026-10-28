library(torch)
library(torchvision)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

pic_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>% #像素值从 [0, 255] 缩放到 [0, 1]
    transform_resize(size = c(64, 64)) %>%  # 放大到 64x64
    transform_normalize(mean = c(0.485, 0.456, 0.406),
                        std = c(0.229, 0.224, 0.225))
}

label_transforms <- function(y) {
  torch_tensor(y, dtype = torch_long())$squeeze()
}

# 下载完数据集后，将 batches.meta.txt 最后一行删除
# 多了一个空行，R 会认为多一个类别 ""
data_root <- "~/data/cifar10" 
train_ds <- cifar10_dataset(
  root = data_root,
  train = TRUE,
  download = TRUE,
  transform = pic_transforms,
  target_transform = label_transforms
)

test_ds <- cifar10_dataset(
  root = data_root,
  train = FALSE,
  transform = pic_transforms,
  target_transform = label_transforms
)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 64)

# 仅仅冻结 Head
MobileNetV2_HeadOnly <- nn_module(
  "MobileNetV2_HeadOnly",
  
  initialize = function(num_classes = 10) {
    full_model <- model_mobilenet_v2(pretrained = TRUE)
    self$features <- full_model$features
    
    for (p in self$features$parameters) {
      p$requires_grad <- FALSE
    }
    
    self$dropout <- nn_dropout(p = 0.2)
    self$fc <- nn_linear(in_features = 1280, out_features = num_classes)
  },
  
  forward = function(x) {
    x %>%
      self$features() %>%
      # 全局平均池化: (B, 1280, H, W) -> (B, 1280, 1, 1)
      nnf_adaptive_avg_pool2d(output_size = c(1, 1)) %>%
      torch_flatten(start_dim = 2) %>%
      self$dropout() %>%
      self$fc()
  }
)

# --- 训练 ---
fitted_frozen <- MobileNetV2_HeadOnly %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams(num_classes = 10) %>%
  set_opt_hparams(lr = 1e-3) %>% # 初始学习率可以稍大
  fit(
    train_dl,
    epochs = 3, # 少量 epoch 即可收敛
    valid_data = test_dl
  )

print(fitted_frozen)

evaluation <- fitted_frozen %>% evaluate(data = test_dl)
get_metrics(evaluation)
test_pred <- torch_argmax(predict(fitted_frozen, newdata = test_dl), dim = 2)
table(as.numeric(test_pred), as.numeric(test_ds$y))

# ---------------------------------- #
# --- Fine-tuning ---
# ---------------------------------- #
MobileNetV2_Hybrid <- nn_module(
  "MobileNetV2_Hybrid",
  
  initialize = function(num_classes = 10, unfreeze_blocks = 0) {
    # unfreeze_blocks: 从后往前数，需要解冻的块数 (0-18)
    full_model <- model_mobilenet_v2(pretrained = TRUE)
    self$features <- full_model$features
    
    for (p in self$features$parameters) p$requires_grad <- FALSE
    
    if (unfreeze_blocks > 0) {
      total_blocks <- length(self$features)
      start_idx <- max(1, total_blocks - unfreeze_blocks + 1)
      for (i in start_idx:total_blocks) {
        for (p in self$features[[i]]$parameters) p$requires_grad <- TRUE
      }
    }
    
    self$dropout <- nn_dropout(p = 0.2)
    self$fc <- nn_linear(in_features = 1280, out_features = num_classes)
  },
  
  extract_features = function(x) {  # 混合模型的提取特征辅助方法 
    x %>% 
      self$features() %>% 
      # 无视输入特征图的空间尺寸，全局平均池化 GAP
      nnf_adaptive_avg_pool2d(output_size = c(1, 1)) %>% 
      torch_flatten(start_dim = 2) # 展平操作
  },
  
  forward = function(x) {
    x %>% 
      self$extract_features() %>%  # 复用上面的方法
      self$dropout() %>% 
      self$fc()
  }
)

fitted_FT <- MobileNetV2_Hybrid %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_hparams(num_classes = 10, unfreeze_blocks = 1) %>%
  set_opt_hparams(lr = 1e-3) %>%
  fit(
    train_dl,
    epochs = 3,
    valid_data = test_dl,
    accelerator = accelerator()
  )

evaluation_FT <- fitted_FT %>% evaluate(data = test_dl)
get_metrics(evaluation_FT)

# --- 混合模型 CNN + xgboost ---

library(xgboost)

# 1. 定义特征提取函数
extract_features <- function(luz_obj, dataloader) {
  model <- luz_obj$model
  model$to(device = "cpu") 
  model$eval()
  
  features_list <- list()
  labels_list   <- list()
  
  with_no_grad({
    coro::loop(for (batch in dataloader) {
      x <- batch$x$to(device = "cpu")
      y <- batch$y$to(device = "cpu") 
      feat <- model$extract_features(x)
      features_list[[length(features_list) + 1]] <- as.matrix(feat)
      labels_list[[length(labels_list) + 1]]     <- as.integer(y)
    })
  })

  list(
    data  = do.call(rbind, features_list), # 特征矩阵
    label = unlist(labels_list)            # 标签向量
  )
}

# 2. 提取测试集特征 (仅作演示，实际应用应包含训练集)
train_extracted <- extract_features(fitted_FT, train_dl)
test_extracted <- extract_features(fitted_FT, test_dl)


# 3. 训练 XGBoost
# 注意：torch 的 label 是 1-based (1-10)，xgboost 需要 0-based (0-9)
library(xgboost)
dtrain <- xgb.DMatrix(
  data = train_extracted$data, 
  label = train_extracted$label - 1
)

dtest <- xgb.DMatrix(
  data = test_extracted$data, 
  label = test_extracted$label - 1
)

evals_list <- list(train = dtrain, test = dtest)

# 3. 参数设定
params <- list(
  objective = "multi:softmax",
  num_class = 10,
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 3,
  gamma = 0.1,
  subsample = 0.7,
  colsample_bytree = 0.7
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  evals = evals_list,
  early_stopping_rounds = 20,
  print_every_n = 20,
  verbose = 1
)

# 计算准确率
preds <- predict(xgb_model, dtest)
accuracy <- mean(preds == (test_extracted$label - 1))
cat(sprintf("CNN + XGBoost Hybrid Accuracy: %.2f%%\n", accuracy * 100))

