# 为了节省 QuerySuggestGPT 的训练时间，所以在已经训练好的模型上迭代
# 所有的配置都要重现：
# 1. 依赖的包环境
# 2. 自定义函数，数据预处理、模型定义、预测函数 suggest_next 等
# 3. 加载数据，加载已经保存的模型

library(torch)
library(tidyverse)
library(tidytext)
library(fs)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

setwd("/Users/liusizhe/data")

load_imdb_data <- function(root_path) {
  cat(sprintf("正在扫描目录: %s ...\n", root_path))
  pos_files <- dir_ls(file.path(root_path, "pos"), glob = "*.txt")
  neg_files <- dir_ls(file.path(root_path, "neg"), glob = "*.txt")
  pos_texts <- map_chr(pos_files, read_file)
  neg_texts <- map_chr(neg_files, read_file)
  tibble(text = c(pos_texts, neg_texts))
}

TidyTextPipeline <- R6::R6Class( # 文本处理管道
  "TidyTextPipeline",
  public = list(
    token_to_idx = NULL,
    idx_to_token = NULL,
    max_len = NULL,
    pad_idx = NULL,
    unk_idx = NULL,
    
    initialize = function(texts, max_vocab = 10000, max_len = 64) {
      self$max_len <- max_len
      
      token_counts <- tibble(text = texts) %>% 
        unnest_tokens(word, text) %>% 
        count(word, sort = TRUE) %>% 
        head(max_vocab)
      
      special_tokens <- c("<PAD>", "<UNK>", "<CLS>")
      full_vocab <- c(special_tokens, token_counts$word)
      self$idx_to_token <- full_vocab
      self$token_to_idx <- set_names(seq_along(full_vocab), full_vocab)
      self$pad_idx <- self$token_to_idx["<PAD>"]
      self$unk_idx <- self$token_to_idx["<UNK>"]
      
      cat("词汇表构建完成。大小:", length(full_vocab), "\n")
    },
    
    encode = function(text) {
      tokens <- tibble(text = text) %>%
        unnest_tokens(word, text) %>% pull(word)
      indices <- self$token_to_idx[tokens]
      indices[is.na(indices)] <- self$unk_idx
      c(self$token_to_idx["<CLS>"], indices) # 加上 CLS
    },
    
    decode = function(indices) {
      tokens <- self$idx_to_token[as.integer(indices)]
      valid <- tokens[!tokens %in% c("<PAD>", "<UNK>", "<CLS>")]
      paste(valid, collapse = " ")
    }
  )
)

ImdbGenDataset <- dataset(
  "ImdbGenDataset",
  initialize = function(df, pipeline) {
    self$max_l <- pipeline$max_len
    target_len <- self$max_l + 1 # 输入+输出
    pad <- pipeline$pad_idx
    unk <- pipeline$unk_idx
    vocab_map <- pipeline$token_to_idx
    # 使用 map 批量处理，比在 getitem 里处理快得多
    token_list <- map(df$text, function(txt) {
      words <- unlist(strsplit(tolower(txt), "\\s+"))
      idxs <- vocab_map[words]
      idxs[is.na(idxs)] <- unk
      full_seq <- c(vocab_map["<CLS>"], idxs)
      
      len <- length(full_seq)
      if (len >= target_len) {
        full_seq[1:target_len]
      } else {
        c(full_seq, rep(pad, target_len - len))
      }
    })
    
    flat_indices <- unlist(token_list)
    n_samples <- length(token_list)
    self$data_tensor <- torch_tensor(flat_indices, dtype = torch_long())$view(c(n_samples, target_len))
    cat("数据集准备就绪。样本数:", n_samples, "\n")
  },
  
  .getitem = function(i) {
    full_seq <- self$data_tensor[i, ]
    list(
      x = full_seq[1:self$max_l],
      y = full_seq[2:(self$max_l + 1)]
    )
  },
  
  .length = function() self$data_tensor$size(1)
)

# --- 4. 模型架构 (Buffers & Dropout) ---
QuerySuggestGPT <- nn_module(
  "QuerySuggestGPT",
  initialize = function(
    vocab_size,
    d_model = 128,
    nhead = 4,
    num_layers = 2,
    max_len = 32,
    pad_idx = 1,
    dropout = 0.1) {
    self$embedding <- nn_embedding(vocab_size, d_model, padding_idx = pad_idx)
    self$pos_embedding <- nn_embedding(max_len, d_model)
    self$dropout <- nn_dropout(dropout)
    
    self$register_buffer("pos_idx", torch_arange(1, max_len, dtype = torch_long()))
    mask <- torch_full(c(max_len, max_len), -Inf)
    mask <- torch_triu(mask, diagonal = 1)
    self$register_buffer("causal_mask", mask)
    
    encoder_layer <- nn_transformer_encoder_layer(
      d_model = d_model, 
      nhead = nhead, 
      dim_feedforward = d_model*4, 
      dropout = dropout,
      batch_first = TRUE
    )
    self$transformer <- nn_transformer_encoder(encoder_layer, num_layers = num_layers)
    self$fc_out <- nn_linear(d_model, vocab_size)
  },
  
  forward = function(x) {
    seq_len <- x$size(2)
    pos <- self$pos_idx[1:seq_len]$unsqueeze(1)
    mask <- self$causal_mask[1:seq_len, 1:seq_len]
    emb <- self$embedding(x) + self$pos_embedding(pos)
    emb <- self$dropout(emb)
    out <- self$transformer(emb, mask = mask)
    logits <- self$fc_out(out)
    # 将 [Batch, Seq, Vocab] -> [Batch, Vocab, Seq]
    logits <- logits$permute(c(1, 3, 2))
    return(logits)
  }
)

suggest_next <- function(model, pipeline, text, max_new = 20, temp = 1.0) {
  model$eval()
  dev <- model$parameters[[1]]$device
  
  idxs <- pipeline$encode(text)
  idxs <- idxs[idxs != pipeline$pad_idx]
  ctx <- torch_tensor(idxs, dtype = torch_long(), device = dev)$unsqueeze(1)
  
  cat(sprintf("\nInput: '%s' (Temp=%.1f)\nSuggest: ", text, temp))
  
  for (i in 1:max_new) {
    with_no_grad({
      w <- pipeline$max_len
      if (ctx$size(2) > w) ctx_cond <- ctx[, (ctx$size(2) - w + 1):ctx$size(2)]
      else ctx_cond <- ctx
      
      logits <- model(ctx_cond)  # 形状 [Batch, Vocab, Seq]
      logits <- logits$transpose(2, 3)  # [Batch, Seq, Vocab]
      last_logits <- logits[, logits$size(2), ]
      
      if (temp <= 0) {
        next_idx <- torch_argmax(last_logits, dim = -1)
      } else {
        probs <- nnf_softmax(last_logits / temp, dim = -1)
        next_idx <- torch_multinomial(probs, 1)
      }
    })
    
    word <- pipeline$idx_to_token[as.integer(next_idx$cpu())]
    if (word == "<PAD>" || is.na(word)) break
    
    cat(word, "")
    ctx <- torch_cat(list(ctx, next_idx$view(c(1,1))), dim = 2)
  }
  cat("\n")
}

# --- 5. 训练执行 ---
base_path <- "/Users/liusizhe/data/imdb/aclImdb/train/" 
df <- load_imdb_data(base_path)
pipeline <- TidyTextPipeline$new(df$text, max_vocab = 5000, max_len = 32)
pad_int <- as.integer(pipeline$pad_idx)



# 加载 Pipeline
pipeline <- read_rds("text_pipeline.rds")
# 第一个实例：作为纯净的“对照组”
loaded_base_model <- QuerySuggestGPT(
  vocab_size = length(pipeline$idx_to_token),
  d_model = 256, nhead = 4, num_layers = 2, max_len = 32, pad_idx = pad_int
)
loaded_base_model$load_state_dict(torch_load("imdb_gpt_base.pt"))
loaded_base_model$eval() 

# 第二个实例：作为准备微调的“实验组”，这个模型将被送去接受 LoRA 手术
lora_host_model <- QuerySuggestGPT(
  vocab_size = length(pipeline$idx_to_token),
  d_model = 256, nhead = 4, num_layers = 2, max_len = 32, pad_idx = pad_int
)
lora_host_model$load_state_dict(torch_load("imdb_gpt_base.pt"))
lora_host_model$eval()

cat("基础模型已从本地加载完毕，可以开始 LoRA 微调。\n")

## ----------------------------------------
## 进入到 LoRA 环节
## ----------------------------------------

nn_lora_linear <- nn_module(
  "LoRALinear",
  
  # 直接接收原有的 layer 对象，不再接收数字
  initialize = function(base_layer, r = 32, lora_alpha = 16) {
    self$linear <- base_layer
    self$r <- r
    self$scaling <- lora_alpha / r  
    self$merged <- FALSE 
    
    # 1. 冻结主干网络
    self$linear$weight$requires_grad_(FALSE)
    if (!is.null(self$linear$bias)) {
      self$linear$bias$requires_grad_(FALSE)
    }
    
    # 2. 自动从传入的层中提取维度和设备
    out_features <- self$linear$weight$size(1)
    in_features <- self$linear$weight$size(2)
    dev <- self$linear$weight$device
    
    # 3. 定义低秩矩阵 A 和 B
    self$lora_A <- nn_parameter(torch_empty(c(r, in_features), device = dev))
    self$lora_B <- nn_parameter(torch_empty(c(out_features, r), device = dev))
    
    self$reset_parameters()
  },
  
  reset_parameters = function() {
    nn_init_kaiming_uniform_(self$lora_A, a = sqrt(5))
    nn_init_zeros_(self$lora_B)
  },
  
  # 【新增】：权重合并机制
  merge = function() {
    if (self$merged) return()
    with_no_grad({
      delta_w <- torch_matmul(self$lora_B, self$lora_A) * self$scaling
      self$linear$weight$add_(delta_w)
    })
    self$merged <- TRUE
  },

  forward = function(x) {
    if (self$merged) {
      return(self$linear(x))
    }
    
    base_out <- self$linear(x)
    lora_out <- torch_matmul(x, self$lora_A$t())
    lora_out <- torch_matmul(lora_out, self$lora_B$t())
    
    return(base_out + lora_out * self$scaling)
  }
)


LoRASciFiGPT <- nn_module(
  "LoRASciFiGPT",
  
  initialize = function(pretrained_model, r = 8, lora_alpha = 16) {
    self$model <- pretrained_model
    
    # 1. 冻结全局主干参数
    for (p in self$model$parameters) {
      p$requires_grad_(FALSE)
    }
    
    # 2. 深度注入：遍历所有的 Transformer 层，替换 FFN
    num_layers <- length(self$model$transformer$layers)
    for (i in seq_len(num_layers)) {
      layer <- self$model$transformer$layers[[i]]
      
      # 替换 FFN 的第一层 (d_model -> d_ff)
      layer$linear1 <- nn_lora_linear(layer$linear1, r, lora_alpha)
      
      # 替换 FFN 的第二层 (d_ff -> d_model)
      layer$linear2 <- nn_lora_linear(layer$linear2, r, lora_alpha)
    }
    
    # 3. 浅层注入：替换输出层
    self$model$fc_out <- nn_lora_linear(self$model$fc_out, r, lora_alpha)
    
    cat(sprintf("已成功将 LoRA 注入到 %d 层 Transformer FFN 及输出层中。\n", num_layers))
  },
  
  # 全局合并接口：将所有层级的 A*B 压回主干
  merge_weights = function() {
    num_layers <- length(self$model$transformer$layers)
    
    # 合并 Transformer 层
    for (i in seq_len(num_layers)) {
      self$model$transformer$layers[[i]]$linear1$merge()
      self$model$transformer$layers[[i]]$linear2$merge()
    }
    
    # 合并输出层
    self$model$fc_out$merge()
    
    cat("所有 LoRA 权重 (Transformer FFN + Output) 已成功合并入主干网络。\n")
  },
  
  forward = function(x) {
    self$model(x)
  }
)

# 再次实例化，这次梯度会被重新激活！

# 打印参数量对比函数
print_trainable_params <- function(model) {
  trainable <- sum(purrr::map_int(model$parameters, function(p) {
    if (p$requires_grad) p$numel() else 0
  }))
  total <- sum(purrr::map_int(model$parameters, function(p) p$numel()))
  ratio <- if (total > 0) trainable / total * 100 else 0
  cat(sprintf("总参数: %d | 可训练参数: %d | 训练比例: %.3f%%\n", 
              total, trainable, ratio))
}

scifi_model_deep <- LoRASciFiGPT(lora_host_model, r = 8)
# 打印参数量验证
print_trainable_params(scifi_model_deep)

# 1. 构建特定领域的小样本数据集

raw_lines <- read_lines("/Users/liusizhe/data/scifi_synthetic.csv")
scifi_df <- tibble(text = raw_lines) %>% 
  filter(text != "")
scifi_ds <- ImdbGenDataset(scifi_df, pipeline)
scifi_dl <- dataloader(scifi_ds, batch_size = 32, shuffle = TRUE)


# 2. 使用 luz 进行参数高效微调 (仅更新 LoRA 层)
# 直接使用我们上面已经实例化好、并成功注入了 LoRA 的模型实例
# 1. 写一个极简的包装器，利用 R 的闭包特性捕获外部的 scifi_model_deep
LuzWrapper <- nn_module(
  "LuzWrapper",
  initialize = function() {
    # 直接赋值我们在外部已经成功注入 LoRA 的实例模型
    self$model <- scifi_model_deep 
  },
  forward = function(x) {
    self$model(x)
  }
)

# 2. 正常使用 luz 训练，因为不需要传任何超参数，彻底避开报错
fitted_scifi <- setup(
  module = LuzWrapper, # 传入这个干净的包装器类
  loss = nn_cross_entropy_loss(ignore_index = pad_int),
  optimizer = optim_adamw
) %>%
  set_opt_hparams(lr = 2e-3) %>% 
  fit(
    scifi_dl,
    epochs = 20,  # 跑 10 个 epoch
    accelerator = accelerator(),
    verbose = TRUE
  )

trained_wrapper <- fitted_scifi$model
# 从 Wrapper 中剥壳，取出我们真正微调好的 LoRA 模型实体
final_scifi_model <- trained_wrapper$model
# 对最终训练好的模型执行合并
final_scifi_model$merge_weights()


# 只会说 generic 废话的通用模型
suggest_next(loaded_base_model, pipeline, "this movie is", temp = 0.9, max_new = 30)
# 赛博朋克风格的新模型
suggest_next(final_scifi_model, pipeline, "this movie is", temp = 0.9, max_new = 30)
