# 数据下载地址
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# 解压至 imdb 子目录
library(torch)
library(tidyverse)
library(tidytext)
library(fs)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

set.seed(42)
torch_manual_seed(42)

# 输入文件夹路径，自动读取 pos/neg 并生成标签，返回 tibble
load_imdb_data <- function(root_path) {
  cat(sprintf("正在扫描目录: %s ...\n", root_path))
  
  pos_files <- dir_ls(file.path(root_path, "pos"), glob = "*.txt")
  neg_files <- dir_ls(file.path(root_path, "neg"), glob = "*.txt")
  pos_texts <- map_chr(pos_files, read_file)
  neg_texts <- map_chr(neg_files, read_file)
  
  tibble(
    text = c(pos_texts, neg_texts),
    label = c(rep(1, length(pos_texts)), rep(0, length(neg_texts)))
  )
}

# 文本预处理类定义
TidyTextPipeline <- R6::R6Class(
  "TidyTextPipeline",
  public = list(
    token_to_idx = NULL,
    idx_to_token = NULL,
    max_len = NULL,
    
    initialize = function(texts,
                          max_vocab = 20000,
                          max_len = 256) {
      self$max_len <- max_len
      token_df <- tibble(text = texts) %>% unnest_tokens(word, text)
      
      top_tokens <- token_df %>% count(word, sort = TRUE) %>%
        head(max_vocab) %>% pull(word)
      
      full_vocab <- c("<PAD>", "<UNK>", "<CLS>", top_tokens)
      self$idx_to_token <- full_vocab
      self$token_to_idx <- set_names(seq_along(full_vocab), full_vocab)
      cat("词汇表构建完成。大小:", length(full_vocab), "\n")
    },
    
    encode = function(text) {
      tokens <- tibble(text = text) %>% unnest_tokens(word, text) %>% pull(word)
      indices <- self$token_to_idx[tokens]
      indices[is.na(indices)] <- self$token_to_idx["<UNK>"]
      cls_idx <- self$token_to_idx["<CLS>"]
      indices <- c(cls_idx, indices)
      
      n <- length(indices)
      pad_idx <- self$token_to_idx["<PAD>"]
      if (n >= self$max_len)
        indices[1:self$max_len]
      else
        c(indices, rep(pad_idx, self$max_len - n))
    },
    
    decode = function(indices) {
      indices <- as.integer(indices)
      tokens <- self$idx_to_token[indices]
      filter_mask <- !is.na(tokens) & tokens != "<PAD>" & tokens != "<CLS>"
      tokens <- tokens[filter_mask]
      paste(tokens, collapse = " ")
    },
    
    vocab_size = function()
      length(self$idx_to_token)
  )
)

SimpleIMDbDataset <- dataset(
  "SimpleIMDbDataset",
  initialize = function(df, pipeline) {
    self$df <- df
    self$pipeline <- pipeline
  },
  .getitem = function(i) {
    list(
      x = torch_tensor(self$pipeline$encode(self$df$text[i]), dtype = torch_long()),
      y = torch_tensor(self$df$label[i], dtype = torch_float())
    )
  },
  .length = function()
    nrow(self$df)
)

# 用原始论文的 PE 构造
positional_encoding <- nn_module(
  "positional_encoding",
  
  initialize = function(d_model,
                        dropout = 0.1,
                        max_len = 5000) {
    self$dropout <- nn_dropout(p = dropout)
    pe <- torch_zeros(max_len, d_model) # 创建位置矩阵 (pe)
    position <- torch_arange(1, max_len)$unsqueeze(2)
    div_term <- torch_exp(
      torch_arange(0,d_model-1,2)*(-log(10000.0)/d_model)
    )
    # 赋值给偶数列 (R索引 1, 3, 5...)
    pe[,seq(1,d_model,by = 2)] <- torch_sin(position*div_term)
    # 赋值给奇数列 (R索引 2, 4, 6...)
    pe[,seq(2,d_model,by = 2)] <- torch_cos(position*div_term)
    # 注册为 buffer
    self$register_buffer("pe", pe$unsqueeze(1))
  },
  
  forward = function(x) {
    x <- x + self$pe[, 1:x$size(2), ]
    return(self$dropout(x))
  }
)

IMDbTransformer <- nn_module(
  "IMDbTransformer",
  initialize = function(vocab_size, d_model, nhead, num_layers, output_dim = 1) {
    
    # 1. 词嵌入 (padding_idx=1 对应 <PAD>)
    self$embedding <- nn_embedding(vocab_size, d_model, padding_idx = 1)
    self$pos_encoder <- positional_encoding(d_model)
    encoder_layer <- nn_transformer_encoder_layer(
      d_model = d_model,
      nhead = nhead,
      batch_first = TRUE)
    self$transformer_encoder <- nn_transformer_encoder(
      encoder_layer, num_layers = num_layers)
    
    # 4. 分类头 (只接受 [CLS] 向量)
    self$fc <- nn_linear(d_model, output_dim)
  },
  
  forward = function(x) {
    src_key_padding_mask <- (x == 1) 
    
    # Mask B: 给 Mean Pooling 用的 (1.0 表示保留, 0.0 表示是垃圾)
    # 形状转换: [Batch, Seq] -> [Batch, Seq, 1]
    # 注意要转成 float 类型以便后面做乘法
    pooling_mask <- (x != 1)$to(dtype = torch_float())$unsqueeze(3)
    
    # --- 2. Embedding & Encoding ---
    # 此时 x 被覆盖为 Embeddings: [Batch, Seq, Dim]
    x <- self$embedding(x) * sqrt(self$embedding$embedding_dim)
    x <- self$pos_encoder(x)
    
    # --- 3. Transformer ---
    # out: [Batch, Seq, Dim]
    out <- self$transformer_encoder(x, src_key_padding_mask = src_key_padding_mask)
    
    # --- 4. Mean Pooling (使用刚才存好的 pooling_mask) ---
    # 广播乘法: [Batch, Seq, Dim] * [Batch, Seq, 1]
    out_masked <- out * pooling_mask
    
    # 求和并归一化
    sum_out <- torch_sum(out_masked, dim = 2)          # [Batch, Dim]
    lens <- torch_sum(pooling_mask, dim = 2)           # [Batch, 1]
    mean_pool <- sum_out / torch_clamp(lens, min = 1)  # 防止除以 0
    
    # Classifier
    self$fc(mean_pool)
  }
)

# A. 加载原始数据 
base_path <- "/Users/liusizhe/data/imdb/aclImdb"
train_df <- load_imdb_data(file.path(base_path, "train"))
test_df  <- load_imdb_data(file.path(base_path, "test"))

# B. 构建 Pipeline (只用训练集)
pipeline <- TidyTextPipeline$new(texts = train_df$text,
                                 max_vocab = 20000,
                                 max_len = 256)

# C. 封装 Dataset & DataLoader
train_ds <- SimpleIMDbDataset(train_df, pipeline)
valid_ds <- SimpleIMDbDataset(test_df, pipeline)
train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 64, shuffle = TRUE)

# D. 训练 (Luz)
fitted <- IMDbTransformer %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adamw,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) %>%
  set_hparams(
    vocab_size = pipeline$vocab_size() + 1,
    d_model = 64,    # 词向量维度
    nhead = 4,        # 注意力头数 (128 / 4 = 32, 能整除)
    num_layers = 1    # Transformer 层数 (演示用 2 层，深了更慢)
  ) %>%
  set_opt_hparams(lr = 0.0005) %>% # Transformer 对学习率较敏感，通常比 LSTM 小一点
  fit(
    train_dl,
    valid_data = valid_dl,
    epochs = 10,
    verbose = TRUE,
    callbacks = list(
      luz_callback_early_stopping(monitor = "valid_loss", patience = 2)
    )
  )

# 保存模型，用于后面章节任务
torch_save(fitted$model$state_dict(), "13_model_weights.pt")
# evaluate(fitted, valid_dl)

# 定义预测函数
predict_sentiment <- function(text) {
  input_ids <- pipeline$encode(text)
  input_tensor <- torch_tensor(input_ids, dtype = torch_long())$unsqueeze(1)
  pred <- predict(fitted, input_tensor)
  prob <- torch_sigmoid(pred)
  score <- as.numeric(prob)
  cat(sprintf("评分: %.4f (%.1f%% 概率是好评)\n", score, score * 100))
}

# 疯狂动物城 2 的几条评论
predict_sentiment("108 minutes of cinemas becoming a sanctuary again")
predict_sentiment("I love the values conveyed in this installment even more than in the first one. Thank you, Disney, for still being willing to tell a story like this in such fractured times. Had it been released ten years ago, I might have called it yet another exquisitely crafted old-fashioned confection from Disney. But precisely because it was born in the world of today, I sincerely hope there will be more films like this.")


