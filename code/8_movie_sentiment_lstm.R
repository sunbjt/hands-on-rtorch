# 数据下载地址
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# 解压至 imdb 子目录
#
################### Part 1.标准模型训练过程 ######################
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
      
      full_vocab <- c("<PAD>", "<UNK>", top_tokens)
      self$idx_to_token <- full_vocab
      self$token_to_idx <- set_names(seq_along(full_vocab), full_vocab)
      cat("词汇表构建完成。大小:", length(full_vocab), "\n")
    },
    
    encode = function(text) {
      tokens <- tibble(text = text) %>% unnest_tokens(word, text) %>% pull(word)
      indices <- self$token_to_idx[tokens]
      indices[is.na(indices)] <- self$token_to_idx["<UNK>"]
      
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
      filter_mask <- !is.na(tokens) & tokens != "<PAD>"
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

# 双向 LSTM + Mean Pooling
IMDbLSTM <- nn_module(
  "IMDbLSTM",
  initialize = function(vocab_size,
                        emb_dim,
                        hidden_dim,
                        output_dim = 1) {
    self$embedding <- nn_embedding(vocab_size, emb_dim, padding_idx = 1)
    self$lstm <- nn_lstm(emb_dim,
                         hidden_dim,
                         batch_first = TRUE,
                         bidirectional = TRUE)
    self$fc <- nn_linear(hidden_dim * 2, output_dim) # 双向所以 *2
    self$dropout <- nn_dropout(0.5)
  },
  forward = function(x) {
    x <- self$embedding(x)
    output <- self$lstm(x)
    # Mean Pooling: 对所有时间步求平均，比只取最后一个稳健
    avg_pool <- torch_mean(output[[1]], dim = 2)
    self$fc(self$dropout(avg_pool))
  }
)

# A. 加载原始数据 
base_path <- "~/data/imdb/aclImdb"
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
fitted <- IMDbLSTM %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) %>%
  set_hparams(
    vocab_size = pipeline$vocab_size(),
    emb_dim = 128,
    hidden_dim = 64
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_dl,
    valid_data = valid_dl,
    epochs = 10,
    verbose = TRUE,
    callbacks = list(
      luz_callback_early_stopping(monitor = "valid_loss", patience = 2) # 没提升就早停
    )
  )

# 绘制损失、acc 曲线
plot(fitted)

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
predict_sentiment("My God, this is so terrible. Didn’t they say Baobote was the bad guy? Then why did they get into his car afterward? What the heck??? Just a pure moe anime.")

################### Part 2.查看 embedding 变化情况 ######################

# callback 函数定义
EmbeddingRecorder <- luz_callback(
  name = "EmbeddingRecorder",
  
  initialize = function() {
    self$history <- list()
  },
  
  on_epoch_end = function() {
    full_weights <- as_array(ctx$model$embedding$weight$detach()$cpu())
    self$history[[length(self$history) + 1]] <- list(
      epoch = ctx$epoch,
      weights = full_weights
    )
  }
)
# 实例化
embed_monitor <- EmbeddingRecorder()

# 训练 2 轮
fitted <- IMDbLSTM %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) %>%
  set_hparams(
    vocab_size = pipeline$vocab_size(),
    emb_dim = 128,
    hidden_dim = 64
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_dl,
    valid_data = valid_dl,
    epochs = 2, 
    verbose = TRUE,
    callbacks = list(
      embed_monitor # 注入我们的回调
    )
  )

# 定义分析函数
analyze_word_pairs <- function(history_list, pipeline, target_words) {
  
  word_indices <- map_int(target_words, function(w) {
    idx <- pipeline$token_to_idx[w]
    if (is.na(idx)) stop(paste("单词不在词表中:", w))
    idx
  })
  names(word_indices) = target_words
  pairs <- combn(target_words, 2, simplify = FALSE)

  map_dfr(history_list, function(record) {
    epoch <- record$epoch
    full_matrix <- record$weights # 这是一个 20002 x 128 的大矩阵
    
    map_dfr(pairs, function(p) {
      w1 <- p[1]
      w2 <- p[2]
      
      idx1 <- word_indices[w1]
      idx2 <- word_indices[w2]
      
      v1 <- full_matrix[idx1, ]
      v2 <- full_matrix[idx2, ]
      
      tibble(
        epoch = epoch,
        pair = paste(w1, "-", w2),
        similarity = lsa::cosine(v1, v2) %>% as.numeric()
      )
    })
  })
}

mixed_words <- c("amazing", "terrible", "bad", "excellent")
df_mixed <- analyze_word_pairs(embed_monitor$history, pipeline, mixed_words)
emb_p <- ggplot(df_mixed, aes(x = epoch, y = similarity, color = pair)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_y_continuous(limits = c(-0.15, 0.15)) + 
  scale_x_continuous(breaks = seq(1, 2, by = 1)) +
  theme_minimal(base_family = 'Noto Sans SC') +
  labs(
    y = "Cosine Similarity",
    x = "Epoch"
  )

emb_p

ggsave(
  filename = 'good-great-wonderful.png',
  plot = emb_p,
  device = 'png',
  path = '~/github/rtorchbook/part-03-advanced-arch/images/',
  width = 15,
  height = 9,
  units = "cm"
)

################### Part 3.Fine-tuning 技术的应用 ######################

# --- 加载 GloVe 预训练向量 ---
library(data.table)
load_pretrained_embeddings <- function(file_path, pipeline, emb_dim = 100) {
  glove_dt <- fread(file_path, 
                    header = FALSE, 
                    quote = "", 
                    encoding = "UTF-8",
                    key = "V1") # 将第一列(单词)设为索引，加速查找
  
  cat(sprintf("GloVe 读取完毕，包含 %d 个词向量。\n", nrow(glove_dt)))
  
  vocab_words <- pipeline$idx_to_token
  vocab_size <- length(vocab_words)
  
  my_vocab_dt <- data.table(
    word = vocab_words,
    idx = 1:vocab_size
  )
  
  # glove_dt 的 V1 是词，V2...V101 是向量
  matched_dt <- merge(my_vocab_dt, glove_dt, by.x = "word", by.y = "V1", sort = FALSE)
  
  # 4. 初始化最终权重矩阵
  # 默认使用正态分布初始化 (用于覆盖那些 GloVe 里没有的词，如 <UNK>)
  final_matrix <- matrix(rnorm(vocab_size * emb_dim, sd = 0.1), 
                         nrow = vocab_size, ncol = emb_dim)
  
  # 将 <PAD> (索引为 1) 设为 0
  pad_idx <- pipeline$token_to_idx["<PAD>"]
  final_matrix[pad_idx, ] <- 0
  
  # 5. 填入匹配到的 GloVe 权重
  if (nrow(matched_dt) > 0) {
    # 提取匹配到的索引
    indices <- matched_dt$idx
    vec_cols <- paste0("V", 2:(emb_dim + 1))
    vectors <- as.matrix(matched_dt[, ..vec_cols])
    final_matrix[indices, ] <- vectors
    
    cat(sprintf("匹配命中率: %.2f%% (%d/%d)\n", 
                (nrow(matched_dt) / vocab_size) * 100, 
                nrow(matched_dt), vocab_size))
  } else {
    warning("未匹配到任何 GloVe 向量，请检查大小写或文件格式！")
  }
  
  rm(glove_dt, matched_dt)
  gc()
  
  return(torch_tensor(final_matrix, dtype = torch_float()))
}


# --- 修改后的模型类 ---

IMDbLSTM <- nn_module(
  "IMDbLSTM",
  initialize = function(vocab_size,
                        emb_dim,
                        hidden_dim,
                        pretrained_weights = NULL, # 新增参数
                        output_dim = 1,
                        freeze_embeddings = FALSE) { # 是否冻结 embedding 层

    self$embedding <- nn_embedding(vocab_size, emb_dim, padding_idx = 1)
    if (!is.null(pretrained_weights)) {
      self$embedding$weight$set_data(pretrained_weights)
      self$embedding$weight$requires_grad_( !freeze_embeddings )
    }
    
    self$lstm <- nn_lstm(emb_dim,
                         hidden_dim,
                         batch_first = TRUE,
                         bidirectional = TRUE)
    
    self$fc <- nn_linear(hidden_dim * 2, output_dim)
    self$dropout <- nn_dropout(0.5)
  },
  
  forward = function(x) {
    x <- self$embedding(x)
    output <- self$lstm(x)
    avg_pool <- torch_mean(output[[1]], dim = 2)
    self$fc(self$dropout(avg_pool))
  }
)

# --- 执行流程修改 ---

# 1. 确保维度匹配 (GloVe 通常是 50, 100, 200, 300)
EMBED_DIM <- 50 
glove_path <- file.path(base_path, "wiki_giga_2024.txt")
pretrained_emb <- load_pretrained_embeddings(glove_path, pipeline, emb_dim = EMBED_DIM)

# 3. 训练 (传入 pretrained_weights)
fitted_emb <- IMDbLSTM %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) %>%
  set_hparams(
    vocab_size = pipeline$vocab_size(),
    emb_dim = EMBED_DIM,  # 50 以匹配 GloVe
    hidden_dim = 64,
    pretrained_weights = pretrained_emb # 注入权重
  ) %>%
  set_opt_hparams(lr = 0.001) %>% # 有预训练权重时，lr 可以稍微调小一点，如 5e-4，防止剧烈震荡
  fit(
    train_dl,
    valid_data = valid_dl,
    epochs = 5, # 有预训练通常收敛更快，但也可以多跑几轮
    verbose = TRUE,
    callbacks = list(
      luz_callback_early_stopping(monitor = "valid_loss", patience = 2)
    )
  )

plot(fitted_emb)
