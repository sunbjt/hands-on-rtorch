library(torch)
library(tidyverse)
library(tidytext)
library(fs)
library(luz)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

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

# --- 5. 训练执行 ---
base_path <- "/Users/liusizhe/data/imdb/aclImdb/train/" 
df <- load_imdb_data(base_path)
pipeline <- TidyTextPipeline$new(df$text, max_vocab = 5000, max_len = 32)
train_ds <- ImdbGenDataset(df, pipeline)
train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

pad_int <- as.integer(pipeline$pad_idx)
fitted <- setup(
  module = QuerySuggestGPT,
  loss = nn_cross_entropy_loss(ignore_index = pad_int),
  optimizer = optim_adamw # 使用 AdamW 通常效果更好
) %>%
  set_hparams(
    vocab_size = length(pipeline$idx_to_token),
    d_model = 256,
    nhead = 4,
    num_layers = 2,
    max_len = pipeline$max_len,
    pad_idx = pad_int,
    dropout = 0.1
  ) %>%
  fit(
    train_dl,
    epochs = 40, # 演示用 5 epoch
    accelerator = accelerator(), # 自动检测 CPU/MPS/CUDA
    verbose = TRUE
  )
plot(fitted)

# --- 测试效果 ---
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

final_model <- fitted$model
# 1. 贪婪模式 (最稳健，但可能重复)
suggest_next(final_model, pipeline, "the movie was", temp = 0.1)
# 2. 正常采样 (平衡)
suggest_next(final_model, pipeline, "i really liked", temp = 0.8)
# 3. 高创造性 (可能胡言乱语)
suggest_next(final_model, pipeline, "the plot is", temp = 1.2)

# 保存参数
setwd("/Users/liusizhe/data")
torch_save(final_model$state_dict(), "imdb_gpt_base.pt")
write_rds(pipeline, "text_pipeline.rds")


## ----------------------------------------
## 可视化结果
## ----------------------------------------
library(ggraph)
library(igraph)
library(showtext)
library(ggplot2)
showtext_auto()

# --- 1. 提取生成路径树数据 (加入 Threshold 逻辑) ---
get_tree_data <- function(model, pipeline, start_text, depth = 3, threshold = 0.05, max_branches = 5) {
  model$eval()
  dev <- model$parameters[[1]]$device
  
  # 初始化节点和边的容器
  nodes_list <- list()
  edges_list <- list()
  node_counter <- 1
  
  # 根节点
  root_id <- paste0("n", node_counter)
  nodes_list[[node_counter]] <- data.frame(id = root_id, label = start_text, depth = 0, stringsAsFactors = FALSE)
  
  # 编码初始文本
  idxs <- pipeline$encode(start_text)
  idxs <- idxs[idxs != pipeline$pad_idx]
  ctx <- torch_tensor(idxs, dtype = torch_long(), device = dev)$unsqueeze(1)
  
  # 递归展开分支
  expand_node <- function(current_ctx, parent_id, current_depth) {
    if (current_depth >= depth) return()
    
    with_no_grad({
      w <- pipeline$max_len
      if (current_ctx$size(2) > w) {
        ctx_cond <- current_ctx[, (current_ctx$size(2) - w + 1):current_ctx$size(2)]
      } else {
        ctx_cond <- current_ctx
      }
      
      # 模型推理
      logits <- model(ctx_cond) 
      logits <- logits$transpose(2, 3)
      last_logits <- logits[, logits$size(2), ]
      
      # 获取概率分布
      probs <- nnf_softmax(last_logits, dim = -1)
      
      # 【核心修改点】：不再固定取 top_k，而是将概率降序排列后应用阈值逻辑
      sorted_res <- torch_sort(probs, dim = -1, descending = TRUE)
      
      all_probs <- as.numeric(sorted_res[[1]]$cpu())
      all_idxs <- as.integer(sorted_res[[2]]$cpu())
    })
    
    # --- 动态阈值过滤逻辑 ---
    
    # 1. 过滤掉低于阈值的词
    valid_mask <- all_probs >= threshold
    valid_probs <- all_probs[valid_mask]
    valid_idxs <- all_idxs[valid_mask]
    
    # 2. 保底机制：如果全军覆没 (没有任何词超过阈值)，强制保留概率最高的 2 个词
    if (length(valid_probs) == 0) {
      valid_probs <- all_probs[1:2]
      valid_idxs <- all_idxs[1:2]
    }
    
    # 3. 防爆炸机制：如果符合条件的词太多，强制截断，防止树枝过于拥挤
    if (length(valid_probs) > max_branches) {
      valid_probs <- valid_probs[1:max_branches]
      valid_idxs <- valid_idxs[1:max_branches]
    }
    
    # ------------------------
    
    # 遍历筛选后的节点并继续递归
    for (i in seq_along(valid_idxs)) {
      word_idx <- valid_idxs[i]
      word <- pipeline$idx_to_token[word_idx]
      prob <- valid_probs[i]
      
      # 忽略特殊 token
      if (word %in% c("<PAD>", "<UNK>", "<CLS>") || is.na(word)) next
      
      # 注册新节点
      node_counter <<- node_counter + 1
      child_id <- paste0("n", node_counter)
      
      nodes_list[[node_counter]] <<- data.frame(id = child_id, label = word, depth = current_depth + 1, stringsAsFactors = FALSE)
      edges_list[[length(edges_list) + 1]] <<- data.frame(from = parent_id, to = child_id, prob = prob, stringsAsFactors = FALSE)
      
      # 拼接上下文，继续向下递归
      next_idx_tensor <- torch_tensor(matrix(word_idx), dtype = torch_long(), device = dev)
      next_ctx <- torch_cat(list(current_ctx, next_idx_tensor), dim = 2)
      
      expand_node(next_ctx, child_id, current_depth + 1)
    }
  }
  
  expand_node(ctx, root_id, 0)
  
  # 合并数据框
  nodes_df <- do.call(rbind, nodes_list)
  edges_df <- do.call(rbind, edges_list)
  
  list(nodes = nodes_df, edges = edges_df)
}

# --- 2. 绘制路径树 (采用优化后的美观版本) ---
plot_generation_tree <- function(tree_data) {
  
  g <- graph_from_data_frame(tree_data$edges, directed = TRUE, vertices = tree_data$nodes)
  
  ggraph(g, layout = 'dendrogram', circular = FALSE) +
    
    geom_edge_diagonal(aes(edge_width = prob, color = prob, edge_alpha = prob), lineend = "round") +
    scale_edge_width(range = c(0.8, 4)) +
    scale_edge_alpha(range = c(0.4, 1)) +
    scale_edge_color_gradient(low = "#b2d8d8", high = "#005691") +

    geom_node_label(aes(label = label), family = "Arial", hjust = 0, nudge_y = -0.05, 
                    size = 3.5, color = "#333333", label.size = 0, fill = alpha("white", 0.8)) +
    
    coord_flip(clip = "off") +         
    # 3. 核心修改：将 mult 的第二个参数从 0.1 改为 0.02，大幅减少左侧起始处的空白
    scale_y_reverse(expand = expansion(mult = c(0.4, 0.02))) + 
    
    theme_void() +
    theme(
      legend.position = "none",
      # 将左侧 margin (最后一个数值) 从 20 调小到 5，进一步收紧左侧边缘
      plot.margin = margin(20, 60, 20, 5) 
    )
}

# --- 3. 运行代码 ---
# 提取数据：将原来的 top_k 替换为 threshold = 0.05 (大于 5% 才保留)
tree_data <- get_tree_data(
  model = final_model, 
  pipeline = pipeline, 
  start_text = "This is one of", 
  depth = 5, 
  threshold = 0.05, 
  max_branches = 2
)

# 画图
library(showtext)
showtext_auto()
plot_generation_tree(tree_data)




