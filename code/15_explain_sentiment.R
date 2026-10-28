## 执行 movie_sentiment_Transformer.R 训练之前的所有代码
## 保证加载环境一致

model <- IMDbTransformer(
  vocab_size = pipeline$vocab_size() + 1,
  d_model = 64,
  nhead = 4,
  num_layers = 1
)

weights <- torch_load('data/13_model_weights.pt')
model$load_state_dict(weights)
model$eval()

explain_sentiment <- function(text, model, pipeline, steps = 50) {
  
  if (inherits(model, "luz_module_fitted")) {
    raw_model <- model$model
  } else {
    raw_model <- model
  }
  
  raw_model$to(device = "cpu")
  
  # 编码
  indices <- pipeline$encode(text)
  input_tensor <- torch_tensor(indices, dtype = torch_long(), device = "cpu")$unsqueeze(1)
  
  raw_model$eval() 
  
  # 获取 d_model 维度
  d_model_val <- raw_model$embedding$embedding_dim
  
  # A. Embedding + Scaling
  embeddings <- raw_model$embedding(input_tensor) * sqrt(d_model_val)
  
  # B. 开启梯度追踪
  embeddings$requires_grad_(TRUE)
  embeddings$retain_grad()
  
  # C. 后续层
  x <- raw_model$pos_encoder(embeddings)
  
  # Mask 生成
  src_key_padding_mask <- (input_tensor == 1)
  pooling_mask <- (input_tensor != 1)$to(dtype = torch_float(), device = "cpu")$unsqueeze(3)
  
  out <- raw_model$transformer_encoder(x, src_key_padding_mask = src_key_padding_mask)
  
  # Mean Pooling
  out_masked <- out * pooling_mask
  sum_out <- torch_sum(out_masked, dim = 2)
  lens <- torch_sum(pooling_mask, dim = 2)
  mean_pool <- sum_out / torch_clamp(lens, min = 1)
  
  # Logits
  logits <- raw_model$fc(mean_pool)
  
  # 反向传播
  raw_model$zero_grad()
  logits$backward()
  
  # 提取梯度
  gradients <- embeddings$grad
  
  # 计算积分梯度近似值 (Sum over embedding dim)
  attributions <- torch_sum(embeddings * gradients, dim = 3)$squeeze(1)
  
  # 转为 R 数值
  attr_scores <- as.numeric(attributions)
  pred_prob <- as.numeric(torch_sigmoid(logits))
  token_labels <- pipeline$idx_to_token[as.numeric(indices)]
  
  return(list(
    tokens = token_labels,
    scores = attr_scores,
    pred_prob = pred_prob
  ))
}


text_review <- "My God, this is so terrible. Didn’t they say Baobote was the bad guy? Then why did they get into his car afterward? What the heck??? Just a pure moe anime"

# 运行解释函数 (使用之前修复过 CPU Bug 的版本)
expl <- explain_sentiment(text_review, model, pipeline)
vidd <- data.frame(token = expl$tokens, score = round(expl$scores, 2))
wide_df <- bind_cols(vidd[1:10, ], vidd[11:20, ], vidd[21:30, ])
wide_df

# library(datapasta)
# tribble_paste(wide_df)




