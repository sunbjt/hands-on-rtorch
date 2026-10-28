library(torch)
library(ggplot2)
library(reshape2)
library(gridExtra)

# ==========================================
# 1. 核心通用组件：自注意力模块
# ==========================================
SelfAttention <- nn_module(
  "SelfAttention",
  initialize = function(embed_dim) {
    self$embed_dim <- embed_dim
    # 假设输入已经是高维特征，这里只做内部投影
    self$W_q <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$W_k <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$W_v <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$W_o <- nn_linear(embed_dim, embed_dim, bias = FALSE)
  },
  forward = function(x) {
    Q <- self$W_q(x)
    K <- self$W_k(x)
    V <- self$W_v(x)
    # 第 1 维 batch，第 2 维 seq_len（T），第 3 维 embed_dim（D）
    K_t <- torch_transpose(K, dim0 = 2, dim1 = 3)  # 转置一下
    scores <- torch_matmul(Q, K_t) / sqrt(self$embed_dim)
    attention_weights <- nnf_softmax(scores, dim = -1)
    
    context <- torch_matmul(attention_weights, V)
    out <- self$W_o(context)
    return(list(out = out, weights = attention_weights))
  }
)

# ==========================================
# 2. 对照组 (失败案例)：残缺版 Transformer
# 只有注意力，没有位置编码、没有残差、没有大脑
# ==========================================
SimpleTransformer <- nn_module(
  "SimpleTransformer",
  initialize = function(input_dim, embed_dim, output_dim) {
    self$input_proj <- nn_linear(input_dim, embed_dim)
    self$attention <- SelfAttention(embed_dim)
    self$final_proj <- nn_linear(embed_dim, output_dim)
  },
  forward = function(x) {
    # 1. 特征扩容
    x_emb <- self$input_proj(x)
    # 2. 直接进行注意力计算 (路痴状态)
    att_out <- self$attention(x_emb)
    # 3. 输出预测
    preds <- self$final_proj(att_out$out)
    return(list(preds = preds, weights = att_out$weights))
  }
)

# ==========================================
# 3. 实验组 (成功案例)：完全体 Transformer
# 包含注意力、位置编码、残差连接与层归一化、FFN
# ==========================================
TransformerBlock <- nn_module(
  "TransformerBlock",
  initialize = function(embed_dim) {
    self$attention <- SelfAttention(embed_dim)
    self$norm1 <- nn_layer_norm(embed_dim)
    self$norm2 <- nn_layer_norm(embed_dim)
    self$ffn <- nn_sequential(
      nn_linear(embed_dim, embed_dim * 4),
      nn_relu(),
      nn_linear(embed_dim * 4, embed_dim)
    )
  },
  forward = function(x) {
    # 稳定器与注意力
    att_out <- self$attention(x)
    x <- self$norm1(x + att_out$out)
    # 稳定器与逻辑大脑
    ffn_out <- self$ffn(x)
    x <- self$norm2(x + ffn_out)
    return(list(out = x, weights = att_out$weights))
  }
)

FullTransformer <- nn_module(
  "FullTransformer",
  initialize = function(input_dim, embed_dim, output_dim, max_len = 10) {
    self$input_proj <- nn_linear(input_dim, embed_dim)
    self$pos_embed <- nn_embedding(num_embeddings = max_len, embedding_dim = embed_dim)
    self$transformer_block <- TransformerBlock(embed_dim)
    self$final_proj <- nn_linear(embed_dim, output_dim)
  },
  forward = function(x) {
    batch_size <- x$shape[1]
    seq_len <- x$shape[2]
    
    # 1. 特征扩容
    x_emb <- self$input_proj(x)
    # 2. 加入时间戳 (号码牌)
    positions <- torch_arange(1, seq_len, dtype = torch_long())
    pos_embeddings <- self$pos_embed(positions)
    x_combined <- x_emb + pos_embeddings
    # 3. 通过流水线
    block_out <- self$transformer_block(x_combined)
    # 4. 输出预测
    preds <- self$final_proj(block_out$out)
    return(list(preds = preds, weights = block_out$weights))
  }
)

# ==========================================
# 4. 数据与训练设置 (统一的序列反转任务)
# ==========================================
seq_len <- 10
input_dim <- 4
embed_dim <- 32
batch_size <- 64
num_epochs <- 150
learning_rate <- 0.005

# 随机生成 10 天的特征
X_train <- torch_randn(batch_size, seq_len, input_dim)
# 任务目标：严格倒序输出
Y_train <- torch_flip(X_train, dims = 2)

# 实例化两个模型和各自的优化器
model_simple <- SimpleTransformer(input_dim, embed_dim, input_dim)
model_full <- FullTransformer(input_dim, embed_dim, input_dim, max_len = seq_len)

opt_simple <- optim_adamw(model_simple$parameters, lr = learning_rate)
opt_full <- optim_adamw(model_full$parameters, lr = learning_rate)
loss_fn <- nn_mse_loss()

save_epochs <- c(1, 10, 30, 60, 100, 150)
weights_simple <- list()
weights_full <- list()

# ==========================================
# 5. 并行训练对比
# ==========================================
for (epoch in 1:num_epochs) {
  # 训练对照组 (Simple)
  opt_simple$zero_grad()
  out_simple <- model_simple(X_train)
  loss_s <- loss_fn(out_simple$preds, Y_train)
  loss_s$backward()
  opt_simple$step()
  
  # 训练实验组 (Full)
  opt_full$zero_grad()
  out_full <- model_full(X_train)
  loss_f <- loss_fn(out_full$preds, Y_train)
  loss_f$backward()
  opt_full$step()
  
  if (epoch %in% save_epochs) {
    weights_simple[[paste0("Epoch_", epoch)]] <- as.matrix(out_simple$weights[1, , ]$detach()$cpu())
    weights_full[[paste0("Epoch_", epoch)]] <- as.matrix(out_full$weights[1, , ]$detach()$cpu())
    cat(sprintf("Epoch %3d | 残缺版 Loss: %.4f | 完全体 Loss: %.4f\n", epoch, loss_s$item(), loss_f$item()))
  }
}

# ==========================================
# 6. 可视化：极其震撼的对比图
# ==========================================
plot_heatmap <- function(w_matrix, title, is_success = TRUE) {
  melted <- melt(w_matrix)
  colnames(melted) <- c("Output_Pos", "Input_Pos", "Weight")
  
  # 失败组用灰红色调暗示翻车，成功组用深绿色调暗示成功 (可选)
  color_high <- if(is_success) "#005a32" else "#a50f15"
  
  ggplot(melted, aes(x = Input_Pos, y = Output_Pos, fill = Weight)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "#f7fcf5", high = color_high, limits = c(0, 1)) +
    scale_y_reverse(breaks = 1:seq_len, expand = c(0,0)) +
    scale_x_continuous(breaks = 1:seq_len, expand = c(0,0)) +
    labs(title = title, x = "Input X", y = "Target Y") +
    theme_minimal(base_size = 10) + 
    theme(aspect.ratio = 1, plot.title = element_text(hjust = 0.5, face="bold"))
}

# 绘制残缺版图集
plots_simple <- lapply(names(weights_simple), function(name) {
  plot_heatmap(weights_simple[[name]], name, is_success = FALSE)
})
# 绘制完全体图集
plots_full <- lapply(names(weights_full), function(name) {
  plot_heatmap(weights_full[[name]], name, is_success = TRUE)
})

# 对照组：残缺版 Transformer (找不到倒序规律)
grid.arrange(grobs = plots_simple, ncol = 3, top = "")
# 实验组：完全体 Transformer (完美的反对角线)
grid.arrange(grobs = plots_full, ncol = 3, top = "")

