library(tidymodels)
library(torch)
library(luz)
library(tidyverse)
library(showtext)
library(knitr)
library(ggplot2)
library(igraph)
library(tidygraph)
library(ggraph)

options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条
showtext_auto()
set.seed(123)
torch_manual_seed(123)

# --- 1. 数据准备与 Tidymodels 预处理 ---
data(ames)

# 划分训练集与测试集 (80/20)
ames_split <- initial_split(ames, prop = 0.80)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# 构建 Recipe (与 MLP 章节保持一致的优良预处理步骤)
rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  # 0. 标准化目标变量 (加速收敛，预测时需注意还原)
  step_normalize(all_outcomes()) %>%
  # 1. 剔除无用的空间坐标/ID 等无预测力的列
  step_rm(matches("Id|Longitude|Latitude")) %>% 
  # 2. 数值特征偏态处理 (对数变换)
  step_log(Gr_Liv_Area, Lot_Area, base = 10) %>% 
  # 3. 合并稀有类别 (降维防过拟合)
  step_other(all_nominal_predictors(), threshold = 0.01) %>% 
  # 4. 哑变量化 (独热编码)
  step_dummy(all_nominal_predictors()) %>% 
  # 5. 剔除零方差特征
  step_nzv(all_predictors()) %>%
  # 6. 数值特征标准化 (深度学习的基石)
  step_normalize(all_predictors()) %>%
  # 7. 缺失值插补
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

# 训练 Recipe 并应用到训练集和测试集
prep_rec <- prep(rec)
train_df <- bake(prep_rec, new_data = NULL)
test_df  <- bake(prep_rec, new_data = ames_test)

# --- 2. 转换为 Torch Tensors 与 Dataloaders ---
# 提取自变量矩阵和因变量矩阵，并转为 Float32 类型的 Tensor
x_train <- torch_tensor(as.matrix(train_df %>% select(-Sale_Price)), dtype = torch_float32())
y_train <- torch_tensor(matrix(train_df$Sale_Price, ncol = 1), dtype = torch_float32())

x_test <- torch_tensor(as.matrix(test_df %>% select(-Sale_Price)), dtype = torch_float32())
y_test <- torch_tensor(matrix(test_df$Sale_Price, ncol = 1), dtype = torch_float32())

# 定义自定义的 Dataset 类
ames_dataset <- dataset(
  name = "ames_ds",
  initialize = function(x, y) {
    self$x <- x
    self$y <- y
  },
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i, ])
  },
  .length = function() {
    self$x$size(1)
  }
)

# 实例化 Dataset 并包装进 Dataloader
train_ds <- ames_dataset(x_train, y_train)
test_ds  <- ames_dataset(x_test, y_test)

train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
test_dl  <- dataloader(test_ds, batch_size = 128)

# --- 3. 可视化 Attention 版本 Tabular Transformer 架构 ---
ExplainableTabularTransformer <- nn_module(
  "ExplainableTabularTransformer",
  
  initialize = function(num_features, d_model = 32, nhead = 4, dropout = 0.1) {
    self$num_features <- num_features
    self$d_model <- d_model
    
    # 1. 独立特征分词器 (Independent Feature Tokenizer)
    # 抛弃 nn_linear，直接定义每个特征专属的权重和偏置
    self$W_proj <- nn_parameter(torch_randn(num_features, d_model) * 0.02)
    self$b_proj <- nn_parameter(torch_zeros(num_features, d_model))
    
    # 特征身份编码与 [CLS] Token
    self$feature_emb <- nn_parameter(torch_randn(1, num_features, d_model) * 0.02)
    self$cls_token <- nn_parameter(torch_randn(1, 1, d_model) * 0.02)
    
    # 2. Multi-Head Attention 层
    self$self_attn <- nn_multihead_attention(
      embed_dim = d_model, 
      num_heads = nhead, 
      batch_first = TRUE
    )
    
    # 3. 前馈网络 (增加 Dropout 防过拟合)
    self$ffn <- nn_sequential(
      nn_linear(d_model, d_model * 2), # 依照惯例，中间层通常比 d_model 宽
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(d_model * 2, d_model)
    )
    
    # 4. Pre-LN 结构的归一化层
    self$norm1 <- nn_layer_norm(d_model)
    self$norm2 <- nn_layer_norm(d_model)
    self$norm_head <- nn_layer_norm(d_model) # 预测前的最终归一化
    
    self$head <- nn_linear(d_model, 1)
  },
  
  forward = function(x) {
    batch_size <- x$size(1)
    
    # [步骤 A: 独立特征投影] 
    # x: [Batch, Features] -> [Batch, Features, 1]
    x_expanded <- x$unsqueeze(-1) 
    
    # 利用广播机制：[B, F, 1] * [F, d_model] + [F, d_model] -> [B, F, d_model]
    # 此时，每一个特征都由自己专属的权重投影到了高维空间！
    tokens <- (x_expanded * self$W_proj) + self$b_proj
    
    # 加上特征身份标识
    tokens <- tokens + self$feature_emb
    
    # 拼接 [CLS]
    cls_tokens <- self$cls_token$expand(c(batch_size, -1, -1))
    seq_tokens <- torch_cat(list(cls_tokens, tokens), dim = 2)
    
    # [步骤 B: 注意力机制与 Pre-LN 残差]
    # 1. 先做 Norm，再进 Attention
    normed_seq <- self$norm1(seq_tokens)
    
    attn_out <- self$self_attn(
      query = normed_seq, 
      key = normed_seq, 
      value = normed_seq, 
      need_weights = TRUE
    )
    
    # 保存权重以供可视化
    self$last_attn_weights <- attn_out[[2]] 
    
    # 2. 残差连接 1
    seq_tokens <- seq_tokens + attn_out[[1]]
    
    # 3. 先做 Norm，再进 FFN，最后残差连接 2
    seq_tokens <- seq_tokens + self$ffn(self$norm2(seq_tokens))
    
    # [步骤 C: 提取 [CLS] 并预测]
    cls_out <- seq_tokens[, 1, ] 
    
    # Pre-LN 结构通常在进入输出头之前再加一次 Norm
    cls_out <- self$norm_head(cls_out) 
    
    self$head(cls_out)
  }
)

input_dim <- ncol(x_train)
transformer_fit <- ExplainableTabularTransformer |>
  setup(loss = nn_mse_loss(),
        optimizer = optim_adamw,
        metrics = list(luz_metric_rmse())) |>
  set_hparams(num_features = input_dim, d_model = 32, nhead = 4) |>
  set_opt_hparams(lr = 0.001, weight_decay = 1e-4) |>
  fit(data = train_dl, valid_data = test_dl,
      epochs = 20, verbose = TRUE) # epochs可根据需要调大

plot(transformer_fit)

# 获取在测试集上的最终评估指标
cat("\n--- Transformer 测试集评估结果 ---\n")
eval_results <- transformer_fit |> evaluate(test_dl)
eval_results |> get_metrics()

# 假设从 luz 中提取出训练好的底层模型对象：
trained_model <- transformer_fit$model
trained_model$eval()

# --- 对比“最贵”与“最便宜”房子的注意力分布 ---

feature_names <- colnames(train_df %>% select(-Sale_Price))
all_tokens <- c("[CLS]", feature_names)

# 1. 获取两套房子的索引
idx_max <- which.max(test_df$Sale_Price) # 最贵
idx_min <- which.min(test_df$Sale_Price) # 最便宜

# 2. 获取模型所在的设备
model_device <- trained_model$parameters[[1]]$device

# 3. 写一个辅助函数，用于提取单套房子的 [CLS] 注意力
get_cls_attention <- function(idx, label) {
  x_single <- x_test[idx, ]$unsqueeze(1)$to(device = model_device)
  
  with_no_grad({
    # 前向传播
    trained_model(x_single)
    # 提取权重并搬回 CPU
    attn_tensor <- trained_model$last_attn_weights$cpu()
  })
  
  attn_matrix <- as.matrix(attn_tensor[1, , ])
  rownames(attn_matrix) <- all_tokens
  colnames(attn_matrix) <- all_tokens
  
  # 提取第一行 ([CLS] 对其他所有特征的注意力)
  tibble(
    Feature = feature_names,
    Attention = attn_matrix["[CLS]", -1],
    House_Type = label
  )
}

# 4. 现在再次运行你的函数调用
attn_max_df <- get_cls_attention(idx_max, "最贵豪宅")
attn_min_df <- get_cls_attention(idx_min, "最便宜住宅")

# 5. 确定要展示的 Top 特征（以最贵豪宅的 Top 10 为基准）
top_10_features <- attn_max_df %>%
  arrange(desc(Attention)) %>%
  slice_head(n = 12) %>%
  pull(Feature)

# 6. 合并数据并过滤出这 15 个特征
combined_attn <- bind_rows(attn_max_df, attn_min_df) %>%
  filter(Feature %in% top_10_features) %>%
  mutate(
    Color = case_when(
      House_Type == "最贵豪宅" ~ "#08306B",
      House_Type == "最便宜住宅" ~ "#FFC125"
    )
  )

# 7. 使用 ggplot2 绘制并排条形图
ggplot(combined_attn, aes(
  x = reorder(Feature, Attention),
  y = Attention,
  fill = Color
)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_fill_identity(guide = "legend", labels = c("最贵豪宅", "最便宜住房")) +
  labs(
    x = "房屋特征",
    y = "注意力权重 (Attention Weight)",
    fill = "房屋类型"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 10)
  )


# --- 【图表 2 & 3 准备】：全局截获完整的 NxN 注意力张量 ---
x_single_max <- x_test[idx_max, ]$unsqueeze(1)$to(device = model_device)
with_no_grad({
  trained_model(x_single_max)
  # 全局保存注意力张量！
  attn_tensor <- trained_model$last_attn_weights$cpu()
})

attn_matrix_full <- as.matrix(attn_tensor[1, , ])
rownames(attn_matrix_full) <- all_tokens
colnames(attn_matrix_full) <- all_tokens

top_features_to_show <- c("[CLS]", top_10_features) 
small_attn_matrix <- attn_matrix_full[top_features_to_show, top_features_to_show]

# --- 绘制图 2：特征交互热力图 ---
attn_df_long <- small_attn_matrix %>%
  as.data.frame() %>%
  rownames_to_column(var = "Query") %>%
  pivot_longer(cols = -Query, names_to = "Key", values_to = "Attention") %>%
  mutate(
    Query = factor(Query, levels = rev(top_features_to_show)),
    Key = factor(Key, levels = top_features_to_show)
  )

plot_heatmap <- ggplot(attn_df_long, aes(x = Key, y = Query, fill = Attention)) +
  geom_tile(color = "grey80", linewidth = 0.2) + 
  scale_fill_gradient(low = "#FFFFFF", high = "#08306B", name = "注意力权重") +
  labs(title = "Transformer 特征交互热力图", x = "Key (被关注)", y = "Query (发出关注)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    panel.grid = element_blank()
  ) + coord_fixed()
print(plot_heatmap)

# --- 绘制图 3：信息流向网络拓扑图 ---
edges <- small_attn_matrix %>%
  as.data.frame() %>%
  rownames_to_column(var = "Query") %>%
  pivot_longer(cols = -Query, names_to = "Key", values_to = "Attention") %>%
  filter(Attention > 0.015) %>%
  rename(from = Key, to = Query, weight = Attention)

nodes <- tibble(name = top_features_to_show) %>%
  mutate(
    node_type = if_else(name == "[CLS]", "CLS Hub", "Physical Feature"),
    node_size = if_else(name == "[CLS]", 8, 4) 
  )

attn_graph <- tbl_graph(nodes = nodes, edges = edges, directed = TRUE)

set.seed(42)
ggraph(attn_graph, layout = 'fr') +
  geom_edge_link(
    aes(edge_alpha = weight, edge_width = weight),
    arrow = arrow(length = unit(3, 'mm'), type = "closed"),
    end_cap = circle(4, 'mm'),
    color = "grey40"
  ) +
  scale_edge_width(range = c(0.5, 2.5), guide = "none") +
  scale_edge_alpha(range = c(0.3, 0.9), name = "注意力强度") +
  geom_node_point(aes(color = node_type, size = node_size)) +
  scale_color_manual(
    values = c(
      "CLS Hub" = "#08306B",
      "Physical Feature" = "grey70"
    ),
    name = "节点类型"
  ) +
  scale_size_identity() +
  geom_node_text(
    aes(label = name),
    repel = TRUE,
    size = 4,
    fontface = "bold",
    bg.color = "white",
    bg.r = 0.15
  ) +
  theme_void() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", hjust = 0.5))
# Transformer 信息流向拓扑图
# 箭头指向代表信息汇聚的方向，CLS 作为超级枢纽吸收全局特征
