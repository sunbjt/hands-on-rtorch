# 载入数据预处理环节
source('~/github/rtorchbook/code/10_Leave_One_Out.R')
# 载入 dataset 定义和评估定义
source('~/github/rtorchbook/code/10_dataset.R')

# 矩阵分解模型：融合 Embedding 与 偏置项
nn_matrix_factorization <- nn_module(
  "MatrixFactorizationModel",
  
  initialize = function(num_users, num_items, embed_dim = 32) {
    # 核心特征向量层
    self$user_embed  <- nn_embedding(num_users, embed_dim)
    self$item_embed  <- nn_embedding(num_items, embed_dim)
    
    # 偏置项层 (Bias Terms)
    self$user_bias   <- nn_embedding(num_users, 1)
    self$item_bias   <- nn_embedding(num_items, 1)
    self$global_bias <- nn_parameter(torch_zeros(1))
  },
  
  forward = function(x) {
    # 1. 提取隐向量 (p_u 和 q_i)
    u <- self$user_embed(x$user)
    i <- self$item_embed(x$item)
    
    # 2. 计算点积项: p_u^T * q_i
    dot_product <- (u * i)$sum(dim = -1)
    
    # 3. 提取偏置项 (b_u 和 b_i)
    u_b <- self$user_bias(x$user)$squeeze(-1)
    i_b <- self$item_bias(x$item)$squeeze(-1)
    
    # 4. 最终融合: dot_product + b_u + b_i + mu
    dot_product + u_b + i_b + self$global_bias
  }
)

fitted_model <- nn_matrix_factorization %>%
  setup(loss = nnf_binary_cross_entropy_with_logits,
        optimizer = optim_adam) %>%
  set_hparams(num_users = num_users,
              num_items = num_items,
              embed_dim = 32) %>%
  set_opt_hparams(lr = 0.005, weight_decay = 1e-5) %>%
  fit(
    data = dl_train,
    valid_data = dl_eval,
    epochs = 15,
    accelerator = accelerator()
  )

# 执行评估
recall_score <- evaluate_recall(fitted_model, dl_eval, k = 10)
cat(sprintf("=> Recall@10: %.4f\n", recall_score))


