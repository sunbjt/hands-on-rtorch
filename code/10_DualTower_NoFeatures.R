# 载入数据预处理环节
source('~/github/rtorchbook/code/10_Leave_One_Out.R')
# 载入 dataset 定义和评估定义
source('~/github/rtorchbook/code/10_dataset.R')

## 定义双塔模型
dual_tower_model <- nn_module(
  "DualTower",
  initialize = function(num_users,
                        num_items,
                        embedding_dim,
                        latent_dim) {
    self$user_embedding <- nn_embedding(num_users, embedding_dim)
    self$item_embedding <- nn_embedding(num_items, embedding_dim)
    
    # 用户塔：独立的 MLP
    self$user_tower <- nn_sequential(nn_linear(embedding_dim, 64),
                                     nn_relu(),
                                     nn_linear(64, latent_dim))
    
    # 物品塔：独立的 MLP
    self$item_tower <- nn_sequential(nn_linear(embedding_dim, 64),
                                     nn_relu(),
                                     nn_linear(64, latent_dim))
    
    nn_init_normal_(self$user_embedding$weight, std = 0.01)
    nn_init_normal_(self$item_embedding$weight, std = 0.01)
  },
  
  # 【修改点】：统一接收 dataloader 传来的具名列表 x
  forward = function(x) {
    u_rep <- self$user_tower(self$user_embedding(x$user))
    i_rep <- self$item_tower(self$item_embedding(x$item))
    
    # 最终打分退化为点积，彻底解耦
    return(torch_sum(u_rep * i_rep, dim = -1))
  }
)

fitted_model_dual <- dual_tower_model %>%
  setup(loss = nnf_binary_cross_entropy_with_logits, optimizer = optim_adam) %>%
  set_hparams(num_users = num_users,
              num_items = num_items,
              embedding_dim = 32,
              latent_dim = 32) %>%
  set_opt_hparams(lr = 0.005, weight_decay = 1e-5) %>%
  fit(
    data = dl_train,
    valid_data = dl_eval,
    epochs = 15,
    accelerator = accelerator()
  )

# 执行评估
recall_score <- evaluate_recall(fitted_model_dual, dl_eval, k = 10)
cat(sprintf("=> Recall@10: %.4f\n", recall_score))


