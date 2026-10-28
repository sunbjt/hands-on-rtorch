# 载入数据预处理环节
source('~/github/rtorchbook/code/10_Leave_One_Out.R')
# 载入 dataset 定义和评估定义
source('~/github/rtorchbook/code/10_dataset.R')

# ==============================================================================
# 引入附加特征 (User 画像与 Item 属性)
# ==============================================================================

# --- A. 处理 Users 特征 ---
FILE_USERS <- file.path(DATA_DIR, "users.dat")
dt_users <- vroom(
  FILE_USERS,
  delim = "::",
  col_names = c("user_id", "gender", "age", "occupation", "zip"),
  col_select = c(user_id, gender, age, occupation),
  show_col_types = FALSE
)
setDT(dt_users)

# 1. 映射 User ID (复用步骤1的 unique_users 保证严格对齐)
dt_users[, user_idx := match(user_id, unique_users)]
dt_users <- dt_users[!is.na(user_idx)] # 剔除没有交互记录的用户

# 2. 类别特征连续整数化 (Torch Embedding 强制要求从 1 开始连续)
dt_users[, gender_idx := as.integer(as.factor(gender))]
dt_users[, age_idx    := as.integer(as.factor(age))]
dt_users[, occ_idx    := as.integer(as.factor(occupation))]

# 记录类别数量，用于初始化 Embedding 的维度
num_genders <- max(dt_users$gender_idx)
num_ages    <- max(dt_users$age_idx)
num_occs    <- max(dt_users$occ_idx)

# --- B. 处理 Movies 特征 (核心：向量化构建 Multi-hot 矩阵) ---
FILE_MOVIES <- file.path(DATA_DIR, "movies.dat")
dt_movies <- vroom(
  FILE_MOVIES,
  delim = "::",
  col_names = c("movie_id", "title", "genres"),
  col_select = c(movie_id, genres),
  show_col_types = FALSE
)
setDT(dt_movies)

# 1. 映射 Item ID
dt_movies[, item_idx := match(movie_id, unique_items)]
dt_movies <- dt_movies[!is.na(item_idx)]

# 2. 利用 data.table 炸开多值列并透视，瞬间完成 Multi-hot
# 将 Action|Adventure 炸开成长表
dt_genres_long <- dt_movies[, .(genre = unlist(strsplit(genres, "\\|"))), by = item_idx]
dt_genres_long[, value := 1L] # 标记为1

# 透视为宽表 (Multi-hot)，缺失值填 0
dt_genres_wide <- dcast(dt_genres_long, item_idx ~ genre, value.var = "value", fill = 0L)

# 保证宽表的行顺序与全局 item_idx (1:num_items) 严格一一对应
# 这样就可以直接用 item_idx 作为行索引去矩阵里提取 genre 向量
dt_genres_full <- data.table(item_idx = all_items_vec)
dt_genres_full <- merge(dt_genres_full, dt_genres_wide, by = "item_idx", all.x = TRUE)
# 对冷启动/无类型的电影用 0 填充
for(j in names(dt_genres_full)) set(dt_genres_full, which(is.na(dt_genres_full[[j]])), j, 0L)

# 剥离 item_idx 列，转换为纯 R 的 Matrix，方便后续切片喂给 Torch
genre_matrix_global <- as.matrix(dt_genres_full[, -1, with = FALSE])
num_genres <- ncol(genre_matrix_global)

# --- C. 将特征 Join 到训练集和评估集中 ---

# 将 user 画像拼接到 dataset 中
dt_train[dt_users, on = "user_idx", `:=`(gender_idx = i.gender_idx, age_idx = i.age_idx, occ_idx = i.occ_idx)]
dt_eval[dt_users, on = "user_idx",  `:=`(gender_idx = i.gender_idx, age_idx = i.age_idx, occ_idx = i.occ_idx)]

# 提取对齐好行数的 Genre Matrix
# 因为 dt_train$item_idx 是一个整数向量，我们可以直接用它对全局矩阵进行按行切片
train_genre_matrix <- genre_matrix_global[dt_train$item_idx, ]
eval_genre_matrix  <- genre_matrix_global[dt_eval$item_idx, ]

cat(sprintf("=> 特征处理完毕: Gender(%d), Age(%d), Occ(%d), Genres(%d)\n", 
            num_genders, num_ages, num_occs, num_genres))

ds_dual_tower_feat <- dataset(
  name = "DualTowerFeatureDataset",
  
  initialize = function(dt, genre_mat) {
    # 基础 ID 和 Label
    self$users  <- torch_tensor(dt$user_idx, dtype = torch_long())
    self$items  <- torch_tensor(dt$item_idx, dtype = torch_long())
    self$labels <- torch_tensor(dt$label, dtype = torch_float())
    
    # 引入的用户画像特征
    self$genders <- torch_tensor(dt$gender_idx, dtype = torch_long())
    self$ages    <- torch_tensor(dt$age_idx, dtype = torch_long())
    self$occs    <- torch_tensor(dt$occ_idx, dtype = torch_long())
    
    # 引入的物品 Multi-hot 矩阵特征
    self$genres_multi_hot <- torch_tensor(genre_mat, dtype = torch_float())
  },
  
  .getbatch = function(indices) {
    # 统一打包成名为 x 的 list，这会直接传给 forward(x)
    x <- list(
      user            = self$users[indices],
      item            = self$items[indices],
      gender          = self$genders[indices],
      age             = self$ages[indices],
      occ             = self$occs[indices],
      genre_multi_hot = self$genres_multi_hot[indices]
    )
    list(x, self$labels[indices])
  },
  
  .length = function() {
    self$labels$size(1)
  }
)

# 实例化 DataLoader
dl_train_feat <- dataloader(
  ds_dual_tower_feat(dt_train, train_genre_matrix),
  batch_size = 1024 * 4,
  shuffle = TRUE,
  num_workers = 0
)

dl_eval_feat <- dataloader(
  ds_dual_tower_feat(dt_eval, eval_genre_matrix),
  batch_size = 2000,
  shuffle = FALSE,
  num_workers = 0
)

DualTower_WithFeatures <- nn_module(
  "DualTower_WithFeatures",
  
  initialize = function(num_users, num_items, 
                        num_genders, num_ages, num_occs, num_genres,
                        id_dim = 32, feat_dim = 8, latent_dim = 32) {
    
    # 1. Embeddings 初始化 (保持不变)
    self$user_embedding <- nn_embedding(num_users, id_dim)
    self$item_embedding <- nn_embedding(num_items, id_dim)
    self$gender_emb <- nn_embedding(num_genders, feat_dim)
    self$age_emb    <- nn_embedding(num_ages, feat_dim)
    self$occ_emb    <- nn_embedding(num_occs, feat_dim)
    self$genre_linear <- nn_linear(num_genres, feat_dim, bias = FALSE)
    
    user_input_dim <- id_dim + 3 * feat_dim
    item_input_dim <- id_dim + 1 * feat_dim
    
    # 2. 塔结构 (保持不变)
    self$user_tower <- nn_sequential(
      nn_linear(user_input_dim, 64), nn_relu(), nn_linear(64, latent_dim)      
    )
    self$item_tower <- nn_sequential(
      nn_linear(item_input_dim, 64), nn_relu(), nn_linear(64, latent_dim)
    )
    
    nn_init_normal_(self$user_embedding$weight, std = 0.01)
    nn_init_normal_(self$item_embedding$weight, std = 0.01)
  },
  
  # 工程部署接口：在线实时导出用户表征
  get_user_rep = function(user_tensor, gender_tensor, age_tensor, occ_tensor) {
    u_id  <- self$user_embedding(user_tensor)
    u_gen <- self$gender_emb(gender_tensor)
    u_age <- self$age_emb(age_tensor)
    u_occ <- self$occ_emb(occ_tensor)
    
    u_concat <- torch_cat(list(u_id, u_gen, u_age, u_occ), dim = -1L)
    return(self$user_tower(u_concat))
  },
  
  # 工程部署接口：离线批量导出物品表征
  get_item_rep = function(item_tensor, genre_multi_hot_tensor) {
    i_id  <- self$item_embedding(item_tensor)
    
    # 内置 Mean Pooling 逻辑
    genre_counts <- torch_clamp(genre_multi_hot_tensor$sum(dim = -1L, keepdim = TRUE), min = 1)
    normalized_genres <- genre_multi_hot_tensor / genre_counts
    i_gen <- self$genre_linear(normalized_genres) 
    
    i_concat <- torch_cat(list(i_id, i_gen), dim = -1L)
    return(self$item_tower(i_concat))
  },
  
  forward = function(x) {
    # 训练时的逻辑极其清爽，坚决不写重复代码
    u_rep <- self$get_user_rep(x$user, x$gender, x$age, x$occ)
    i_rep <- self$get_item_rep(x$item, x$genre_multi_hot)
    
    return(torch_sum(u_rep * i_rep, dim = -1L))
  }
)

# Luz 训练流水线 (带特征融合版)
fitted_model_feat <- DualTower_WithFeatures %>%
  setup(
    loss = nnf_binary_cross_entropy_with_logits, 
    optimizer = optim_adam
  ) %>%
  set_hparams(
    num_users   = num_users,
    num_items   = num_items,
    num_genders = num_genders,
    num_ages    = num_ages,
    num_occs    = num_occs,
    num_genres  = num_genres,
    id_dim      = 32, 
    feat_dim    = 8,   
    latent_dim  = 32   
  ) %>%
  set_opt_hparams(
    lr = 0.005,
    weight_decay = 1e-5
  ) %>%
  fit(
    data        = dl_train_feat,   
    valid_data  = dl_eval_feat,    
    epochs      = 15,
    accelerator = accelerator()
  )

# plot(fitted_model_feat)
recall_score_feat <- evaluate_recall(fitted_model_feat, dl_eval_feat, k = 10)
cat(sprintf("=> 带特征双塔 Recall@10: %.4f\n", recall_score_feat))

# ==============================================================================
# 表征批量导出 (特征双塔版)
# ==============================================================================

# 提取底层原生 Torch 模型，并设置为评估模式
device = torch_device('cpu')
model_core <- fitted_model_feat$model$to(device = device)
model_core$eval()

# --- A. 导出 Item 全量表征 ---
# 你的 genre_matrix_global 在前面的逻辑中已经完美对齐了 1:num_items
item_ids <- torch_tensor(all_items_vec, dtype = torch_long())
item_genres <- torch_tensor(genre_matrix_global, dtype = torch_float())

with_no_grad({
  item_embeddings <- model_core$get_item_rep(item_ids, item_genres)
})

# 转换为标准 R 矩阵，并将行名绑定为真实的电影业务 ID
item_embeddings_matrix <- as_array(item_embeddings)
rownames(item_embeddings_matrix) <- as.character(unique_items)

# --- B. 导出 User 全量表征 ---
# 注意细节：前面清洗 dt_users 时，你剔除了缺失用户。
# 为了保证导出的矩阵严格等于 num_users 行，我们需要左连接补齐，防止 Tensor 维度报错。
dt_users_full <- data.table(user_idx = 1:num_users)
dt_users_full <- merge(dt_users_full, dt_users, by = "user_idx", all.x = TRUE)

# 对于冷启动用户（在 user.dat 中无记录），赋予一个默认类别索引 (R 的 Embedding 索引必须大于 0)
# 这里用 1L 进行占位填充
for(col in c("gender_idx", "age_idx", "occ_idx")) {
  set(dt_users_full, which(is.na(dt_users_full[[col]])), col, 1L)
}

user_ids_tensor <- torch_tensor(dt_users_full$user_idx, dtype = torch_long())
user_genders_tensor <- torch_tensor(dt_users_full$gender_idx, dtype = torch_long())
user_ages_tensor <- torch_tensor(dt_users_full$age_idx, dtype = torch_long())
user_occs_tensor <- torch_tensor(dt_users_full$occ_idx, dtype = torch_long())

with_no_grad({
  user_embeddings_tensor <- model_core$get_user_rep(
    user_ids_tensor, 
    user_genders_tensor, 
    user_ages_tensor, 
    user_occs_tensor
  )
})

# 转换为标准 R 矩阵，并将行名绑定为真实的用户业务 ID
user_embeddings_matrix <- as_array(user_embeddings_tensor)
rownames(user_embeddings_matrix) <- as.character(unique_users)

# ------------------------------------------------------------------------------
# 结果验证
# ------------------------------------------------------------------------------
cat(sprintf("=> Item 表征矩阵维度: [%d, %d]\n", nrow(item_embeddings_matrix), ncol(item_embeddings_matrix)))
cat(sprintf("=> User 表征矩阵维度: [%d, %d]\n", nrow(user_embeddings_matrix), ncol(user_embeddings_matrix)))

# 原始业务 movieId = '1' (Toy Story) 的前 5 维融合特征向量
print(item_embeddings_matrix["1", 1:5])


