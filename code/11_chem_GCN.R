# install.packages("rcdk")

# 观察分子式的结构和 smiles 格式的关系

library(rcdk)

# 1. 挑选 ESOL 数据集里的几个经典分子 SMILES
smiles_list <- c(
  "c1ccsc1",                   
  "O=C=O",
  "CC(C)=CCCC(C)=CC=O",  # 或者是 "CC(C)=CCC/C(C)=C/C=O"
  "c1ccccc1O",   # 苯酚
  "CCO",
  "CC(=O)O"
)
names_list <- c("噻吩", "二氧化碳", "柠檬醛", "苯酚", "乙醇", "乙酸")
formulas_list <- c("C4H4S", "CO2", "C10H16O", "C6H6O", "C2H6O", "C2H4O2")

par(mfrow = c(2, 3), mar = c(0, 0, 5, 0))  # 顶部留空间
dep <- get.depictor(width = 300, height = 200, style = "cow", zoom = 2)
for (i in seq_along(smiles_list)) {
  mol <- parse.smiles(smiles_list[i])[[1]]
  img <- view.image.2d(mol, depictor = dep)
  
  # 拼接标题：中文名 + 分子式 + SMILES
  title_text <- sprintf("%s\n(%s)\nSMILES: %s", 
                        names_list[i], formulas_list[i], smiles_list[i])
  
  plot(c(0, 1), c(0, 1), type = "n", axes = FALSE, xlab = "", ylab = "", 
       main = title_text,
       cex.main = 1.0,
       col.main = "#333333",
       font.main = 1,
       asp = 1)
  
  rasterImage(img, 0, 0, 1, 1)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

# --- 2. 构建 R6 Dataset ---
library(torch)
library(luz)
library(tidyverse)
library(rcdk)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

# ==========================================
# 1. 终极版特征提取函数 (微观图特征 + 宏观物理特征)
# ==========================================
get_mol_graph_rcdk <- function(smiles, max_atoms = 55) {
  mol_list <- suppressWarnings(parse.smiles(smiles))
  if (length(mol_list) == 0 || is.null(mol_list[[1]])) return(NULL)
  
  mol <- mol_list[[1]]
  atoms <- get.atoms(mol)
  n_atoms <- length(atoms)
  
  if (n_atoms == 0 || n_atoms > max_atoms) return(NULL)
  
  # --- A. 扩展节点特征 ---
  allowed_elements <- c(6L, 7L, 8L, 9L, 15L, 16L, 17L, 35L, 53L)
  x_base <- matrix(0, nrow = max_atoms, ncol = 18)
  
  for (i in seq_len(n_atoms)) {
    a <- atoms[[i]]
    
    # 1-10维: 元素 One-hot
    atomic_num <- get.atomic.number(a)
    idx <- match(atomic_num, allowed_elements)
    if (is.na(idx)) idx <- length(allowed_elements) + 1
    x_base[i, idx] <- 1
    
    # 11维: 形式电荷
    charge <- tryCatch(get.formal.charge(a), error = function(e) 0)
    if (is.null(charge) || is.na(charge)) charge <- 0
    x_base[i, 11] <- charge
    
    # 12维: 氢原子连接数
    h_count <- tryCatch(get.hydrogen.count(a), error = function(e) 0)
    if (is.null(h_count) || is.na(h_count)) h_count <- 0
    x_base[i, 12] <- h_count
    
    # 13维: 原子度数 (后面统一赋值)
    
    # 14-16维: 杂化态 (sp, sp2, sp3)
    hyb <- tryCatch(get.hybridization(a), error = function(e) NA)
    hyb_idx <- match(hyb, c("SP1", "SP2", "SP3"))
    if (!is.na(hyb_idx)) x_base[i, 13 + hyb_idx] <- 1
    
    # 17维: 是否在环中
    in_ring <- tryCatch(is.in.ring(a), error = function(e) FALSE)
    x_base[i, 17] <- ifelse(in_ring, 1, 0)
    
    # 18维: 芳香性
    aromatic <- tryCatch(is.aromatic(a), error = function(e) FALSE)
    x_base[i, 18] <- ifelse(aromatic, 1, 0)
  }
  
  # 邻接矩阵
  adj_raw <- get.adjacency.matrix(mol)
  
  # 13维: 原子度数
  degree_vec <- rowSums(adj_raw)
  x_base[1:n_atoms, 13] <- degree_vec
  
  # 拼装最终特征矩阵
  x <- x_base
  
  # 邻接矩阵 + 自环
  adj <- matrix(0, nrow = max_atoms, ncol = max_atoms)
  adj[1:n_atoms, 1:n_atoms] <- adj_raw
  diag(adj)[1:n_atoms] <- 1
  
  mask <- rep(0, max_atoms)
  mask[1:n_atoms] <- 1
  
  # 宏观特征
  mass <- tryCatch(get.exact.mass(mol), error = function(e) 0)
  if (is.null(mass) || is.na(mass)) mass <- 0
  
  tpsa <- tryCatch(get.tpsa(mol), error = function(e) 0)
  if (is.null(tpsa) || is.na(tpsa)) tpsa <- 0
  
  global_features <- c(mass, tpsa)
  
  list(x = x, adj = adj, mask = mask, global_x = global_features)
}

# ==========================================
# 2. Dataset 类 (适配 luz 多输入 & 物理特征归一化)
# ==========================================
ESOLDataset <- dataset(
  name = "ESOLDataset",
  
  initialize = function(csv_path, max_atoms = 55) {
    df <- read_csv(csv_path, show_col_types = FALSE)
    
    self$data <- list()
    self$y <- numeric()
    
    cat("Parsing SMILES with rcdk and building dual-stream graphs...\n")
    pb <- txtProgressBar(min = 0, max = nrow(df), style = 3)
    
    valid_idx <- 1
    for (i in 1:nrow(df)) {
      smiles <- df$smiles[i]
      # 读取目标值：溶解度 LogS
      y_val <- df$`measured log solubility in mols per litre`[i] 
      
      graph <- get_mol_graph_rcdk(smiles, max_atoms)
      
      if (!is.null(graph)) {
        self$data[[valid_idx]] <- graph
        self$y[valid_idx] <- y_val
        valid_idx <- valid_idx + 1
      }
      setTxtProgressBar(pb, i)
    }
    close(pb)
  },
  
  .getitem = function(i) {
    item <- self$data[[i]]
    
    # === 宏观特征粗略归一化 (极其重要，防止梯度爆炸) ===
    # 分子量通常 < 1000，TPSA 通常 < 300，除以常数将其缩放到 0~1 左右
    norm_mass <- item$global_x[1] / 1000.0  
    norm_tpsa <- item$global_x[2] / 300.0
    
    # luz 期望返回: list( inputs_list, target_tensor )
    list(
      list(
        x = torch_tensor(item$x, dtype = torch_float()),
        adj = torch_tensor(item$adj, dtype = torch_float()),
        mask = torch_tensor(item$mask, dtype = torch_float()),
        # 将归一化后的物理特征也作为输入传给模型
        global_x = torch_tensor(c(norm_mass, norm_tpsa), dtype = torch_float())
      ),
      # 目标张量
      torch_tensor(self$y[i], dtype = torch_float())
    )
  },
  
  .length = function() {
    length(self$y)
  }
)


# 支持 Batch 处理且带有度归一化 (Degree Normalization) 的 GCN 层
BatchedGCNLayer <- nn_module(
  "BatchedGCNLayer",
  initialize = function(in_features, out_features) {
    self$W <- nn_linear(in_features, out_features, bias = FALSE)
    # 增加偏置项，增强每一层特征的表达能力
    self$bias <- nn_parameter(torch_zeros(out_features)) 
  },
  forward = function(h, adj) {
    # h 形状: [Batch, N, in_features]
    # adj 形状: [Batch, N, N]
    
    # === 计算度矩阵并进行归一化 ===
    # 1. 计算每个原子的度数 (沿行求和)，形状变为 [Batch, N, 1]
    degree <- adj$sum(dim = 3, keepdim = TRUE)
    
    # 2. 计算 D^-1 (加上一个极小值防止除以 0)
    degree_inv <- 1.0 / torch_clamp(degree, min = 1e-5)
    
    # 3. 利用广播机制，将邻接矩阵的每一行除以该行的度数
    # [Batch, N, N] * [Batch, N, 1] -> [Batch, N, N]
    adj_norm <- adj * degree_inv 
    
    # === 正常的图卷积操作 ===
    # 特征变换
    support <- self$W(h) 
    
    # 使用归一化后的 adj_norm 进行邻居聚合，并加上 Bias
    output <- torch_bmm(adj_norm, support) + self$bias
    
    return(output)
  }
)


# 补充缺失的双重池化层 (Mean + Max)
MaskedDualReadout <- nn_module(
  "MaskedDualReadout",
  initialize = function() {},
  forward = function(h, mask) {
    mask_expanded <- mask$unsqueeze(3)$expand_as(h)
    h_masked <- h * mask_expanded
    
    # --- Mean Pooling ---
    sum_h <- h_masked$sum(dim = 2) 
    num_atoms <- mask$sum(dim = 2, keepdim = TRUE)$clamp(min = 1)
    mean_h <- sum_h / num_atoms
    
    # --- Max Pooling ---
    max_h <- h_masked$max(dim = 2)[[1]]
    
    # 将 Mean 和 Max 拼接 (维度翻倍)
    return(torch_cat(list(mean_h, max_h), dim = 2))
  }
)

GraphPredictor <- nn_module(
  "GraphPredictor",
  initialize = function(in_dim, hidden_dim, num_global_features = 2) {
    self$gcn1 <- BatchedGCNLayer(in_dim, hidden_dim)
    self$gcn2 <- BatchedGCNLayer(hidden_dim, hidden_dim)
    self$gcn3 <- BatchedGCNLayer(hidden_dim, hidden_dim)
    self$readout <- MaskedDualReadout() # 依然使用之前写好的双重池化
    
    # 图向量维度: hidden_dim * 2 (双重池化)
    graph_emb_dim <- hidden_dim * 2
    
    # === 拼接后的总维度 ===
    total_dim <- graph_emb_dim + num_global_features
    
    # 对拼接后的向量进行整体 layerNorm
    self$bn <- nn_layer_norm(total_dim)
    
    # FC 层接收拼接后的长向量
    self$fc1 <- nn_linear(total_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, 1)
    
    self$relu <- nn_relu()
    self$dropout <- nn_dropout(0.2)
  },
  
  forward = function(inputs) {
    x <- inputs$x
    adj <- inputs$adj
    mask <- inputs$mask
    global_x <- inputs$global_x # [Batch, 2]
    
    # 1. 走图卷积分支
    h <- self$relu(self$gcn1(x, adj))
    h <- self$relu(self$gcn2(h, adj))
    h <- self$relu(self$gcn3(h, adj))
    h_graph <- self$readout(h, mask) # [Batch, hidden_dim * 2]
    
    # 2. === 核心：特征融合 (Late Fusion) ===
    # 沿着特征维度 (dim=2) 拼接图向量和全局物理特征
    h_combined <- torch_cat(list(h_graph, global_x), dim = 2) # [Batch, total_dim]
    
    # 3. 走预测分支
    h_combined <- self$bn(h_combined) 
    out <- self$relu(self$fc1(h_combined))
    out <- self$dropout(out)
    out <- self$fc2(out) 
    
    return(out)
  }
)

# 实例化数据集与切分
csv_path <- '/Users/liusizhe/data/delaney-processed.csv' 
esol_data <- ESOLDataset(csv_path)

set.seed(42)
train_indices <- sample(1:length(esol_data), size = 0.8 * length(esol_data))
valid_indices <- setdiff(1:length(esol_data), train_indices)

train_ds <- dataset_subset(esol_data, train_indices)
valid_ds <- dataset_subset(esol_data, valid_indices)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 64, shuffle = FALSE)

# ==========================================
# 使用 Luz 搭建流水线并训练
# ==========================================
fitted_model <- GraphPredictor |> 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mse())
  ) |> 
  set_hparams(in_dim = 18, hidden_dim = 64) |> 
  set_opt_hparams(lr = 0.001, weight_decay = 1e-4) |>
  fit(
    data = train_dl,
    valid_data = valid_dl,
    epochs = 150,                  
    callbacks = list(
      luz_callback_lr_scheduler(
        lr_reduce_on_plateau,
        patience = 10,
        factor = 0.5,
        min_lr = 1e-5
      ),
      # 早停机制依然保留 monitor
      luz_callback_early_stopping(
        monitor = "valid_loss",
        patience = 25
      )
    )
  )

plot(fitted_model)



# ==========================================
# 模型推断函数 (双流特征 + 适配 Mac MPS)
# ==========================================
predict_solubility <- function(model, smiles_string) {
  graph <- get_mol_graph_rcdk(smiles_string)
  device <- model$parameters[[1]]$device
  
  # 推断时也要进行同样的特征归一化
  norm_mass <- graph$global_x[1] / 1000.0  
  norm_tpsa <- graph$global_x[2] / 300.0
  
  # 把全局特征转为 Tensor，增加 Batch 维度并推送到 MPS 设备
  global_tensor <- torch_tensor(c(norm_mass, norm_tpsa), dtype = torch_float())$
    unsqueeze(1)$
    to(device = device)
  
  inputs <- list(
    x = torch_tensor(graph$x, dtype = torch_float())$unsqueeze(1)$to(device = device),
    adj = torch_tensor(graph$adj, dtype = torch_float())$unsqueeze(1)$to(device = device),
    mask = torch_tensor(graph$mask, dtype = torch_float())$unsqueeze(1)$to(device = device),
    # 将缺失的全局物理特征传给模型
    global_x = global_tensor
  )
  
  model$eval()
  with_no_grad({
    pred <- model(inputs)
  })
  
  return(as.numeric(pred$cpu()))
}

test_smiles <- c(
  "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 布洛芬 (难溶于水)
  "C1=NC2=C(N1)C(=O)NC(=O)N2",      # 尿酸 (极难溶于水)
  "C(C(C(C(C(CO)O)O)O)O)O",         # 甘露醇 (极易溶于水)
  "CCO"                             # 乙醇 (与水互溶)
)

names <- c("布洛芬", "尿酸", "甘露醇", "乙醇")
for (i in seq_along(test_smiles)) {
  pred_val <- predict_solubility(fitted_model$model, test_smiles[i])
  cat(sprintf("%-10s : %.2f\n", names[i], pred_val))
}

