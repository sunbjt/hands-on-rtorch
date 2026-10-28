library(vroom)
library(data.table)
library(torch)
library(luz)
set.seed(42)
options(luz.force_progress_bar = TRUE) # 在 positron 编辑器中正确输出动态进度条

#DATA_DIR <- "~/data/ml-10m100K"
DATA_DIR <- "~/data/ml-1m"
FILE_PATH <- file.path(DATA_DIR, "ratings.dat")

dt_ratings <- vroom(
  FILE_PATH,
  delim = "::",
  col_names = c("user_id", "movie_id", "rating", "timestamp"),
  col_select = c(user_id, movie_id, rating, timestamp),
  show_col_types = FALSE
)
setDT(dt_ratings)

# 提取唯一 ID 并计算维度
unique_users <- unique(dt_ratings$user_id)
unique_items <- unique(dt_ratings$movie_id)
num_users <- length(unique_users)
num_items <- length(unique_items)
all_items_vec <- 1:num_items

# 映射为从 1 开始的连续整数索引 (Torch Embedding 强制要求)
dt_ratings[, user_idx := match(user_id, unique_users)]
dt_ratings[, item_idx := match(movie_id, unique_items)]

# 严谨的时序划分 (80% Train, 20% Test)
setorder(dt_ratings, user_idx, timestamp)
dt_ratings[, `:=`(seq_num = 1:.N, total_num = .N), by = user_idx]

# 按照用户观影的时间顺序，预留 20% 做 test 数据集。
dt_train_raw <- dt_ratings[seq_num <= floor(0.8 * total_num)]
dt_test_raw  <- dt_ratings[seq_num > floor(0.8 * total_num)]

# 过滤测试集中的冷启动物品
valid_train_items <- unique(dt_train_raw$item_idx)
dt_test_raw <- dt_test_raw[item_idx %in% valid_train_items]

# ==========================================
# 3. 构造正负样本 (核心评估基准)
# ==========================================

# 预先计算每个用户的全局交互历史，用于负采样时避开已交互物品
dt_user_history <- dt_ratings[, .(interacted = list(item_idx)), by = user_idx]

# 构造评估集 (严格留一法: 1 正 + 99 负) 
# 取测试集第一条作为目标正样本
dt_eval_pos <- dt_test_raw[, .SD[1], by = user_idx][, .(user_idx, item_idx, label = 1)]

# 仅为有效的评估用户生成 99 个负样本
valid_eval_users <- dt_eval_pos$user_idx
dt_eval_neg <- dt_user_history[user_idx %in% valid_eval_users,
                               .(item_idx = sample(
                                 setdiff(
                                   all_items_vec, unlist(interacted)
                                   ), 99, replace = FALSE),
                                      label = 0), by = user_idx]

dt_eval <- rbindlist(list(dt_eval_pos, dt_eval_neg), use.names = TRUE)
setorder(dt_eval, user_idx, -label) # 保证每个用户 100 行连续排列

# --- 3.2 构造训练集 (1:1 正负样本均衡，修复 Loss 爆炸问题) ---
dt_train_pos <- dt_train_raw[, .(user_idx, item_idx, label = 1)]
train_counts <- dt_train_pos[, .(pos_count = .N), by = user_idx]

# 按每个用户的正样本数量，抽取等量的训练负样本
dt_train_neg <-
  train_counts[, .(item_idx =
                     sample(setdiff(all_items_vec,
                                    dt_user_history[user_idx == .BY$user_idx, unlist(interacted)]),
                            pos_count, replace = TRUE),
                   label = 0), by = user_idx]

dt_train <- rbindlist(list(dt_train_pos, dt_train_neg), use.names = TRUE)
dt_train <- dt_train[sample(.N)] # 随机打乱训练集

