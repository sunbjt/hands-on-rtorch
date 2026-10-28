# 强制使用 x86_64 架构（因 LibTorch 是 x86）
FROM --platform=linux/amd64 rocker/r-ver:4.5.2

# 按需安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libgit2-dev \
    libmagick++-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# R 包安装
RUN R -e " \
    pkgs <- c('rlang', 'ellipsis', 'coro', 'jsonlite', \
             'magick', 'RestRserve', 'base64enc', \
             'plumber', 'curl', 'torch', 'torchvision'); \
    install.packages(pkgs, dependencies = TRUE) \
  "

# 安装 torch 环境
RUN R -e "torch::install_torch()"

# 复制应用代码
WORKDIR /app
COPY server_RestRserve.R utils.R label_list.json /app/

EXPOSE 8080
CMD ["Rscript", "server_RestRserve.R"]