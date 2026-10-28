# 文件名必须保存为: app.R
library(shiny)
library(httr)
library(jsonlite)
library(base64enc)

# --- UI 定义 ---
ui <- fluidPage(
  titlePanel("视觉模型解释器 (Grad-CAM)"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("upload", "上传一张图片 (jpg/png)", accept = c("image/jpeg", "image/png")),
      actionButton("analyze", "开始分析", class = "btn-primary"),
      hr(),
      helpText("提示：请确保后台 Plumber 服务已在 8000 端口启动。")
    ),
    
    mainPanel(
      fluidRow(
        column(6, 
               h4("原始图片"),
               uiOutput("orig_ui")
        ),
        column(6, 
               h4("模型注意力 (Grad-CAM)"),
               uiOutput("cam_ui")
        )
      ),
      hr(),
      h4(textOutput("pred_text")) #稍微加大字体
    )
  )
)

# --- Server 定义 ---
server <- function(input, output, session) {
  
  # 辅助函数：将本地文件转为 base64 用于前端预览
  base64uri <- function(file) {
    paste0("data:image/png;base64,", base64encode(file))
  }
  
  # 响应分析按钮
  api_response <- eventReactive(input$analyze, {
    req(input$upload)
      
      # 构造上传请求
      # input$upload$datapath 是 Shiny 保存的临时文件路径
      resp <- tryCatch({
        POST(
          url = "http://127.0.0.1:8000/explain",
          body = list(image = upload_file(input$upload$datapath)),
          encode = "multipart"
        )
      }, error = function(e) {
        showNotification(paste("连接 API 失败:", e$message), type = "error")
        return(NULL)
      })
      
      content(resp, as = "parsed")
  })
  
  # 展示原始图片
  output$orig_ui <- renderUI({
    req(input$upload)
    tags$img(src = base64uri(input$upload$datapath), width = "100%")
  })
  
  # 展示 Grad-CAM 图片
  output$cam_ui <- renderUI({
    res <- api_response()
    req(res)
    # 构造 Base64 图片标签
    tags$img(
      src = paste0("data:image/png;base64,", 
                   res$image_base64), 
      width = "100%")
  })
  
  # 展示预测类别
  output$pred_text <- renderText({
    res <- api_response()
    req(res)
    paste("该图片的预测类别是:", res$class_idx)
  })
}

# 启动应用
shinyApp(ui, server)