# 相关词

## 部署步骤
1. 拉取镜像
    ```
    docker pull harbor.laibokeji.com/aiserver/semantic-search-and-keyword:v1.0
    ```
2. 启动docker容器 (默认进入容器，退出会自动停止运行)
   ```
   docker run --gpus all --shm-size 4g --name keyword-prompt --net host -it -v [共享目录]:/data harbor.laibokeji.com/aiserver/semantic-search-and-keyword:v1.0 bash
   ```
3. 进入docker内的/data目录,拉取本项目代码
   ```
   cd /data
   git clone https://gitee.com/whLaibo/keyword-extraction.git
   ```
4. 进入本项目，执行命令：
    ```
    bash sbin/start_nginx.sh # 启动nginx
    bash sbin/start_keyword.sh # 启动前缀树与聚合搜索
    ```
   
## 其他配置
    nginx配置文件：keyword-extraction/config/ngconf/nginx.conf
    gunicorn配置文件(路径)：keyword-extraction/sbin/start_search.sh
    gpu选择、模型数据路径： keyword-extraction/config/project_config.py


## 主要逻辑

### 相关词推荐
    1. 首先使用预先设置的模版，将关键字变成标题（bert->基于bert的研究) 
    2.  作为原词的相关词

### 关键词的提取
    1. 优先级 正文->章节标题->论文题目
    2. 正文使用 jieba+词频统计+tfidf+专业词库 提取关键词
    3. 章节标题、论文题使用unlim模型生成对应的关键词

