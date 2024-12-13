# github-semantic-search
基于向量匹配及 LLM 二次过滤的 Github 仓库搜索工具：拒绝重复造轮子，快速找到已有高质量仓库

Vector Matching and LLM-Based Secondary Filtering for GitHub Repository Search: Avoiding Reinvention and Rapidly Identifying High-Quality Existing Repositories

## 使用

### 1. 在线使用

访问 [github-semantic-search](https://huggingface.co/spaces/Aniun/github-semantic-search)

### 2. 本地运行
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 获取 github 仓库数据 + 向量化存储
python deal_data.py

# 3. 运行聊天界面
python chat.py
```



## 功能

- 基于向量匹配的 Github 仓库搜索
- 基于 LLM 的仓库二次过滤
- 基于 LLM 的仓库关键词扩展
- 基于 LLM 的仓库描述生成
