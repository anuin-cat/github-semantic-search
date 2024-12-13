import os
import json
import asyncio
import requests

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 获取当前目录根路径
current_file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(current_file_path)
data_path = os.path.join(root_path, "data_simple") 
db_path = os.path.join(root_path, "database", "init")

# 1. 根据 star 数量区间获取 GitHub 仓库，同时根据 star 数量从多到少排序（闭区间）并保存 GitHub 仓库
def get_top_repo_by_star(per_page=1000, page=1, min_star_num=0, max_star_num=500000):
    query = f'stars:{min_star_num}..{max_star_num} pushed:>2021-01-01'
    sort = 'stars'
    order = 'desc'
    search_url = f'{os.getenv('GITHUB_API_URL')}/search/repositories?q={query}&sort={sort}&order={order}&per_page={per_page}&page={page}'
    headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}

    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        total_count = response.json()['total_count']
        total_page = total_count // per_page + 1
        print(f"Total page: {total_page}, current page: {page}")
        if response.json()['incomplete_results']: print("Incomplete results")
        return response.json()['items'], response.json()['items'][-1]['stargazers_count'], total_count
    else:
        print(f"Failed to retrieve repositories: {response.status_code}")
        print("")
        # 直接退出
        exit(1)

def save_repo_by_star(max_star=500000):
    # github 限制每次请求最多得到 100 个仓库，因此 page 固定为 1
    top_repositories, max_star, count = get_top_repo_by_star(per_page=1000, page=1, min_star_num=1000, max_star_num=max_star)

    for i, repo in enumerate(top_repositories):
        owner = repo['owner']['login']
        name = repo['name']
        unique_id = f"{name} -- {owner}"
        stars = repo['stargazers_count']
        print(f"Repository {i}: {name}, Stars: {stars}")

        # 存储为 json 格式
        with open(os.path.join(data_path, f'{unique_id}.json'), 'w') as f:
            json.dump(repo, f, indent=4)

    if count < 100: exit(1)

    return max_star

def main_repo():
    max_star = 500000 # 最多 star 的仓库有 500k
    num = 1
    while True:
        print("=" * 50)
        print(f"Round {num}, Max star: {max_star}")
        max_star = save_repo_by_star(max_star)
        num += 1

# 2. 将数据转换为向量
async def create_vector_db(docs, embeddings, batch_size=800):
    # 初始化第一批数据
    vector_db = await FAISS.afrom_documents(docs[0:batch_size], embeddings)
    if len(docs) < batch_size: return vector_db
    
    # 创建任务x``
    tasks = []
    for start_idx in range(batch_size, len(docs), batch_size):
        end_idx = min(start_idx + batch_size, len(docs))
        tasks.append(FAISS.afrom_documents(docs[start_idx:end_idx], embeddings))

    # 执行任务
    results = await asyncio.gather(*tasks)

    # 合并结果
    for temp_db in results:
        vector_db.merge_from(temp_db)
    return vector_db

async def main_convert_to_vector():
    # 读取文件
    files = os.listdir(data_path)

    # 构建 document
    docs = []
    for file in tqdm(files):
        if not file.endswith(".json"): continue
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        content_map = {
            "name": data["name"],
            "description": data["description"],
        }
        content = json.dumps(content_map)
        doc = Document(page_content=content, metadata={"html_url": data["html_url"], 
                                                    "topics": data["topics"],
                                                    "created_at": data["created_at"],
                                                    "updated_at": data["updated_at"],
                                                    "star_count": data["stargazers_count"]})
        docs.append(doc)
    print(f"Total {len(docs)} documents.")

    # 初始化 Embedding 实例
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),
                                  base_url=os.getenv("OPENAI_BASE_URL"),
                                  model="text-embedding-3-small")
    print("Embedding model success: text-embedding-3-small")

    # 文档嵌入
    if os.path.exists(os.path.join(db_path, "init.faiss")):
        vector_db = FAISS.load_local(db_path, embeddings=embeddings,
                                        index_name="init",
                                        allow_dangerous_deserialization=True)
    else:
        vector_db = await create_vector_db(docs, embeddings=embeddings)
        vector_db.save_local(db_path, index_name="init")
    return vector_db

if __name__ == "__main__":
    # 1. 获取仓库信息
    # main_repo()

    # 2. 构建向量数据库
    asyncio.run(main_convert_to_vector())
