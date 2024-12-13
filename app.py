import os
import time
import json
import asyncio
import gradio as gr

# set the env
from dotenv import load_dotenv
load_dotenv()

# get the root path of the project
current_file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(current_file_path)

from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

class OurLLM:
    def __init__(self, model="gpt-4o"):
        '''
        params: 
            model: str, 
                模型名称 ["GLM-4-Flash", "GLM-4V-Flash", 
                         "gpt-4o-mini", "gpt-4o", "o1-mini", 
                         "gemini-1.5-flash-002", "gemini-1.5-pro-002",
                         "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]
        '''

        self.model_name = model

        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        OPENAI_API_KEY_DF = os.getenv('OPENAI_API_KEY_DF', OPENAI_API_KEY)
        OPENAI_API_KEY_AZ = os.getenv('OPENAI_API_KEY_AZ', OPENAI_API_KEY)
        OPENAI_API_KEY_CD = os.getenv('OPENAI_API_KEY_CD')
        OPENAI_API_KEY_O1 = os.getenv('OPENAI_API_KEY_O1')
        OPENAI_API_KEY_GLM = os.getenv('OPENAI_API_KEY_GLM')
        OPENAI_API_KEY_SC = os.getenv('OPENAI_API_KEY_SC')

        OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
        OPENAI_BASE_URL_GLM = os.getenv('OPENAI_BASE_URL_GLM')
        OPENAI_BASE_URL_SC = os.getenv('OPENAI_BASE_URL_SC')

        # 创建 API Key 映射
        apiKeyMap = {
            'gemini': {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY_DF},
            'gpt': {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY_AZ},
            'o1': {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY_O1},
            'claude': {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY_CD},
            'glm': {"base_url": OPENAI_BASE_URL_GLM, "api_key": OPENAI_API_KEY_GLM},
            'qwen': {"base_url": OPENAI_BASE_URL_SC, "api_key": OPENAI_API_KEY_SC},
        }

        for name, info in apiKeyMap.items():
            if name in model.lower():
                self.base_url = info["base_url"]
                self.api_key = info["api_key"]
                break
        assert self.base_url is not None, f"Base URL not found for model: {model}"
        assert self.api_key is not None, f"API key not found for model: {model}"

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{input}"),
                # ("ai", "{chat_history}"),
            ]
        )
        self.chat_prompt = chat_prompt
        self.llm = self.get_llm(model)

    def clean_json(self, s):
        return s.replace("```json", "").replace("```", "").strip()

    def get_system_prompt(self, mode="assistant"):
        prompt_map = {
            "assistant": dedent("""
                你是一个智能助手，擅长用简洁的中文回答用户的问题。
                请确保你的回答准确、清晰、有条理，并且符合中文的语言习惯。
                重要提示：
                1. 回答要简洁明了，避免冗长
                2. 使用适当的专业术语
                3. 保持客观中立的语气
                4. 如果不确定，要明确指出
            """),
            # search
            "keyword_expand": dedent("""
                你是一个搜索关键词扩展专家，擅长将用户的搜索意图转化为多个相关的搜索词或短语。
                用户会输入一段描述他们搜索需求的文本，请你生成与之相关的关键词列表。
                你需要返回一个可以直接被 json 库解析的响应，包含以下内容：
                {
                    "keywords": [关键词列表],
                }
                重要提示：
                1. 关键词应该包含同义词、近义词、上位词、下位词
                2. 短语要体现不同的表达方式和组合
                3. 描述句子要涵盖不同的应用场景和用途
                4. 所有内容必须与原始搜索意图高度相关
                5. 扩展搜索意图到相关的应用场景和工具，例如:
                    - 如果搜索"PDF转MD"，应包含PDF内容提取、PDF解析工具、PDF数据处理等
                    - 如果搜索"图片压缩"，应包含批量压缩工具、图片格式转换等
                    - 如果搜索"代码格式化"，应包含代码美化工具、语法检查器、代码风格统一等
                    - 如果搜索"文本翻译"，应包含机器翻译API、多语言翻译工具、离线翻译软件等
                    - 如果搜索"数据可视化"，应包含图表生成工具、数据分析库、交互式图表等
                    - 如果搜索"网络爬虫"，应包含数据采集框架、反爬虫绕过、数据解析工具等
                    - 如果搜索"API测试"，应包含接口测试工具、性能监控、自动化测试框架等
                6. 所有内容主要使用英文表达，并对部分关键词添加额外的中文表示
                7. 返回内容不要使用任何 markdown 格式 以及任何特殊字符
            """),
            "zh2en": dedent("""
                你是一个专业的中译英翻译专家，尤其擅长学术论文的翻译工作。
                请将用户提供的中文内容翻译成地道、专业的英文。

                重要提示：
                1. 使用学术论文常用的表达方式和术语
                2. 保持专业、正式的语气
                3. 确保译文的准确性和流畅性
                4. 对专业术语进行准确翻译
                5. 遵循英文学术写作的语法规范
                6. 保持原文的逻辑结构
                7. 适当使用学术论文常见的过渡词和连接词
                8. 如遇到模糊的表达，选择最符合学术上下文的翻译
                9. 避免使用口语化或非正式的表达
                10. 注意时态和语态的准确使用
            """),
            "github_score": dedent("""
                你是一个语义匹配评分专家，擅长根据用户需求和仓库描述进行语义匹配度评分。
                用户会输入两部分内容:
                1. 用户的具体需求描述
                2. 多个仓库的描述列表(以1,2,3等数字开头)
                
                请你仔细分析用户需求，并对每个仓库进行评分。
                确保返回一个可以直接被 json 库解析的响应，包含以下内容：
                {
                    "indices": [仓库编号列表，按分数从高到低],
                    "scores": [编号对应的匹配度评分列表，0-100的整数，表示匹配程度]
                }
                
                重要提示：
                1. 评分范围为0-100的整数，高于60分表示具有明显相关性
                2. 评分要客观反映仓库与需求的契合度
                3. 只返回评分大于 60 的仓库
                4. 返回内容不要使用任何 markdown 格式 以及任何特殊字符
            """)
        }
        return prompt_map[mode]

    def get_llm(self, model="gpt-4o-mini"):
        '''
        params:
            model: str, 模型名称 ["gpt-4o-mini", "gpt-4o", "o1-mini", "gemini-1.5-flash-002"]
        '''
        llm = ChatOpenAI(
            model=model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        print(f"Init model {model} successfully!")
        return llm
    
    def ask_question(self, question, system_prompt=None):
        # 1. 获取系统提示
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        # 2. 生成聊天提示
        prompt = self.chat_prompt.format(input=question, system_prompt=system_prompt)
        config = {
            "configurable": {"response_format": {"type": "json_object"}}
        }
        
        # 3. 调用 LLM 进行回答
        for _ in range(10):
            try:
                response = self.llm.invoke(prompt, config=config)
                response.content = self.clean_json(response.content)
                return response
            except Exception as e:
                print(e)
                time.sleep(10)
                continue
        print(f"Failed to call llm for prompt: {prompt[0:10]}")
        return None
    
    async def ask_questions_parallel(self, questions, system_prompt=None):
        # 1. 获取系统提示
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # 2. 定义异步函数
        async def call_llm(prompt):
            for _ in range(10):
                try:
                    response = await self.llm.ainvoke(prompt)
                    response.content = self.clean_json(response.content)
                    return response
                except Exception as e:
                    print(e)
                    await asyncio.sleep(10)
                    continue
            print(f"Failed to call llm for prompt: {prompt[0:10]}")
            return None

        # 3. 构建 prompt
        prompts = [self.chat_prompt.format(input=question, system_prompt=system_prompt) for question in questions]

        # 4. 异步调用
        tasks = [call_llm(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        return results

class RepoSearch:
    def __init__(self):
        db_path = os.path.join(root_path, "database", "init")
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), 
                                      base_url=os.getenv("OPENAI_BASE_URL"),
                                      model="text-embedding-3-small")
        
        assert os.path.exists(db_path), f"Database not found: {db_path}"
        self.vector_db = FAISS.load_local(db_path, embeddings, 
                                          index_name="init",
                                          allow_dangerous_deserialization=True)
        
    def search(self, query, k=10):
        '''
            name + description + html_url + topics
        '''
        results = self.vector_db.similarity_search(query + " technology", k=k)

        simple_str = ""
        simple_list = []
        for i, doc in enumerate(results):
            content = json.loads(doc.page_content)
            metadata = doc.metadata
            if content["description"] is None:
                content["description"] = ""
            # desc = content["description"] if len(content["description"]) < 300 else content["description"][:300] + "..."
            simple_str += f"\t**{i+1}. {content['name']}** || {content['description']}\n" # 用于大模型匹配
            simple_list.append({
                "name": content["name"],
                "description": content["description"],
                **metadata,  # 解包所有 metadata 字段
            })

        return simple_str, simple_list

def main():
    search = RepoSearch()
    llm = OurLLM(model="gpt-4o")

    def respond(
        prompt: str,
        history,
        is_llm_filter: bool = False,
        is_keyword_expand: bool = False,
        match_num: int = 40
    ):
        # 1. 初始化历史记录
        if not history:
            history = [{"role": "system", "content": "You are a friendly chatbot"}]
        history.append({"role": "user", "content": prompt})
        response = {"role": "assistant", "content": ""}
        yield history

        # 2. 扩展用户问题关键词
        if is_keyword_expand:
            response["content"] = "开始扩展关键词..."
            yield history + [response]

            query = llm.ask_question(prompt, system_prompt=llm.get_system_prompt("keyword_expand")).content
            prompt = ", ".join(json.loads(query)["keywords"])

        # 3. 语义向量匹配
        response["content"] = "开始语义向量匹配..."
        yield history + [response]
        match_str, simple_list = search.search(prompt, match_num)

        # 4. 通过 LLM 评分得到最匹配的仓库索引
        if not is_llm_filter:
            simple_strs = [f"\t**{i+1}. {repo['name']}** [✨ {repo['star_count'] // 1000}k] || **Description:** {repo['description']} || **Url:** {repo['html_url']} \n" for i, repo in enumerate(simple_list)]
            response["content"] = "".join(simple_strs)
            yield history + [response]
        else:
            response["content"] = "开始通过 LLM 评分得到最匹配的仓库..."
            yield history + [response]

            query = ' ## 用户需要的仓库内容：' + prompt + '\n ## 搜索结果列表：' + match_str
            out = llm.ask_question(query, system_prompt=llm.get_system_prompt("github_score")).content
            matched_index = json.loads(out)["indices"]

            # 5. 通过索引得到最匹配的仓库
            result = [simple_list[idx-1] for idx in matched_index]
            simple_strs = [f"\t**{i+1}. {repo['name']}** [✨ {repo['star_count'] // 1000}k] || **Description:** {repo['description']} || **Url:** {repo['html_url']} \n" for i, repo in enumerate(result)]
            response["content"] = "".join(simple_strs)
            yield history + [response]

    with gr.Blocks() as demo:
        gr.Markdown("## Github semantic search (基于语义的 github 仓库搜索) 🌐")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 添加控制参数
                llm_filter = gr.Checkbox(
                    label="使用LLM过滤结果",
                    value=False,
                    info="是否使用 LLM 对搜索结果进行二次过滤"
                )
                keyword_expand = gr.Checkbox(
                    label="扩展关键词搜索",
                    value=False,
                    info="是否使用 LLM 扩展搜索关键词"
                )
                match_number = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=40,
                    step=10,
                    label="语义匹配数量",
                    info="进行语义匹配后返回的仓库数量，若使用 LLM 过滤，建议适当增加数量"
                )
            
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Agent",
                    type="messages",
                    avatar_images=(None, "https://img1.baidu.com/it/u=2193901176,1740242983&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=500"),
                    height="65vh"
                )
                prompt = gr.Textbox(max_lines=2, label="Chat Message")
                
        # 更新submit调用，包含新的参数
        prompt.submit(
            respond, 
            [prompt, chatbot, llm_filter, keyword_expand, match_number], 
            [chatbot]
        )
        prompt.submit(lambda: "", None, [prompt])

    demo.launch(share=False)


if __name__ == "__main__":
    main()