import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline

# 设置远程模型路径
base_model = "Qwen/Qwen2-7B-Instruct"

# 加载模型和分词器（直接从远程加载）
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 初始化聊天模型
chat_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# 创建 HuggingFacePipeline 的 LLM
llm = HuggingFacePipeline(pipeline=chat_pipeline)

# 为聊天机器人定义一个提示模板
prompt_template = PromptTemplate(
    input_variables=["input", "search_results"],
    template="""
You are a helpful assistant. Based on the following search results, answer the user's query in a helpful and detailed manner.

User's Query: {input}

Search Results: {search_results}

Answer in markdown format:
"""
)

# 百度搜索功能
def baidu_search(query):
    search_url = f"https://www.baidu.com/s?wd={query}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.102 Safari/537.36"
        )
    }
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result')
        search_results = []
        for result in results:
            title = result.find('h3').get_text(strip=True) if result.find('h3') else ""
            abstract = result.find('div', class_='c-abstract').get_text(strip=True) if result.find('div', class_='c-abstract') else ""
            if title and abstract:
                search_results.append(f"{title}: {abstract}")
        if search_results:
            return " ".join(search_results[:5])  # 返回前5个搜索结果
        else:
            return "未找到相关信息。"
    except requests.exceptions.RequestException as e:
        return f"请求发生错误: {e}"

# 创建一个集成百度搜索工具和 LLM 的链
def chain(input_text):
    # 获取百度搜索结果
    search_results = baidu_search(input_text)
    # 创建 LLMChain
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    # 运行链
    response = llm_chain.run({"input": input_text, "search_results": search_results})
    return response

# 示例用法
response = chain("青岛今天的天气如何")
print(response)
