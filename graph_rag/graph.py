"""
Script responsible for build a knowledge graph using
Neo4j from unstructured text
"""

import os
import re
from dotenv import load_dotenv
from typing import List
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredFileLoader,BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


load_dotenv()

class GraphBuilder():
    """
    封装了从多源非结构化文本构建完整知识图谱所需的核心功能

    _extended_summary_
    """
    def __init__(self, graph_model="gpt-3.5-turbo-0125"):
        self.graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
        if graph_model=="llama3":
            self.llm = ChatOllama(base_url=os.getenv("OLLAMA_SERVER"), model=os.getenv("OLLAMA_MODEL"))
        else:
            self.llm = ChatOpenAI(model=graph_model, temperature=0)


    def graph_document_text(self, text_chunks):
        """
        使用LLMGraphTransformer将非结构化文本转换为知识图谱

        参数:
            text_chunks (List): 文档分块列表
        """
        # neo4j提供的transformer：可将chunks转换成图谱
        llm_transformer = LLMGraphTransformer(llm=self.llm)

        # https://api.python.langchain.com/en/latest/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html
        graph_docs = llm_transformer.convert_to_graph_documents(text_chunks)
        print(f"graph_docs:  {graph_docs}")
        # 添加至neo4j 图数据库
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )
    
    def bs4_extractor(self, html: str) -> str:
        soup = Soup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    def graph_text_content(self, paths, chunk_size, chunk_overlap):
        """
        提供文本文档，在生成图谱之前提取并分块文本

        参数:
            paths ([str]): 文本文档路径,tuple
            chunk_size (int): 文本分块大小
            chunk_overlap (int): 文本分块重叠
        """

        # 遍历paths
        for path in paths:
            print('------------------start-----------------------\n')
             # 判断path是以https://或者http://开头的URL
            if path.startswith("https://") or path.startswith("http://") :
                # 加载文件
                loader = RecursiveUrlLoader(url=path, max_depth=2, extractor=self.bs4_extractor)
                text_docs = loader.load()
                # 写到临时文件中
                with open("temp.txt", "w") as f:
                    f.write(text_docs)
                f.close()
            # 根据文件后缀选择合适的加载器
            elif os.path.splitext(path)[1].lower() == '.txt':
                text_docs = TextLoader(path).load()
            elif os.path.splitext(path)[1].lower() == '.pdf':
                text_docs = PyPDFLoader(path).load()
            elif os.path.splitext(path)[1].lower() == '.html':
                text_docs = BSHTMLLoader(path).load()
            else:
                raise ValueError(f"不支持的文件类型:  {os.path.splitext(path)[1].lower()}")

            # 文本分块
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_documents(text_docs)
            if text_chunks is not None:
                print(text_chunks)
                print('------------------split-----------------------\n')
                # 转换并添加至graph库
                self.graph_document_text(text_chunks)
            print('------------------end-----------------------\n')

    def index_graph(self):
        """
        创建已填充图谱的索引以辅助高效搜索
        """
        self.graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    def reset_graph(self):
        """
        清除整个图谱
        """
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )
