"""
Script responsible for build a knowledge graph using
Neo4j from unstructured text
"""

import os
from dotenv import load_dotenv
from typing import List
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


load_dotenv()
# os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
# os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
# os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

class GraphBuilder():
    """
    封装了从多源非结构化文本构建完整知识图谱所需的核心功能

    _extended_summary_
    """
    def __init__(self, use_llama=False):
        self.graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
        if use_llama:
            self.llm = ChatOllama(base_url=os.getenv("OLLAMA_SERVER"), model=os.getenv("OLLAMA_MODEL"))
        else:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


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
            # 获取文件后缀
            file_extension = os.path.splitext(path)[1].lower()

            # 根据文件后缀选择合适的加载器
            if file_extension == '.txt':
                text_docs = TextLoader(path).load()
                # print(text_docs)
                # 将原始文本分解成块
                text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                text_chunks = text_splitter.split_documents(text_docs[:3])
                if text_chunks is not None:
                    print(text_chunks)
                    # 转换并添加至graph库
                    self.graph_document_text(text_chunks)
            elif file_extension == '.pdf':
                docs = PyPDFLoader(path).load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs = text_splitter.split_documents(docs)
                print(docs)
                self.graph_document_text(docs)

            else:
                raise ValueError(f"不支持的文件类型:  {file_extension}")


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
