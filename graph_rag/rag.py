"""
用于管理所有 GraphRAG 功能的模块。主要围绕构建一个能够解析知识图谱并返回相关结果的检索器。
"""
import os
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from graph_rag.entities import Entities

class GraphRAG():
    """
    封装所有查询图以进行检索增强生成所需方法的类。
    """

    def __init__(self, chat_model="gpt-3.5-turbo-0125"):
        self.graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
        self.llm = ChatOpenAI(model=chat_model, temperature=0.7)

    def create_entity_extract_chain(self):
        """
        创建一个链，从用户提出的问题中提取实体。
        使得有效地在图中搜索与实体相对应的节点

        Returns:
            Runnable: 使用 LLM 从用户问题中提取实体的可运行链。
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你进行提取文本中出现的所有人物、专业名词、对象、组织或商业实体等",
                ),
                (
                    "human",
                    "使用给定的格式从以下内容中提取信息"
                    "input: {question}",
                ),
            ]
        )

        entity_extract_chain = prompt | self.llm.with_structured_output(Entities)
        return entity_extract_chain

    def generate_full_text_query(self, input_query: str) -> str:
        """
        为给定的输入字符串生成全文搜索查询。

        此函数构造适用于全文搜索的查询字符串。
        它通过将输入字符串拆分为单词并向每个单词附加相似度阈值（约 2 个更改的字符）来处理输入字符串，
        然后使用 AND 运算符将它们组合起来。可用于将用户问题的实体映射到数据库值，并允许出现一些拼写错误。

        Args:
            input_query (str): 从用户问题中提取的实体名称。

        Returns:
            str: 构建的全文搜索查询字符串。
        """
        full_text_query = ""

        # 拆分单词并删除为cipher query保留的任何特殊字符
        words = [el for el in remove_lucene_chars(input_query).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> str:
        """
        创建一个检索器，使用从用户查询中提取的实体来请求图中的上下文，并返回与该查询相关的邻近节点和边。

        Args:
            question (str): 用户为此图 RAG 提出的问题。

        Returns:
            str: 完整构建的图查询，将检索与用户问题相关的上下文。
        """

        entity_extract_chain = self.create_entity_extract_chain()
        result = ""
        entities = entity_extract_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def create_vector_index(self) -> Neo4jVector:
        """
        使用现有graph创建一个向量索引。该向量表示基于指定的属性。
        使用 OpenAIEmbeddings。

        Returns:
            Neo4jVector: 配置中指定的图节点的向量表示。
        """
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model="text-embedding-3-small"),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index

    def retriever(self, question: str) -> str:
        """
        图 RAG 检索器，结合结构化和非结构化检索方法，根据用户问题形成单一检索器。
        混合检索：包含向量相似性搜索&结构化图遍历搜索

        Args:
            question (str): 用户为此图 RAG 提出的问题。

        Returns:
            str: 从图中检索的两种形式的数据。
        """
        print(f"Search query: {question}")
        # 创建向量索引，执行相似性搜索
        vector_index = self.create_vector_index()
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        # 检索结构化数据（图遍历搜索）
        structured_data = self.structured_retriever(question)
        final_data = f"""结构化数据:
            {structured_data}
            非结构化数据:
            {"#Document ". join(unstructured_data)}
        """
        print(f"final_data: {final_data}")
        return final_data

    def create_search_query(self, chat_history: List, question: str) -> str:
        """
        将聊天记录与当前问题结合起来，形成一个提示，可以由 LLM 执行，以使用历史记录回答新问题。

        Args:
            chat_history (List): 此对话中捕获的消息列表。
            question (str): 用户为此图 RAG 提出的当前问题。

        Returns:
            str: 格式化的提示，可以与问题和聊天记录一起发送给 LLM。
        """
        search_query = ChatPromptTemplate.from_messages([
            (
                "system",
                """根据以下对话和跟进问题，将跟进问题重新表述为独立问题，使用其原始语言。
                聊天记录:
                {chat_history}
                跟进输入: {question}
                独立问题:"""
            )
        ])
        formatted_query = search_query.format(
            chat_history=chat_history, question=question)
        return formatted_query
