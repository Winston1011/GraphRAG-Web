"""
RAG应用程序的主要入口点。将其他RAG功能集成到UI中。
"""
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from graph_rag.graph import GraphBuilder
from graph_rag.rag import GraphRAG
import json
import time
import asyncio

# 从.env文件加载环境变量
load_dotenv()

# OpenAI & Neo4j 客户端设置
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

def graph_content(file_paths, progress_bar, status_text, chunk_size, chunk_overlap, graph_model):
    """
    生成新图的入口点。将来会在UI中添加控件以执行这些操作。
    """
    print("开始从文本内容构建知识图谱:")
    graph_builder = GraphBuilder(graph_model=graph_model)

    status_text.text("开始解析&存储")
    progress_bar.progress(1/5)
    time.sleep(0.5)
    progress_bar.progress(2/5)
    time.sleep(0.5)
    progress_bar.progress(3/5)
    time.sleep(0.5)
    progress_bar.progress(4/5)
    graph_builder.graph_text_content(file_paths, chunk_size, chunk_overlap)
    status_text.text("完成")
    progress_bar.progress(5/5)

    graph_builder.index_graph()

def reset_graph():
    """
    重置图，删除所有关系和节点。
    """
    graph_builder = GraphBuilder()
    graph_builder.reset_graph()

async def response_generator(question: str, model: str, embedding_model: str):
    """
    对于给定的问题,将制定搜索查询并使用自定义的GraphRAG检索器从知识图谱中获取相关内容。

    参数:
        question (str): 用户为该图RAG提出的问题

    返回:
        str: 基于图的提问结果
    """
    rag = GraphRAG(chat_model=model)
    search_query = rag.create_search_query(st.session_state.chat_history, question)

    template = """根据以下上下文回答问题:
    {context}

    问题: {question}
    使用自然语言并简洁回答。
    回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model=model, temperature=0)

    chain = (
        RunnableParallel(
            {
                "context": lambda x: rag.retriever(search_query, embedding_model),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # 创建一个异步事件循环来运行异步函数
    chunks = []
    async for chunk in chain.astream({"chat_history": st.session_state.chat_history, "question": question}):
        chunks.append(chunk)
        yield chunk  # 逐步返回 chunk

def init_ui():
    """
    应用程序的主要入口点。创建与LLM交互的聊天界面。
    """
    st.set_page_config(page_title="Langchain GraphRAG Bot", layout="wide")
    st.title("Langchain GraphRAG Bot")

    # 初始化会话状态
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "你好,我是你的专属知识库小助手。请问有什么问题吗？"}
        ]

    # 清除按钮
    if st.button("Clear", key="clear_button", help="Clear chat history"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "你好,我是你的专属知识库小助手。请问有什么问题吗？"}
        ]
        # st.experimental_rerun()

    # 在应用程序重新运行时显示聊天记录中的消息
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.header("Graph Management")
        st.write("Below are options to populate and reset your graph database")
        password = st.text_input("请输入密码:", type="password")
        col1, col2 = st.columns(2)

        intput_url = st.text_input("请输入待提取文本的URL:")
        # 支持多文件上传
        uploaded_files = st.file_uploader("上传txt/pdf/html文件:", type=["txt", "pdf", "html"], accept_multiple_files=True)
        file_paths = []
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                if uploaded_file.size > 20 * 1024 * 1024:  # 限制文件大小为20MB
                    st.error(f"文件 '{uploaded_file.name}' 超出限制50KB")
                else:
                    file_path = uploaded_file.name
                    file_paths.append(file_path)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File '{file_path}' uploaded successfully!")

        # 添加模型选择下拉菜单
        graph_model = st.selectbox("构建图谱模型:", 
            [
                "ernie-speed-128k",
                "gpt-3.5-turbo-0125",
                "gpt-4o-mini", 
                "llama3"
            ]
        )
        chat_model = st.selectbox("对话模型选择:", 
            [
                "gpt-3.5-turbo-0125",
                "gpt-4o-mini", 
                "gpt-4o"
            ]
        )
        embedding_model = st.selectbox("Embedding模型选择:", 
            [
                "text-embedding-3-small",
                "text-embedding-ada-002", 
                "nomic-embed-text"
            ]
        )

        # chunk_size
        chunk_size = st.number_input("chunk_size:", min_value=8, max_value=2048, value=512, step=8)
        # chunk_overlap
        chunk_overlap = st.number_input("chunk_overlap:", min_value=8, max_value=512, value=24, step=8)

        # 判断file_paths是否为空
       
        passwordFlag = password == "epochdz"
        with col1:
            if st.button("Populate Graph", disabled=(len(file_paths) == 0 and len(intput_url)==0) or not passwordFlag):
                progress_bar = st.progress(0)
                status_text = st.empty()
                if len(intput_url) > 0 :
                    file_paths.append(intput_url)
                graph_content(file_paths, progress_bar, status_text, chunk_size, chunk_overlap, graph_model)

        with col2:
            if st.button("Reset Graph", disabled=not passwordFlag):
                reset_graph()
    
       # 接受用户输入
    if user_query := st.chat_input("Ask a question...."):
        # 将用户消息添加到聊天记录
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        # 在聊天消息容器中显示用户消息
        with st.chat_message("user"):
            st.markdown(user_query)
        # 在聊天消息容器中显示助手响应
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_parts = []

            # 创建一个事件循环来运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def update_response():
                async for chunk in response_generator(user_query, chat_model, embedding_model):
                    response_parts.append(chunk)
                    response_placeholder.markdown("".join(response_parts))
                response = "".join(response_parts)
                # 将助手响应添加到聊天记录
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            loop.run_until_complete(update_response())
        

if __name__ == "__main__":
    init_ui()
