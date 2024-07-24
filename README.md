# Graph RAG
<p align="center">
  <img src="graph_rag\images\langchain_graph_rag.png" alt="graph-rag" width="500"/>
</p>

## 先安装图数据库（Docker部署）,也可以选择neo4j cloud版本
```
docker run `
    -p 7474:7474 -p 7687:7687 `
    -v ${PWD}/data:/data -v ${PWD}/plugins:/plugins `
    --name neo4j-v5-apoc `
    -e NEO4J_apoc_export_file_enabled=true `
    -e NEO4J_apoc_import_file_enabled=true `
    -e NEO4J_apoc_import_file_use_neo4j_config=true `
    -e NEO4J_PLUGINS='["apoc"]' `
    -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" `
    neo4j:5.20.0
```

## 运行程序
在graph_rag下新增 .env文件，eg:
```
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://apis.openai.com/v1
NEO4J_URI=neo4j://xxx:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxx
OLLAMA_SERVER=http://xxx:8034
OLLAMA_MODEL=llama3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

安装依赖项. [Poetry](https://python-poetry.org/) 

`poetry install`

运行程序

`streamlit run .\graph_rag\app.py`