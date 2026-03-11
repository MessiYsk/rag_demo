from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import Config
from vector_store import VectorStore

PROMPT_TEMPLATE = """你是一个专业的知识助手。请根据以下参考资料回答用户的问题。

要求：
1. 仅根据参考资料中的内容回答，不要编造信息
2. 如果参考资料中没有相关内容，请明确告知用户
3. 回答要清晰、准确、有条理

参考资料：
{context}

用户问题：{question}

回答："""


def _get_llm():
    """根据配置返回对应的 LLM。"""
    if Config.MODEL_PROVIDER == "openai_compatible":
        return ChatOpenAI(
            model=Config.get_llm_model(),
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.LLM_API_BASE_URL,
            temperature=0.3,
        )
    else:
        return ChatOllama(
            model=Config.get_llm_model(),
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3,
        )


def rag_query(question: str, vector_store: VectorStore) -> dict:
    """执行 RAG 查询：检索 → 生成。

    返回:
        {"answer": str, "sources": list[Document], "scores": list[float]}
    """
    # 1. 检索相关文档（带分数）
    results_with_scores = vector_store.search_with_scores(question)

    if not results_with_scores:
        return {
            "answer": "未找到相关文档，请先上传文档。",
            "sources": [],
            "scores": [],
        }

    relevant_docs = [doc for doc, _ in results_with_scores]
    scores = [score for _, score in results_with_scores]

    # 2. 构建上下文（仅使用文本文档，跳过图片）
    text_docs = [doc for doc in relevant_docs if doc.metadata.get("type") != "image"]
    context = "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}, 片段 {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
        for doc in text_docs
    )

    # 3. 调用 LLM 生成回答
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": relevant_docs,
        "scores": scores,
    }
