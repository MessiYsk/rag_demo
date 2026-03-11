import os
import streamlit as st
from dotenv import load_dotenv

# 先加载配置文件，确保侧边栏能读取到默认值
load_dotenv(os.path.join(os.path.dirname(__file__), "config.env"))


def _apply_sidebar_config():
    """侧边栏配置，在导入依赖配置的模块之前调用。"""
    st.sidebar.header("⚙️ 配置")

    provider = st.sidebar.selectbox(
        "模型提供商",
        ["openai_compatible", "ollama"],
        format_func=lambda x: "OpenAI 兼容 API（DeepSeek/千问/Kimi/豆包/OpenAI…）" if x == "openai_compatible" else "Ollama（本地部署）",
        index=0 if os.getenv("MODEL_PROVIDER", "openai_compatible") != "ollama" else 1,
    )
    os.environ["MODEL_PROVIDER"] = provider

    if provider == "openai_compatible":
        # --- LLM 配置 ---
        st.sidebar.subheader("LLM（对话模型）")
        llm_api_key = st.sidebar.text_input(
            "LLM API Key",
            value=os.getenv("LLM_API_KEY", ""),
            type="password",
        )
        if llm_api_key:
            os.environ["LLM_API_KEY"] = llm_api_key

        llm_preset = st.sidebar.selectbox(
            "LLM 服务商",
            ["自定义", "OpenAI", "DeepSeek", "通义千问", "Kimi", "豆包"],
            key="llm_preset",
        )
        preset_urls = {
            "OpenAI": "https://api.openai.com/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
            "通义千问": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "Kimi": "https://api.moonshot.cn/v1",
            "豆包": "https://ark.cn-beijing.volces.com/api/v3",
        }
        llm_default_url = preset_urls.get(llm_preset, os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1"))

        llm_base_url = st.sidebar.text_input("LLM API Base URL", value=llm_default_url)
        os.environ["LLM_API_BASE_URL"] = llm_base_url

        llm_model = st.sidebar.text_input(
            "LLM 模型名称",
            value=os.getenv("LLM_MODEL_NAME", ""),
            placeholder="留空使用默认值",
            help="DeepSeek: deepseek-chat | 千问: qwen-turbo | Kimi: moonshot-v1-8k | 豆包: ep-xxx",
        )
        os.environ["LLM_MODEL_NAME"] = llm_model

        # --- Embedding 配置 ---
        st.sidebar.subheader("Embedding（向量化模型）")
        st.sidebar.caption("留空则复用 LLM 的 API Key 和 Base URL")

        emb_api_key = st.sidebar.text_input(
            "Embedding API Key",
            value=os.getenv("EMBEDDING_API_KEY", ""),
            type="password",
            placeholder="留空则复用 LLM Key",
        )
        os.environ["EMBEDDING_API_KEY"] = emb_api_key

        emb_preset = st.sidebar.selectbox(
            "Embedding 服务商",
            ["与 LLM 相同", "OpenAI", "DeepSeek", "通义千问", "豆包"],
            key="emb_preset",
        )
        if emb_preset == "与 LLM 相同":
            emb_default_url = ""
        else:
            emb_default_url = preset_urls.get(emb_preset, "")

        emb_base_url = st.sidebar.text_input(
            "Embedding API Base URL",
            value=emb_default_url or os.getenv("EMBEDDING_API_BASE_URL", ""),
            placeholder="留空则复用 LLM Base URL",
        )
        os.environ["EMBEDDING_API_BASE_URL"] = emb_base_url

        embedding_model = st.sidebar.text_input(
            "Embedding 模型名称",
            value=os.getenv("EMBEDDING_MODEL_NAME", ""),
            placeholder="留空使用默认值",
            help="OpenAI: text-embedding-3-small | 豆包: ep-xxx | 千问: text-embedding-v3",
        )
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
    else:
        ollama_url = st.sidebar.text_input(
            "Ollama 地址",
            value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        os.environ["OLLAMA_BASE_URL"] = ollama_url

    st.sidebar.subheader("参数调节")
    chunk_size = st.sidebar.slider("Chunk 大小", 100, 2000, int(os.getenv("CHUNK_SIZE", "500")), step=100)
    chunk_overlap = st.sidebar.slider("Chunk 重叠", 0, 500, int(os.getenv("CHUNK_OVERLAP", "50")), step=10)
    top_k = st.sidebar.slider("检索 Top-K", 1, 10, int(os.getenv("SEARCH_TOP_K", "4")))
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["SEARCH_TOP_K"] = str(top_k)


st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide")
st.title("📚 RAG 知识问答系统")

_apply_sidebar_config()

# 配置就绪后再导入依赖模块，强制重新加载确保配置生效
import importlib  # noqa: E402
import config as _cfg  # noqa: E402
importlib.reload(_cfg)
from config import Config  # noqa: E402

import vector_store as _vs_mod  # noqa: E402
importlib.reload(_vs_mod)
from vector_store import VectorStore  # noqa: E402

import rag_chain as _rc_mod  # noqa: E402
importlib.reload(_rc_mod)
from rag_chain import rag_query  # noqa: E402

from document_loader import load_document_from_bytes, is_image_file, load_image_as_base64  # noqa: E402
from text_splitter import split_text  # noqa: E402

# --- 初始化 session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# --- 侧边栏：文件上传 ---
st.sidebar.header("📄 文档上传")
uploaded_files = st.sidebar.file_uploader(
    "选择文档文件",
    type=["pdf", "txt", "csv", "docx", "png", "jpg", "jpeg", "gif", "bmp", "webp"],
    accept_multiple_files=True,
)

if st.sidebar.button("🔄 处理文档", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("请先上传文档！")
    else:
        with st.spinner("正在处理文档..."):
            vs = VectorStore()
            vs.clear()

            total_chunks = 0
            image_count = 0
            for f in uploaded_files:
                file_bytes = f.read()
                if is_image_file(f.name):
                    base64_url = load_image_as_base64(f.name, file_bytes)
                    vs.add_image_document(f.name, base64_url)
                    image_count += 1
                else:
                    text = load_document_from_bytes(f.name, file_bytes)
                    chunks = split_text(text, source=f.name)
                    vs.add_documents(chunks)
                    total_chunks += len(chunks)

            st.session_state.vector_store = vs
            st.session_state.doc_count = len(uploaded_files)
            st.sidebar.success(f"已处理 {len(uploaded_files)} 个文件，共 {total_chunks} 个文本块，{image_count} 张图片")

if st.sidebar.button("🗑️ 清空向量库", use_container_width=True):
    if st.session_state.vector_store:
        st.session_state.vector_store.clear()
    st.session_state.vector_store = None
    st.session_state.doc_count = 0
    st.session_state.messages = []
    st.sidebar.info("向量库已清空")

# 显示当前状态
st.sidebar.markdown("---")
provider_label = "OpenAI 兼容 API" if Config.MODEL_PROVIDER == "openai_compatible" else "Ollama"
st.sidebar.markdown(f"**当前提供商:** {provider_label}")
st.sidebar.markdown(f"**LLM:** {Config.get_llm_model()} @ `{Config.LLM_API_BASE_URL}`")
st.sidebar.markdown(f"**Embedding:** {Config.get_embedding_model()} @ `{Config.EMBEDDING_API_BASE_URL}`")
st.sidebar.markdown(f"**已加载文档:** {st.session_state.doc_count} 个")

# --- 主区域：Tab 切换 ---
tab_chat, tab_vector = st.tabs(["💬 问答对话", "🔍 向量库检查"])


def _render_source(src, score=None):
    """渲染单条参考来源（文本或图片）。"""
    score_text = f" | **相似度距离:** {score:.4f}" if score is not None else ""
    if src.metadata.get("type") == "image":
        st.markdown(f"**来源:** {src.metadata.get('source', '未知')} | **类型:** 图片{score_text}")
        image_data = src.metadata.get("image_data", "")
        if image_data:
            st.image(image_data, caption=src.metadata.get("source", ""), width=300)
    else:
        st.markdown(
            f"**来源:** {src.metadata.get('source', '未知')} | "
            f"**片段:** {src.metadata.get('chunk_index', '?')}"
            f"{score_text}"
        )
        st.text(src.page_content[:300] + ("..." if len(src.page_content) > 300 else ""))
    st.markdown("---")

# === Tab 1: 对话界面 ===
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for idx, src in enumerate(msg["sources"]):
                        score = msg.get("scores", [])[idx] if msg.get("scores") else None
                        _render_source(src, score)

    if question := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.vector_store:
            answer = "⚠️ 请先在侧边栏上传并处理文档。"
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    result = rag_query(question, st.session_state.vector_store)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("📎 参考来源"):
                        for idx, src in enumerate(result["sources"]):
                            score = result["scores"][idx] if result.get("scores") else None
                            _render_source(src, score)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "scores": result.get("scores", []),
                })

# === Tab 2: 向量库检查 ===
with tab_vector:
    st.subheader("向量库数据验证")

    if not st.session_state.vector_store:
        st.info("还没有向量数据。请先在侧边栏上传并处理文档。")
    else:
        vs = st.session_state.vector_store

        # 统计信息
        stats = vs.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("总文本块数", stats["total_chunks"])
        col2.metric("集合名称", stats["collection_name"])
        col3.metric("存储路径", stats["persist_dir"])

        st.markdown("---")

        # 文档详情
        st.subheader("存储的文档块详情")
        docs = vs.get_all_documents(limit=50)

        if docs:
            for i, doc in enumerate(docs):
                is_image = doc["metadata"].get("type") == "image"
                label = "图片" if is_image else f"片段 {doc['metadata'].get('chunk_index', '?')}"
                with st.expander(
                    f"块 {i} — {doc['metadata'].get('source', '未知来源')} "
                    f"({label}) "
                    f"[向量维度: {doc['embedding_dim']}]"
                ):
                    if is_image:
                        st.markdown("**图片内容：**")
                        image_data = doc["metadata"].get("image_data", "")
                        if image_data:
                            st.image(image_data, caption=doc["metadata"].get("source", ""), width=300)
                    else:
                        st.markdown("**文本内容：**")
                        st.text(doc["content"])
                    st.markdown("**元数据：**")
                    st.json(doc["metadata"])
                    if doc["embedding_preview"]:
                        st.markdown(f"**向量前 8 维：** `{[round(v, 6) for v in doc['embedding_preview']]}`")
                        st.caption(f"向量总维度: {doc['embedding_dim']}")
        else:
            st.warning("向量库为空")

        # 相似度检索测试
        st.markdown("---")
        st.subheader("相似度检索测试")
        search_mode = st.radio("检索方式", ["文本检索", "图片检索"], horizontal=True)

        search_triggered = False
        search_results = []

        if search_mode == "文本检索":
            test_query = st.text_input("输入测试查询", placeholder="例如：什么是 RAG？")
            if test_query:
                search_results = vs.search_with_scores(test_query)
                search_triggered = True
        else:
            search_image = st.file_uploader(
                "上传图片进行以图搜索",
                type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
                key="search_image",
            )
            if search_image:
                img_bytes = search_image.read()
                img_data_url = load_image_as_base64(search_image.name, img_bytes)
                st.image(img_data_url, caption="查询图片", width=200)
                embedding = vs.embeddings.embed_image(img_data_url)
                raw_results = vs.db._collection.query(
                    query_embeddings=[embedding],
                    n_results=Config.SEARCH_TOP_K,
                    include=["documents", "metadatas", "distances"],
                )
                if raw_results and raw_results["ids"] and raw_results["ids"][0]:
                    from langchain_core.documents import Document as LCDoc
                    for j in range(len(raw_results["ids"][0])):
                        doc = LCDoc(
                            page_content=raw_results["documents"][0][j] or "",
                            metadata=raw_results["metadatas"][0][j] or {},
                        )
                        dist = raw_results["distances"][0][j]
                        search_results.append((doc, dist))
                search_triggered = True

        if search_triggered and search_results:
            for idx, (doc, score) in enumerate(search_results):
                is_img = doc.metadata.get("type") == "image"
                type_label = "图片" if is_img else "文本"
                st.markdown(
                    f"**#{idx + 1}** | 距离: `{score:.4f}` | "
                    f"来源: {doc.metadata.get('source', '未知')} | "
                    f"类型: {type_label}"
                )
                if is_img:
                    image_data = doc.metadata.get("image_data", "")
                    if image_data:
                        st.image(image_data, caption=doc.metadata.get("source", ""), width=300)
                else:
                    st.text(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
                st.markdown("---")
        elif search_triggered:
            st.info("未找到相关结果")
