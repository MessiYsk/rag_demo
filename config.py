import os
from dotenv import load_dotenv

_project_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(_project_dir, "config.env"))


class Config:
    # 模型提供商: "openai_compatible" 或 "ollama"
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai_compatible").lower()

    # LLM 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")

    # Embedding 配置（留空则复用 LLM 的配置）
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY
    EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "") or LLM_API_BASE_URL
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")

    # Ollama 配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # 文本分块
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 检索
    SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "4"))

    # 向量库路径
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

    @classmethod
    def get_llm_model(cls):
        if cls.LLM_MODEL_NAME:
            return cls.LLM_MODEL_NAME
        return "gpt-4o-mini" if cls.MODEL_PROVIDER == "openai_compatible" else "qwen2"

    @classmethod
    def get_embedding_model(cls):
        if cls.EMBEDDING_MODEL_NAME:
            return cls.EMBEDDING_MODEL_NAME
        return "text-embedding-3-small" if cls.MODEL_PROVIDER == "openai_compatible" else "nomic-embed-text"
