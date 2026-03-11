import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from config import Config


class DoubaoMultimodalEmbeddings(Embeddings):
    """适配豆包 /embeddings/multimodal 端点的 Embedding 实现。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        # 拼接多模态端点地址
        self.url = base_url.rstrip("/") + "/embeddings/multimodal"

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用豆包多模态 embedding 接口。"""
        # 豆包多模态接口每次只接受一个 input
        all_embeddings = []
        for text in texts:
            resp = requests.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "input": [{"type": "text", "text": text}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # 豆包多模态返回格式: data.embedding（对象）而非 data[0].embedding（数组）
            emb_data = data["data"]
            if isinstance(emb_data, list):
                all_embeddings.append(emb_data[0]["embedding"])
            else:
                all_embeddings.append(emb_data["embedding"])
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def embed_image(self, image_data_url: str) -> list[float]:
        """对图片 data URL 调用豆包多模态 embedding 接口。"""
        resp = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "input": [{"type": "image_url", "image_url": {"url": image_data_url}}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        emb_data = data["data"]
        if isinstance(emb_data, list):
            return emb_data[0]["embedding"]
        return emb_data["embedding"]


def _is_doubao_multimodal() -> bool:
    """判断是否需要使用豆包多模态 embedding 端点。"""
    base_url = Config.EMBEDDING_API_BASE_URL
    model = Config.get_embedding_model()
    # 豆包 API 地址 + vision/multimodal 类模型名
    return "volces.com" in base_url and "vision" in model


def _get_embeddings():
    """根据配置返回对应的 embedding 模型。"""
    if Config.MODEL_PROVIDER == "openai_compatible":
        if _is_doubao_multimodal():
            return DoubaoMultimodalEmbeddings(
                model=Config.get_embedding_model(),
                api_key=Config.EMBEDDING_API_KEY,
                base_url=Config.EMBEDDING_API_BASE_URL,
            )
        return OpenAIEmbeddings(
            model=Config.get_embedding_model(),
            openai_api_key=Config.EMBEDDING_API_KEY,
            openai_api_base=Config.EMBEDDING_API_BASE_URL,
        )
    else:
        return OllamaEmbeddings(
            model=Config.get_embedding_model(),
            base_url=Config.OLLAMA_BASE_URL,
        )


class VectorStore:
    def __init__(self, collection_name: str = "rag_demo"):
        self.embeddings = _get_embeddings()
        self.collection_name = collection_name
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    def add_documents(self, documents: list[Document]):
        """将文档块添加到向量库。"""
        self.db.add_documents(documents)

    def add_image_document(self, file_name: str, base64_data_url: str):
        """将图片作为单个文档写入向量库。"""
        import uuid
        embedding = self.embeddings.embed_image(base64_data_url)
        collection = self.db._collection
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[file_name],
            metadatas=[{
                "source": file_name,
                "type": "image",
                "image_data": base64_data_url,
            }],
        )

    def search(self, query: str, top_k: int = None) -> list[Document]:
        """检索与查询最相关的文档块。"""
        k = top_k or Config.SEARCH_TOP_K
        return self.db.similarity_search(query, k=k)

    def search_with_scores(self, query: str, top_k: int = None) -> list[tuple[Document, float]]:
        """检索并返回文档和相似度分数（分数越小越相似）。"""
        k = top_k or Config.SEARCH_TOP_K
        return self.db.similarity_search_with_score(query, k=k)

    def get_stats(self) -> dict:
        """获取向量库统计信息。"""
        collection = self.db._collection
        count = collection.count()
        return {
            "persist_dir": Config.CHROMA_PERSIST_DIR,
            "collection_name": self.collection_name,
            "total_chunks": count,
        }

    def get_all_documents(self, limit: int = 20) -> list[dict]:
        """获取向量库中存储的文档详情（用于调试/验证）。"""
        collection = self.db._collection
        count = collection.count()
        if count == 0:
            return []

        data = collection.get(
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )

        results = []
        for i in range(len(data["ids"])):
            embedding = data["embeddings"][i] if data["embeddings"] is not None else None
            results.append({
                "id": data["ids"][i],
                "content": data["documents"][i] if data["documents"] else "",
                "metadata": data["metadatas"][i] if data["metadatas"] else {},
                "embedding_preview": embedding[:8].tolist() if embedding is not None else [],
                "embedding_dim": len(embedding) if embedding is not None else 0,
            })
        return results

    def clear(self):
        """清空向量库。"""
        self.db.delete_collection()
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )
