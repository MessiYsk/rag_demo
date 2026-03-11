from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config


def split_text(text: str, source: str = "") -> list[Document]:
    """将文本分割为多个文档块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": source, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
