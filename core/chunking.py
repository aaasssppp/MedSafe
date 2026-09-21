"""
Document chunking utilities for MedSafe.
Splits medical text into semantically cohesive passages while preserving test rows.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config


def split_medical_text(
    text: str,
    source_name: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    extra_metadata: Dict[str, Any] = None,
) -> List[Document]:
    """
    Split medical report or reference text into LangChain Document chunks.

    Args:
        text: Raw cleaned medical text
        source_name: Filename or document identifier
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        extra_metadata: Optional additional metadata to include

    Returns:
        List of Document instances
    """
    if not text or not text.strip():
        return []

    c_size = chunk_size or Config.CHUNK_SIZE
    c_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    # Splitting separators tuned for medical reports and laboratory tables
    separators = [
        "\n--- Page",  # Page boundaries
        "\n\n",       # Paragraph/table block boundaries
        "\n",         # Individual lab test line boundaries
        "; ",         # Semi-colon separated test results
        ". ",         # Sentence boundaries
        " ",          # Word boundaries
        "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=separators,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    documents: List[Document] = []
    base_meta = {"source": source_name, "total_chunks": len(raw_chunks)}
    if extra_metadata:
        base_meta.update(extra_metadata)

    for idx, chunk in enumerate(raw_chunks):
        meta = base_meta.copy()
        meta["chunk_index"] = idx
        meta["chunk_id"] = f"{source_name}__chunk_{idx}"
        documents.append(Document(page_content=chunk.strip(), metadata=meta))

    return documents
