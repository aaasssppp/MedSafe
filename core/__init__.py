"""
MedSafe Core Package.
Exposes processing, OCR, chunking, embeddings, and RAG services.
"""

from core.pdf_processor import extract_pdf_text, clean_medical_text
from core.ocr import extract_text_ocr, is_ocr_available
from core.chunking import split_medical_text
from core.embeddings import GeminiEmbeddings
from core.medical_analyzer import generate_initial_report_analysis
from core.rag import MedicalRAGService, medical_rag_service

__all__ = [
    "extract_pdf_text",
    "clean_medical_text",
    "extract_text_ocr",
    "is_ocr_available",
    "split_medical_text",
    "GeminiEmbeddings",
    "generate_initial_report_analysis",
    "MedicalRAGService",
    "medical_rag_service",
]
