"""
RAG Engine for MedSafe.
Implements the dual-retrieval pipeline:
1. Patient Medical Report Retriever (session/user documents)
2. Verified Medical Knowledge Base Retriever (offline pre-indexed documents)
3. Safety-grounded LLM synthesis via Google Gemini
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from google import genai

from config import Config
from core.embeddings import GeminiEmbeddings
from core.pdf_processor import extract_pdf_text
from core.chunking import split_medical_text
from core.medical_analyzer import generate_initial_report_analysis

logger = logging.getLogger("medsafe.rag")

SAFETY_SYSTEM_INSTRUCTION = """You are MedSafe, a trusted AI medical report assistant.
Your goal is to help patients understand complex laboratory reports, diagnostic findings, and clinical terms in plain, compassionate, and accessible language.

CRITICAL SAFETY DIRECTIVES:
1. Ground your response in the provided Patient Medical Report Findings and Verified Medical Knowledge Base.
2. DO NOT formulate a medical diagnosis (e.g., never say "You have diabetes" or "You are suffering from anemia"). Instead, describe what the specific marker or value measures, note whether it falls within or outside typical reference intervals, and explain what clinical literature suggests about such elevations or reductions.
3. DO NOT advise starting, stopping, or modifying dosages of prescription medications.
4. If a question cannot be answered using the provided context or if the value is missing from the report, state clearly: "Your uploaded report does not contain information regarding [topic]." Do not guess or extrapolate unlisted lab numbers.
5. Clearly distinguish between what is documented in the patient's individual report versus what is general medical context from the verified knowledge base.
6. Always encourage open dialogue with the patient's healthcare provider.
"""

QA_PROMPT_TEMPLATE = """{system_instruction}

=== PATIENT MEDICAL REPORT FINDINGS ===
{report_context}

=== VERIFIED MEDICAL KNOWLEDGE BASE ===
{kb_context}

=== PATIENT QUESTION ===
{question}

Please provide a clear, accurate, and empathetic response adhering to all safety rules above.
"""

DISCLAIMER_NOTE = (
    "\n\n---\n*🛡️ **MedSafe Safety Notice**: This explanation is for educational purposes and is not a substitute "
    "for clinical diagnosis or medical care. Please consult your physician regarding all test results and health decisions.*"
)


class MedicalRAGService:
    """
    Orchestrates the dual-RAG pipeline for MedSafe.
    """

    def __init__(self):
        self.embeddings = GeminiEmbeddings()
        self.kb_embeddings = GeminiEmbeddings() 
        self.kb_vector_store: Optional[FAISS] = None
        self.user_vector_store: Optional[FAISS] = None
        self._llm_client: Optional[genai.Client] = None

        # Initialize LLM client if API key is present
        if Config.GEMINI_API_KEY:
            self._llm_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Load existing verified knowledge base at startup (zero embedding API calls)
        self.load_knowledge_base()

        # Load existing user vector store if present on disk
        self.load_user_vector_store()

    def load_knowledge_base(self) -> bool:
        """
        Load existing FAISS index for the verified medical knowledge base from disk.
        Does NOT build or re-embed the knowledge base.
        """
        kb_path = Config.KB_VECTOR_STORE_PATH
        index_file = os.path.join(kb_path, "index.faiss")

        if os.path.exists(index_file):
            try:
                self.kb_vector_store = FAISS.load_local(
                    kb_path,
                    self.kb_embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Successfully loaded verified medical knowledge base from %s", kb_path)
                return True
            except Exception as e:
                logger.error("Failed to load existing knowledge base vector store: %s", e)
                self.kb_vector_store = None
                return False
        else:
            logger.warning(
                "No verified knowledge base index found at '%s'. "
                "Run 'python ingestion/ingest_knowledge_base.py' to build the medical knowledge index.",
                kb_path,
            )
            self.kb_vector_store = None
            return False

    def load_user_vector_store(self) -> bool:
        """Load user report vector store from disk if previously processed."""
        user_path = Config.USER_VECTOR_STORE_PATH
        index_file = os.path.join(user_path, "index.faiss")

        if os.path.exists(index_file):
            try:
                self.user_vector_store = FAISS.load_local(
                    user_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded user report index from %s", user_path)
                return True
            except Exception as e:
                logger.warning("Could not load user report index: %s", e)
                self.user_vector_store = None
                return False
        return False

    def is_knowledge_base_ready(self) -> bool:
        """Check if verified knowledge base is loaded and available."""
        return self.kb_vector_store is not None

    def is_user_report_ready(self) -> bool:
        """Check if at least one user medical report has been processed."""
        return self.user_vector_store is not None

    def get_status(self) -> Dict[str, Any]:
        """Return system status including knowledge base and user report readiness."""
        kb_docs_count = 0
        if self.kb_vector_store and hasattr(self.kb_vector_store, "docstore"):
            try:
                kb_docs_count = len(self.kb_vector_store.docstore._dict)
            except Exception:
                kb_docs_count = 1 if self.kb_vector_store else 0

        user_chunks_count = 0
        if self.user_vector_store and hasattr(self.user_vector_store, "docstore"):
            try:
                user_chunks_count = len(self.user_vector_store.docstore._dict)
            except Exception:
                user_chunks_count = 1 if self.user_vector_store else 0

        return {
            "api_key_configured": bool(Config.GEMINI_API_KEY),
            "kb_indexed": self.is_knowledge_base_ready(),
            "kb_chunks_count": kb_docs_count,
            "user_reports_processed": self.is_user_report_ready(),
            "user_chunks_count": user_chunks_count,
            "llm_model": Config.LLM_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
        }

    def process_user_reports(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process uploaded medical report PDFs:
        1. Extract text (with OCR fallback)
        2. Chunk text into documents
        3. Build user FAISS index
        4. Generate initial clinical summary
        """
        if not file_paths:
            return {
                "status": "error",
                "message": "No files provided for processing.",
                "summary": "",
                "files": [],
            }

        all_documents: List[Document] = []
        combined_texts: List[str] = []
        processed_files_info = []

        for f_path in file_paths:
            fname = os.path.basename(f_path)
            extraction_result = extract_pdf_text(f_path)

            text = extraction_result.get("text", "")
            method = extraction_result.get("method", "digital_text")
            page_count = extraction_result.get("page_count", 0)

            if text:
                chunks = split_medical_text(text, source_name=fname)
                all_documents.extend(chunks)
                combined_texts.append(f"=== File: {fname} ===\n{text}")
                processed_files_info.append({
                    "filename": fname,
                    "pages": page_count,
                    "method": method,
                    "chunks": len(chunks),
                    "status": "success",
                })
            else:
                processed_files_info.append({
                    "filename": fname,
                    "pages": page_count,
                    "method": method,
                    "chunks": 0,
                    "status": extraction_result.get("status", "empty"),
                    "message": extraction_result.get("message", "No text extracted."),
                })

        if not all_documents:
            return {
                "status": "warning",
                "message": "Could not extract readable text from any uploaded document.",
                "summary": "⚠️ No readable text could be extracted from the uploaded files. Please ensure documents are legible.",
                "files": processed_files_info,
            }

        # Build fresh FAISS index for this session's reports
        try:
            self.user_vector_store = FAISS.from_documents(all_documents, self.embeddings)
            # Persist user store on disk
            self.user_vector_store.save_local(Config.USER_VECTOR_STORE_PATH)
            logger.info("Built user report vector store with %d chunks.", len(all_documents))
        except Exception as e:
            logger.error("Failed to build user vector store: %s", e)
            return {
                "status": "error",
                "message": f"Vector indexing failed: {e}",
                "summary": "",
                "files": processed_files_info,
            }

        # Generate initial analysis/summary
        full_text = "\n\n".join(combined_texts)
        analysis_result = generate_initial_report_analysis(full_text)

        return {
            "status": "success",
            "message": f"Successfully processed {len(processed_files_info)} document(s).",
            "summary": analysis_result.get("summary", ""),
            "files": processed_files_info,
            "total_chunks": len(all_documents),
        }

    def clear_user_reports(self):
        """Clear user reports vector store and reset session."""
        self.user_vector_store = None
        user_path = Config.USER_VECTOR_STORE_PATH
        if os.path.exists(user_path):
            for item in os.listdir(user_path):
                try:
                    os.remove(os.path.join(user_path, item))
                except Exception as e:
                    logger.warning("Could not delete %s: %e", item, e)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute RAG query against both User Medical Reports and Verified Knowledge Base.
        """
        clean_question = question.strip()
        if not clean_question:
            return {
                "response": "Please enter a question about your medical report.",
                "sources": [],
                "status": "empty_question",
            }

        if not Config.GEMINI_API_KEY:
            return {
                "response": (
                    "⚠️ **API Key Required**: GEMINI_API_KEY is not configured in `.env`. "
                    "Please set your Gemini API key to enable question answering."
                ),
                "sources": [],
                "status": "missing_api_key",
            }

        report_chunks_text = ""
        kb_chunks_text = ""
        sources = []

        # 1. Retrieve from user medical report
        if self.user_vector_store:
            try:
                retriever = self.user_vector_store.as_retriever(
                    search_kwargs={"k": Config.USER_REPORT_TOP_K}
                )
                user_docs = retriever.invoke(clean_question)
                if user_docs:
                    report_chunks_text = "\n\n".join(doc.page_content for doc in user_docs)
                    for doc in user_docs:
                        src = doc.metadata.get("source", "Medical Report")
                        label = f"📄 User Report: {src}"
                        if label not in sources:
                            sources.append(label)
            except Exception as e:
                logger.error("Error querying user vector store: %s", e)

        # 2. Retrieve from verified medical knowledge base
        if self.kb_vector_store:
            try:
                retriever = self.kb_vector_store.as_retriever(
                    search_kwargs={"k": Config.KB_TOP_K}
                )
                kb_docs = retriever.invoke(clean_question)
                if kb_docs:
                    kb_chunks_text = "\n\n".join(doc.page_content for doc in kb_docs)
                    for doc in kb_docs:
                        src = doc.metadata.get("source", "Medical Reference")
                        label = f"📚 Verified KB: {src}"
                        if label not in sources:
                            sources.append(label)
            except Exception as e:
                logger.error("Error querying knowledge base vector store: %s", e)

        # Guard against zero context
        if not report_chunks_text and not kb_chunks_text:
            msg = (
                "No medical reports have been processed yet, and no verified knowledge base is loaded. "
                "Please upload and process a medical report using the sidebar to begin."
            )
            return {
                "response": msg,
                "sources": [],
                "status": "no_context",
            }

        report_context_formatted = (
            report_chunks_text
            if report_chunks_text
            else "[No specific patient report findings available for this query.]"
        )
        kb_context_formatted = (
            kb_chunks_text
            if kb_chunks_text
            else "[No additional reference literature retrieved for this query.]"
        )

        # Construct grounded prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            system_instruction=SAFETY_SYSTEM_INSTRUCTION,
            report_context=report_context_formatted,
            kb_context=kb_context_formatted,
            question=clean_question,
        )

        try:
            if not self._llm_client:
                self._llm_client = genai.Client(api_key=Config.GEMINI_API_KEY)

            response = self._llm_client.models.generate_content(
                model=Config.LLM_MODEL,
                contents=prompt,
            )

            answer = response.text + DISCLAIMER_NOTE

            return {
                "response": answer,
                "sources": sources,
                "status": "success",
            }

        except Exception as e:
            err_msg = str(e)
            logger.error("Gemini LLM call failed: %s", err_msg)
            if "401" in err_msg or "UNAUTHENTICATED" in err_msg:
                user_msg = (
                    "⚠️ **Authentication Notice**: Your `GEMINI_API_KEY` in `.env` is invalid or expired (HTTP 401). "
                    "Please replace it with a valid Gemini API key from [Google AI Studio](https://aistudio.google.com/) "
                    "to generate AI answers."
                )
            else:
                user_msg = f"⚠️ An error occurred while generating your response: {err_msg}."
            return {
                "response": user_msg,
                "sources": sources,
                "status": "llm_error",
            }


# Singleton service instance
medical_rag_service = MedicalRAGService()
