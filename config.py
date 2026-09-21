import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Application configuration."""

    # Project directories
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
    PROCESSED_FOLDER = os.path.join(BASE_DIR, "data", "processed")
    KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
    KB_VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store", "knowledge_base")
    USER_VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store", "user_reports")
    KB_METADATA_PATH = os.path.join(BASE_DIR, "data", "knowledge_base_metadata.json")

    # API and Model Settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
    AUTO_FALLBACK_EMBEDDINGS = os.getenv("AUTO_FALLBACK_EMBEDDINGS", "true").lower() in ("true", "1", "yes")

    # OCR and Extraction Settings
    OCR_CHAR_THRESHOLD = int(os.getenv("OCR_CHAR_THRESHOLD", "50"))
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

    # Security & Upload constraints
    SECRET_KEY = os.getenv("SECRET_KEY", "medsafe-dev-secret-key-change-in-prod")
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload
    ALLOWED_EXTENSIONS = {"pdf"}

    # RAG Retrieval parameters
    USER_REPORT_TOP_K = int(os.getenv("USER_REPORT_TOP_K", "3"))
    KB_TOP_K = int(os.getenv("KB_TOP_K", "3"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


# Ensure required directories exist at configuration load time
for folder in [
    Config.UPLOAD_FOLDER,
    Config.PROCESSED_FOLDER,
    Config.KB_DIR,
    Config.KB_VECTOR_STORE_PATH,
    Config.USER_VECTOR_STORE_PATH,
]:
    os.makedirs(folder, exist_ok=True)
