"""
Offline Incremental Knowledge Base Ingestion Script for MedSafe.

Usage:
  # Scan knowledge_base/ and index new/modified documents:
  python ingestion/ingest_knowledge_base.py

  # Process a specific document:
  python ingestion/ingest_knowledge_base.py --file medications.pdf

  # Force rebuild the entire knowledge base:
  python ingestion/ingest_knowledge_base.py --rebuild

  # Check current indexing status:
  python ingestion/ingest_knowledge_base.py --status
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_community.vectorstores import FAISS

# Ensure project root is in sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from core.embeddings import GeminiEmbeddings
from core.pdf_processor import extract_pdf_text
from core.chunking import split_medical_text

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure CLI logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("medsafe.ingest")


def calculate_file_hash(filepath: str) -> str:
    """Calculate the SHA-256 cryptographic hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_metadata() -> Dict[str, Any]:
    """Load indexing metadata tracking file."""
    meta_path = Config.KB_METADATA_PATH
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read metadata file (%s), initializing new metadata.", e)
            return {}
    return {}


def save_metadata(metadata: Dict[str, Any]) -> None:
    """Save indexing metadata atomically."""
    meta_path = Config.KB_METADATA_PATH
    temp_path = meta_path + ".tmp"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(temp_path, meta_path)


def print_status(metadata: Dict[str, Any]) -> None:
    """Print current indexing status and document inventory."""
    print("=" * 65)
    print("🩺 MedSafe Verified Knowledge Base - Index Status")
    print("=" * 65)
    print(f"Index Location: {Config.KB_VECTOR_STORE_PATH}")
    print(f"Metadata File:  {Config.KB_METADATA_PATH}")
    print(f"Document Folder:{Config.KB_DIR}")

    index_exists = os.path.exists(os.path.join(Config.KB_VECTOR_STORE_PATH, "index.faiss"))
    print(f"FAISS Index on Disk: {'✅ Found' if index_exists else '❌ Not Built'}")
    print("-" * 65)

    if not metadata:
        print("No documents currently recorded in metadata.")
    else:
        print(f"{'Filename':<25} {'Chunks':<8} {'Indexed Date':<20} {'SHA-256 (prefix)':<12}")
        print("-" * 65)
        total_chunks = 0
        for fname, info in metadata.items():
            chunks = info.get("chunks", 0)
            total_chunks += chunks
            date_str = info.get("processed_at", "")[:19].replace("T", " ")
            hash_prefix = info.get("hash", "")[:10]
            print(f"{fname:<25} {chunks:<8} {date_str:<20} {hash_prefix:<12}")
        print("-" * 65)
        print(f"Total Documents: {len(metadata)} | Total Indexed Chunks: {total_chunks}")
    print("=" * 65)


def process_document(
    filepath: str,
    metadata: Dict[str, Any],
    vector_store: Optional[FAISS],
    embeddings: GeminiEmbeddings,
) -> Tuple[Optional[FAISS], bool]:
    """
    Process a single verified medical document with hash check and incremental update.

    Returns:
        (updated_vector_store, was_processed_bool)
    """
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        print(f"[ERROR] File does not exist: {filepath}")
        return vector_store, False

    current_hash = calculate_file_hash(filepath)
    is_modified = False
    old_chunk_ids: List[str] = []

    # Check if document has already been processed
    if filename in metadata:
        existing_info = metadata[filename]
        if existing_info.get("hash") == current_hash:
            print(
                f"[SKIP] '{filename}' is already indexed and unchanged. "
                f"({existing_info.get('chunks', 0)} chunks, hash: {current_hash[:10]}...)"
            )
            return vector_store, False
        else:
            print(f"[UPDATE] '{filename}' content was modified! Re-indexing with updated hash...")
            is_modified = True
            old_chunk_ids = existing_info.get("chunk_ids", [])
    else:
        print(f"[NEW] Found new medical document: '{filename}'")

    # Step 1: Extract text
    extraction = extract_pdf_text(filepath)
    raw_text = extraction.get("text", "")
    if not raw_text.strip():
        print(f"[WARNING] No readable text could be extracted from '{filename}'. Skipping.")
        return vector_store, False

    # Step 2: Split text into chunks
    chunks = split_medical_text(
        raw_text,
        source_name=filename,
        extra_metadata={"category": "verified_medical_reference"},
    )
    if not chunks:
        print(f"[WARNING] No valid chunks produced for '{filename}'. Skipping.")
        return vector_store, False

    new_chunk_ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]

    # Step 3: Handle modified document cleanup
    if is_modified and vector_store is not None and old_chunk_ids:
        try:
            vector_store.delete(ids=old_chunk_ids)
            print(f"       Removed {len(old_chunk_ids)} superseded chunks from FAISS index.")
        except Exception as e:
            logger.warning("Could not remove old chunks via delete(): %s", e)

    # Step 4: Add new chunks to FAISS index
    if vector_store is not None:
        vector_store.add_documents(chunks, ids=new_chunk_ids)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings, ids=new_chunk_ids)

    # Step 5: Save index immediately to disk
    vector_store.save_local(Config.KB_VECTOR_STORE_PATH)

    # Step 6: Record document metadata
    metadata[filename] = {
        "file": filename,
        "hash": current_hash,
        "processed_at": datetime.now().isoformat(),
        "chunks": len(chunks),
        "chunk_ids": new_chunk_ids,
        "method": extraction.get("method", "digital_text"),
        "pages": extraction.get("page_count", 0),
    }
    save_metadata(metadata)

    print(
        f"[SUCCESS] Indexed '{filename}' -> {len(chunks)} chunks "
        f"({extraction.get('method')}, {extraction.get('page_count')} pages)."
    )
    return vector_store, True


def run_ingestion(file_arg: Optional[str] = None, rebuild: bool = False) -> None:
    """Main ingestion coordinator."""
    print("\n" + "=" * 65)
    print("🩺 MedSafe Verified Knowledge Base - Ingestion Pipeline")
    print("=" * 65)

    if not Config.GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY is not configured in .env.")
        print("Please add your Gemini API key to .env before running ingestion.")
        sys.exit(1)

    embeddings = GeminiEmbeddings()
    metadata = {} if rebuild else load_metadata()

    # Load existing FAISS index if not rebuilding
    vector_store = None
    index_file = os.path.join(Config.KB_VECTOR_STORE_PATH, "index.faiss")

    if not rebuild and os.path.exists(index_file):
        try:
            vector_store = FAISS.load_local(
                Config.KB_VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Loaded existing FAISS index from {Config.KB_VECTOR_STORE_PATH}")
        except Exception as e:
            logger.warning("Could not load existing FAISS index (%s). A new one will be created.", e)
            vector_store = None
    elif rebuild:
        print("[REBUILD] Rebuilding knowledge base from scratch. Existing index will be replaced.")

    # Determine files to process
    files_to_process: List[str] = []

    if file_arg:
        # User specified a specific document
        target_path = file_arg
        if not os.path.isabs(target_path):
            if os.path.exists(os.path.join(Config.KB_DIR, target_path)):
                target_path = os.path.join(Config.KB_DIR, target_path)
            elif os.path.exists(target_path):
                target_path = os.path.abspath(target_path)

        if not os.path.exists(target_path):
            print(f"❌ File not found: '{file_arg}' in '{Config.KB_DIR}' or current working directory.")
            sys.exit(1)

        files_to_process = [target_path]
    else:
        # Scan knowledge_base directory
        if not os.path.exists(Config.KB_DIR):
            os.makedirs(Config.KB_DIR, exist_ok=True)

        candidate_files = [
            os.path.join(Config.KB_DIR, f)
            for f in os.listdir(Config.KB_DIR)
            if f.lower().endswith(".pdf")
        ]

        if not candidate_files:
            print(f"⚠️ No PDF documents found in '{Config.KB_DIR}'.")
            print("Place medical reference PDFs in that directory, then re-run this script.")
            return

        files_to_process = sorted(candidate_files)

    print(f"Found {len(files_to_process)} candidate document(s) to inspect.\n")

    processed_count = 0
    for fpath in files_to_process:
        vector_store, was_processed = process_document(
            filepath=fpath,
            metadata=metadata,
            vector_store=vector_store,
            embeddings=embeddings,
        )
        if was_processed:
            processed_count += 1

    print("\n" + "-" * 65)
    print(f"Ingestion completed. {processed_count} document(s) added or updated.")
    print(f"Persistent vector store saved to: {Config.KB_VECTOR_STORE_PATH}")
    print(f"Metadata catalog saved to:        {Config.KB_METADATA_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedSafe Verified Medical Knowledge Base Ingestion CLI"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific PDF file to index (e.g. --file medications.pdf)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the entire knowledge base from scratch, ignoring previous index",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current indexing status and document inventory",
    )

    args = parser.parse_args()

    if args.status:
        meta = load_metadata()
        print_status(meta)
    else:
        run_ingestion(file_arg=args.file, rebuild=args.rebuild)
