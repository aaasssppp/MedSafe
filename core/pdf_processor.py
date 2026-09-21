"""
PDF Processing and Text Extraction Pipeline for MedSafe.
Implements multi-step extraction:
1. Native digital text extraction (via PyMuPDF / pypdf)
2. Sufficiency heuristic check (character/word density)
3. Automatic fallback to OCR for scanned or image-only documents
"""

import os
import re
import logging
from typing import Dict, Any, List

from config import Config
from core.ocr import extract_text_ocr

logger = logging.getLogger("medsafe.pdf_processor")


def clean_medical_text(text: str) -> str:
    """
    Clean and normalize extracted medical report text.
    Preserves table columns, numerical readings, units, and ranges.
    """
    if not text:
        return ""

    # Replace non-breaking spaces and exotic whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace 3 or more consecutive newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace but preserve line breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    return "\n".join(lines).strip()


def extract_pdf_text(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a medical PDF with automated OCR fallback.

    Returns:
        dict: {
            "text": str,
            "method": "digital_text" | "ocr" | "none",
            "page_count": int,
            "status": "success" | "ocr_fallback" | "empty" | "error",
            "message": str
        }
    """
    if not os.path.exists(pdf_path):
        return {
            "text": "",
            "method": "none",
            "page_count": 0,
            "status": "error",
            "message": f"File does not exist: {pdf_path}",
        }

    raw_text = ""
    page_count = 0
    pages_text: List[str] = []

    # -------------------------------------------------------------
    # STEP 1: Attempt native digital text extraction
    # -------------------------------------------------------------
    used_fitz = False
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        for i in range(page_count):
            p_text = doc[i].get_text()
            if p_text and p_text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{p_text.strip()}")
        doc.close()
        raw_text = "\n\n".join(pages_text).strip()
        used_fitz = True
    except ImportError:
        logger.info("fitz not installed, trying pypdf...")
    except Exception as e:
        logger.warning("fitz extraction encountered an issue: %s", e)

    if not used_fitz:
        try:
            import pypdf

            reader = pypdf.PdfReader(pdf_path)
            page_count = len(reader.pages)
            for i, page in enumerate(reader.pages):
                p_text = page.extract_text() or ""
                if p_text.strip():
                    pages_text.append(f"--- Page {i + 1} ---\n{p_text.strip()}")
            raw_text = "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.error("pypdf extraction failed: %s", e)

    # -------------------------------------------------------------
    # STEP 2: Check whether meaningful text was extracted
    # -------------------------------------------------------------
    # Calculate non-whitespace character count
    non_ws_chars = len(re.sub(r"\s+", "", raw_text))

    if non_ws_chars >= Config.OCR_CHAR_THRESHOLD:
        cleaned = clean_medical_text(raw_text)
        logger.info(
            "Extracted %d chars via digital text extraction from %s (%d pages).",
            len(cleaned),
            os.path.basename(pdf_path),
            page_count,
        )
        return {
            "text": cleaned,
            "method": "digital_text",
            "page_count": page_count,
            "status": "success",
            "message": f"Extracted via digital text ({page_count} pages).",
        }

    # -------------------------------------------------------------
    # STEP 3: Fallback to OCR if text is empty or insufficient
    # -------------------------------------------------------------
    logger.info(
        "Digital text insufficient (%d chars < %d threshold) for %s. Triggering OCR pipeline...",
        non_ws_chars,
        Config.OCR_CHAR_THRESHOLD,
        os.path.basename(pdf_path),
    )

    ocr_result = extract_text_ocr(pdf_path)
    ocr_text = ocr_result.get("text", "")
    ocr_non_ws = len(re.sub(r"\s+", "", ocr_text))

    if ocr_non_ws >= Config.OCR_CHAR_THRESHOLD:
        cleaned = clean_medical_text(ocr_text)
        return {
            "text": cleaned,
            "method": "ocr",
            "page_count": ocr_result.get("page_count", page_count),
            "status": "success",
            "message": f"Extracted via OCR fallback ({ocr_result.get('page_count', page_count)} pages).",
        }

    # If OCR didn't produce enough text or wasn't available
    if ocr_result.get("status") == "ocr_unavailable":
        return {
            "text": clean_medical_text(raw_text),
            "method": "digital_text" if raw_text else "none",
            "page_count": page_count,
            "status": "ocr_unavailable",
            "message": (
                f"Digital text extraction found minimal content ({non_ws_chars} chars), "
                f"and OCR is currently unavailable: {ocr_result.get('error')}"
            ),
        }

    return {
        "text": clean_medical_text(raw_text),
        "method": "ocr" if ocr_text else "digital_text",
        "page_count": page_count,
        "status": "empty" if not raw_text and not ocr_text else "success",
        "message": (
            "Extracted text has limited density. Report may be handwritten, low contrast, or blank."
        ),
    }
