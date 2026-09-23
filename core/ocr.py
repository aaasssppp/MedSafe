"""
OCR Extraction Module for MedSafe.
Provides optical character recognition for scanned or image-based PDF medical reports.
"""

import os
import io
import logging
from typing import Dict, Any, Tuple

from config import Config

logger = logging.getLogger("medsafe.ocr")

# Configure tesseract binary path if specified in config/env
_TESSERACT_AVAILABLE = False
_PYTESSERACT_ERROR = None

try:
    import pytesseract
    from PIL import Image

    if Config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

    _TESSERACT_AVAILABLE = True
except ImportError as e:
    _PYTESSERACT_ERROR = str(e)


def is_ocr_available() -> Tuple[bool, str]:
    """Check if pytesseract and Tesseract engine are available."""
    if not _TESSERACT_AVAILABLE:
        return False, f"pytesseract or PIL library not installed: {_PYTESSERACT_ERROR}"

    try:
        pytesseract.get_tesseract_version()
        return True, "Tesseract OCR is ready."
    except Exception as e:
        return (
            False,
            f"Tesseract OCR binary not found or failed to execute: {e}. "
            "Please install Tesseract OCR (e.g. from https://github.com/UB-Mannheim/tesseract/wiki) "
            "or set TESSERACT_CMD in your .env file.",
        )

def extract_table_with_gemini_vision(img: "Image.Image", page_num: int) -> str:
    """
    Uses Gemini's vision input to read a table directly from a page image,
    preserving row/column structure that plain OCR (pytesseract.image_to_string)
    cannot — since plain OCR has no concept of rows or columns, only raw text.
    """
    import json
    from google import genai

    api_key = Config.GEMINI_API_KEY
    if not api_key:
        logger.warning("No GEMINI_API_KEY set; cannot run vision table extraction.")
        return ""

    prompt = """Extract every lab test row from this medical report page image.
Return ONLY valid JSON, no other text, in this exact structure:
{"tests": [{"test_name": "", "value": "", "unit": "", "reference_range": "", "flag": ""}]}
If a field is unreadable, use null. Do not invent values not visible in the image.
If there is no table on this page, return {"tests": []}."""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=Config.LLM_MODEL,
            contents=[prompt, img],
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        data = json.loads(raw)
    except Exception as e:
        logger.error("Gemini vision table extraction failed on page %d: %s", page_num, e)
        return ""

    tests = data.get("tests", [])
    if not tests:
        return ""

    rows = [
        f"{t.get('test_name','')}: {t.get('value','')} {t.get('unit','') or ''} "
        f"(Ref: {t.get('reference_range','')}) [{t.get('flag','')}]"
        for t in tests
    ]
    return f"--- Page {page_num} (Gemini Vision Table Extraction) ---\n" + "\n".join(rows)


def extract_text_ocr(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a scanned or image-based PDF using OCR.
    Uses PyMuPDF (fitz) or pypdf to render/extract page images,
    then applies pytesseract.
    """
    if not os.path.exists(pdf_path):
        return {
            "text": "",
            "method": "ocr",
            "page_count": 0,
            "status": "error",
            "error": f"File not found: {pdf_path}",
        }

    # ocr_ok, ocr_msg = is_ocr_available()
    # if not ocr_ok:
    #     logger.warning("OCR fallback requested but OCR is unavailable: %s", ocr_msg)
    #     return {
    #         "text": "",
    #         "method": "ocr",
    #         "page_count": 0,
    #         "status": "ocr_unavailable",
    #         "error": ocr_msg,
    #     }

    extracted_pages = []

    # Strategy 1: Try PyMuPDF (fitz) - high performance page rendering
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        tesseract_ok, _ = is_ocr_available()

        for page_num in range(page_count):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            page_text = extract_table_with_gemini_vision(img, page_num + 1)
            if not page_text and tesseract_ok:
                raw = pytesseract.image_to_string(img)
                if raw.strip():
                    page_text = f"--- Page {page_num + 1} (OCR) ---\n{raw.strip()}"
            if page_text:
                extracted_pages.append(page_text)
    

        doc.close()

        combined_text = "\n\n".join(extracted_pages).strip()
        return {
            "text": combined_text,
            "method": "ocr",
            "page_count": page_count,
            "status": "success",
            "error": None,
        }
    except ImportError:
        logger.info("fitz (PyMuPDF) not available, falling back to pdf2image or pypdf image extraction.")
    except Exception as e:
        logger.error("fitz OCR extraction failed: %s", e)

    # Strategy 2: Try pdf2image
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, dpi=200)
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if text.strip():
                extracted_pages.append(f"--- Page {i + 1} (OCR) ---\n{text.strip()}")

        combined_text = "\n\n".join(extracted_pages).strip()
        return {
            "text": combined_text,
            "method": "ocr",
            "page_count": len(images),
            "status": "success",
            "error": None,
        }
    except ImportError:
        pass
    except Exception as e:
        logger.error("pdf2image extraction failed: %s", e)

    # Strategy 3: Try extracting embedded raster images via pypdf
    try:
        import pypdf

        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)

        for i, page in enumerate(reader.pages):
            page_text = []
            for img_file in page.images:
                img = Image.open(io.BytesIO(img_file.data))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    page_text.append(text.strip())
            if page_text:
                extracted_pages.append(f"--- Page {i + 1} (OCR) ---\n" + "\n".join(page_text))

        combined_text = "\n\n".join(extracted_pages).strip()
        return {
            "text": combined_text,
            "method": "ocr",
            "page_count": page_count,
            "status": "success" if combined_text else "empty",
            "error": None if combined_text else "No text could be recognized via image OCR.",
        }
    except Exception as e:
        logger.error("pypdf image OCR extraction failed: %s", e)
        return {
            "text": "",
            "method": "ocr",
            "page_count": 0,
            "status": "error",
            "error": f"OCR processing failed across all engines: {e}",
        }
