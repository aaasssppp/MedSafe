# MedSafe 🩺

### AI-Powered Medical Report Understanding Assistant

> **Understand your health. Have informed, empowering conversations with your doctor.**

MedSafe is a production-grade Flask web application designed to help patients understand complex medical reports, laboratory panels, and clinical documents in plain, accessible language. 

By combining **automated OCR fallback**, **dual-stream Retrieval-Augmented Generation (RAG)**, **offline incremental knowledge indexing**, **SHA-256 document tracking**, and **strict clinical safety guardrails**, MedSafe transforms intimidating medical test results into structured, understandable insights while rigorously preventing clinical hallucination and unauthorized self-diagnosis.

---

## 📑 Table of Contents

1. [Overview & Philosophy](#-overview--philosophy)
2. [Key Capabilities](#-key-capabilities)
3. [Architecture Overview](#-architecture-overview)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Environment Variables](#-environment-variables)
7. [OCR & Text Extraction Pipeline](#-ocr--text-extraction-pipeline)
8. [Dual-RAG Architecture](#-dual-rag-architecture)
   - [User Document RAG](#1-patient-medical-report-rag)
   - [Verified Medical Knowledge Base RAG](#2-verified-medical-knowledge-base-rag)
9. [Offline Incremental Ingestion CLI](#-offline-incremental-ingestion-cli)
   - [SHA-256 Document Tracking & Duplicate Prevention](#sha-256-document-tracking--duplicate-prevention)
   - [Adding a New Knowledge Base Document](#how-to-add-a-new-knowledge-base-document)
   - [Updating an Existing Knowledge Base Document](#how-to-update-an-existing-knowledge-base-document)
10. [Running the Application](#-running-the-application)
11. [Medical Safety & Guardrails](#-medical-safety--guardrails)
12. [Interview & Presentation Walkthrough](#-interview--presentation-walkthrough)

---

## 💡 Overview & Philosophy

Medical laboratory reports are fraught with dense acronyms (HbA1c, eGFR, ALT, MCV), reference ranges, and numerical values that can trigger severe health anxiety or confusion. Patients frequently turn to search engines, leading to alarmist misinterpretations or erroneous self-medication.

### Core Principles
- **No Diagnostic Claims**: MedSafe never tells a user *"You have disease X"*. It describes what the biological marker evaluates, notes whether a reported number falls within or outside laboratory reference thresholds, and discusses what clinical literature associates with such variations.
- **No Prescription Recommendations**: MedSafe never suggests initiating, modifying, or halting prescription medications.
- **Strict Grounding**: Responses are anchored exclusively in (1) the patient's individual uploaded report findings and (2) verified medical reference guides.
- **Separation of Concerns**: Clinical literature is ingested offline into a persistent vector database and loaded read-only by Flask at startup (zero startup embedding calls). The patient's uploaded reports are processed independently on demand.

---

## ✨ Key Capabilities

1. **Intelligent PDF Ingestion**:
   - Native digital text extraction via PyMuPDF / `pypdf`.
   - Heuristic density check to detect scanned, flattened, or image-only documents.
   - Automatic fallback to high-resolution page rendering and Tesseract OCR.
2. **Initial MedSafe Report Analysis**:
   - Instant structured breakdown upon clicking *"Process Reports"*.
   - Identifies document type, recorded tests, normal vs. out-of-range parameters, and generates proactive questions for the patient's next physician appointment.
3. **Dual-RAG Query Engine**:
   - Retrieves patient-specific metrics from the active session.
   - Retrieves authoritative clinical definitions from the verified medical knowledge base.
   - Blends both sources with demarcated prompt sections for Google Gemini (`gemini-2.5-flash`).
4. **Offline Incremental Ingestion**:
   - Standalone CLI (`python ingestion/ingest_knowledge_base.py`).
   - Computes cryptographic SHA-256 hashes to prevent redundant embeddings.
   - Detects modified documents, deletes superseded chunks via FAISS document IDs, and updates the index seamlessly.
5. **Modern Dark Medical UI**:
   - Purpose-built dark theme with medical teal accents, drag-and-drop report upload, document management with deletion option, live knowledge base status indicator, quick suggestion chips, and responsive chat.

---

## 🏛️ Architecture Overview

```
                      +------------------------------------------+
                      |   VERIFIED MEDICAL KNOWLEDGE BASE        |
                      |   knowledge_base/*.pdf                   |
                      +--------------------+---------------------+
                                           |
                                           v [Offline / On-Demand]
                      +------------------------------------------+
                      |   ingestion/ingest_knowledge_base.py     |
                      |   - SHA-256 Hash Verification            |
                      |   - Chunking & Google Gemini Embeddings  |
                      |   - Incremental Update / Duplicate Check |
                      +--------------------+---------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |   PERSISTENT FAISS KNOWLEDGE BASE        |
                      |   data/vector_store/knowledge_base/      |
                      +--------------------+---------------------+
                                           |
                                           v [Loaded Read-Only at Flask Startup]
+--------------------------+               |
| Patient PDF Report       |               |
+------------+-------------+               |
             |                             |
             v                             |
+--------------------------+               |
| core/pdf_processor.py    |               |
|  - Digital Text Check    |               |
|  - OCR Fallback Pipeline |               |
+------------+-------------+               |
             |                             |
             v                             |
+--------------------------+               |
| Session Report Index     |               |
| data/vector_store/users/ |               |
+------------+-------------+               |
             |                             |
             +--------------+   +----------+
                            |   |
                            v   v
             +----------------------------------+
             |    DUAL-RAG QUERY ENGINE         |
             |    core/rag.py                   |
             |    - Patient Findings Context    |
             |    - Verified KB Context         |
             |    - Safety Directives & Prompt  |
             +------------------+---------------+
                                |
                                v
             +----------------------------------+
             |    GOOGLE GEMINI (LLM)           |
             |    gemini-2.5-flash              |
             +------------------+---------------+
                                |
                                v
             +----------------------------------+
             |    GROUNDED EDUCATIONAL ANSWER   |
             |    + Medical Sources & Footnotes |
             +----------------------------------+
```

---

## 📁 Project Structure

```
MedSafe/
│
├── app.py                             # Flask application routes and REST endpoints
├── config.py                          # Centralized configuration and environment loader
├── requirements.txt                   # Production Python dependencies
├── .env.example                       # Template for environment configuration
├── .env                              # Local active environment variables (API keys)
├── README.md                          # Comprehensive technical documentation
│
├── core/
│   ├── __init__.py                    # Core package public exports
│   ├── pdf_processor.py               # Multi-step text extraction with sufficiency heuristic
│   ├── ocr.py                         # OCR engine (pytesseract + page rendering)
│   ├── chunking.py                    # Medical-aware text chunking with metadata
│   ├── embeddings.py                  # Gemini embeddings with local fallback
│   ├── medical_analyzer.py            # Initial clinical summary generator
│   └── rag.py                         # Dual-retrieval RAG engine and safety prompt router
│
├── ingestion/
│   ├── __init__.py                    # Ingestion package initialization
│   └── ingest_knowledge_base.py       # Offline CLI incremental ingestion script
│
├── knowledge_base/
│   ├── diabetes.pdf                   # Clinical guide: HbA1c, FPG, glycemic markers
│   ├── hypertension.pdf               # Clinical guide: ACC/AHA blood pressure classifications
│   ├── blood_tests.pdf                # Clinical guide: CBC, CMP, Lipid panel reference values
│   ├── medications.pdf                # Clinical guide: Common cardiometabolic medications
│   └── generate_seed_docs.py          # Utility script to generate seed reference PDFs
│
├── data/
│   ├── sample_medical_report.pdf      # Sample patient lab report for immediate testing
│   ├── knowledge_base_metadata.json   # SHA-256 document inventory and chunk index catalog
│   ├── uploads/                       # User uploaded report storage
│   ├── processed/                     # Cached report artifacts
│   └── vector_store/
│       ├── knowledge_base/            # Persisted FAISS index for medical knowledge
│       │   ├── index.faiss
│       │   └── index.pkl
│       └── user_reports/              # Dynamic session FAISS index for active reports
│
├── static/
│   ├── css/
│   │   └── style.css                  # Dark medical theme styling
│   ├── js/
│   │   └── app.js                     # UI interactivity, file uploads, chat client
│   └── images/                        # Branding and medical icons
│
└── templates/
    └── index.html                     # Main application layout
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- (Optional) [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed if you wish to run OCR on scanned/image-based PDFs.

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

Configure your parameters:

```env
# Google AI Studio API Key (Obtain from https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSyYourActualApiKeyHere

# Gemini Model Selection
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/gemini-embedding-001

# Embedding Backend ("gemini" for cloud embeddings, "local" for offline testing)
EMBEDDING_PROVIDER=gemini
AUTO_FALLBACK_EMBEDDINGS=true

# Optional Tesseract OCR path on Windows (leave blank if in PATH)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Text threshold: if digital extraction yields fewer than 50 chars, triggers OCR
OCR_CHAR_THRESHOLD=50
```

---

## 🔍 OCR & Text Extraction Pipeline

Medical reports may arrive as native digital PDFs (exported from a hospital portal) or as low-contrast smartphone scans/faxes.

### Three-Step Decision Pipeline
```
                    PDF Document
                         │
                         ▼
        [Step 1] Attempt Digital Text Extraction
        (PyMuPDF / fitz or pypdf)
                         │
                         ▼
        [Step 2] Sufficiency Check:
        Character count >= OCR_CHAR_THRESHOLD (50)?
                        / \
                   YES /   \ NO
                      /     \
                     v       v
         Use Digital Text   [Step 3] Trigger OCR Engine
                            (Render pages to 200 DPI images
                             -> pytesseract.image_to_string)
                                     │
                                     v
                            Cleaned Extracted Text
```

- **Resilience**: If Tesseract is not installed on the host machine, MedSafe catches `TesseractNotFoundError` gracefully, notifies the user via the UI, and proceeds with digital text extraction rather than crashing.

---

## 🧠 Dual-RAG Architecture

### 1. Patient Medical Report RAG
When the patient uploads one or more PDF reports and clicks **"Process Reports"**:
1. Files are extracted via `core/pdf_processor.py`.
2. Content is split into chunks (~800 characters with 150 overlap) with rich metadata (`source`, `chunk_id`, `chunk_index`).
3. Embeddings are generated and stored in a user FAISS vector store (`data/vector_store/user_reports/`).
4. An **Initial MedSafe Report Analysis** is generated by `core/medical_analyzer.py` and displayed in the main window.

### 2. Verified Medical Knowledge Base RAG
- Curated reference literature located in `knowledge_base/`.
- Pre-indexed into `data/vector_store/knowledge_base/`.
- **Never re-embedded on Flask startup**: Flask loads the serialized FAISS index from disk in milliseconds without making external API calls.

### Combined Retrieval Prompt
When the patient asks: *"What does HbA1c mean in my report and what questions should I ask my doctor?"*
1. Top-k chunks (default: 3) are retrieved from the patient's uploaded report.
2. Top-k chunks (default: 3) are retrieved from the verified clinical literature.
3. The prompt explicitly isolates the two contexts:
   ```
   === PATIENT MEDICAL REPORT FINDINGS ===
   Fasting Plasma Glucose: 138 mg/dL (HIGH)
   Hemoglobin A1c: 7.4 % (HIGH)
   
   === VERIFIED MEDICAL KNOWLEDGE BASE ===
   Normal HbA1c is below 5.7%. Readings >= 6.5% meet the diagnostic threshold for diabetes...
   
   === PATIENT QUESTION ===
   What does HbA1c mean in my report?
   ```
4. Gemini outputs an empathetic, grounded explanation with source attribution.

---

## ⚡ Offline Incremental Ingestion CLI

The knowledge base ingestion is intentionally separated from the web server.

### Basic Commands

#### 1. Ingest All New or Modified Documents
```bash
python ingestion/ingest_knowledge_base.py
```
Scans `knowledge_base/` for all `.pdf` files. Skips documents that are already indexed and unchanged.

#### 2. Process a Specific Document
```bash
python ingestion/ingest_knowledge_base.py --file medications.pdf
```
Processes and indexes only the specified file, merging it into the existing index without touching other documents.

#### 3. View Current Index Status & Metadata
```bash
python ingestion/ingest_knowledge_base.py --status
```
Outputs a formatted table showing all indexed documents, chunk counts, index dates, and SHA-256 prefixes:
```
=================================================================
🩺 MedSafe Verified Knowledge Base - Index Status
=================================================================
Index Location: D:\VScode\MedSafe\data\vector_store\knowledge_base
Metadata File:  D:\VScode\MedSafe\data\knowledge_base_metadata.json
Document Folder:D:\VScode\MedSafe\knowledge_base
FAISS Index on Disk: ✅ Found
-----------------------------------------------------------------
Filename                  Chunks   Indexed Date         SHA-256 (prefix)
-----------------------------------------------------------------
blood_tests.pdf           3        2026-09-21 21:25:17  68fefed811  
diabetes.pdf              2        2026-09-21 21:25:17  45657b44a1  
hypertension.pdf          3        2026-09-21 21:25:17  d57025f198  
medications.pdf           3        2026-09-21 21:27:25  e7b39a101f  
-----------------------------------------------------------------
Total Documents: 4 | Total Indexed Chunks: 11
=================================================================
```

#### 4. Force Full Rebuild
```bash
python ingestion/ingest_knowledge_base.py --rebuild
```

### SHA-256 Document Tracking & Duplicate Prevention
Each processed document is cataloged in `data/knowledge_base_metadata.json`:
```json
{
  "diabetes.pdf": {
    "file": "diabetes.pdf",
    "hash": "45657b44a17ef0d5b4cb54cf17f920daebca1e27a69622d99d34e5a953e5eeb4",
    "processed_at": "2026-09-21T21:25:17.382104",
    "chunks": 2,
    "chunk_ids": [
      "diabetes.pdf__chunk_0",
      "diabetes.pdf__chunk_1"
    ],
    "method": "digital_text",
    "pages": 1
  }
}
```

### How to Add a New Knowledge Base Document
1. Drop your new medical PDF into `knowledge_base/` (e.g. `knowledge_base/thyroid_guide.pdf`).
2. Run single-file ingestion:
   ```bash
   python ingestion/ingest_knowledge_base.py --file thyroid_guide.pdf
   ```
3. MedSafe chunks the file, assigns IDs `thyroid_guide.pdf__chunk_0`, embeds it, and appends it to `data/vector_store/knowledge_base/`. Previous documents are not re-embedded.

### How to Update an Existing Knowledge Base Document
If an existing medical reference (e.g. `medications.pdf`) is revised:
1. Save the updated PDF into `knowledge_base/medications.pdf`.
2. Run:
   ```bash
   python ingestion/ingest_knowledge_base.py --file medications.pdf
   ```
3. MedSafe detects that the SHA-256 hash has changed (`[UPDATE]`).
4. It calls `FAISS.delete(ids=old_chunk_ids)` to cleanly remove obsolete passages.
5. It embeds the new chunks, adds them with new IDs, and updates the metadata file.

---

## 💻 Running the Application

### Start the Flask Server
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:5000
```

### Quick Test with Included Sample Report
1. Open `http://127.0.0.1:5000`.
2. In the left sidebar, click **"Upload Medical Reports"** and select:
   `data/sample_medical_report.pdf` (Metropolitan Healthcare Jane Doe Lab Report).
3. Click **"⚡ Process Reports"**.
4. Observe the **Initial MedSafe Report Analysis** card appear, highlighting:
   - High Fasting Plasma Glucose (138 mg/dL)
   - High HbA1c (7.4%)
   - High Total Cholesterol (228 mg/dL)
   - High Triglycerides (185 mg/dL)
   - Low Protective HDL (44 mg/dL)
   - Suggested discussion points for the patient's physician.
5. Ask any query in the chat box or click one of the suggestion chips:
   - *"What does HbA1c mean in my report?"*
   - *"Are any values outside the reference range?"*
   - *"What questions should I ask my doctor about my lipid panel?"*

---

## 🛡️ Medical Safety & Guardrails

MedSafe is architected around strict safety directives:

1. **Non-Diagnostic Phrasing**:
   - ❌ *Disallowed*: "You have type 2 diabetes and hypertension."
   - ✅ *Allowed*: "Your report lists an HbA1c value of 7.4%, which is above the typical reference interval of 4.0 - 5.6%. According to clinical reference guidelines, values of 6.5% or greater are associated with glycemic elevation. Please discuss this finding with your physician."
2. **No Pharmacotherapy Prescribing**:
   - MedSafe does not recommend dosages, adjustments, or discontinuation of medications.
3. **No Hallucinated Lab Values**:
   - If the patient inquires about an unmeasured test (e.g., Vitamin D or TSH when not present in the uploaded panel), MedSafe explicitly states that the test is absent from the report.
4. **Permanent Prominent Disclaimers**:
   - UI banner, welcome card, and every single generated response conclude with an educational reminder to seek professional clinical advice.

---

## 🎓 Interview & Presentation Walkthrough

When explaining MedSafe in technical interviews or hackathons, highlight these architectural differentiators:

| Feature | Standard Toy Chatbot | MedSafe Architecture |
| :--- | :--- | :--- |
| **Knowledge Base Lifecycle** | Re-indexes all PDFs on every Flask reboot, burning API quota. | **Offline pre-indexing**: Loaded in-memory from serialized FAISS files in 10ms. |
| **Ingestion Pipeline** | Naive loop over directory. | **Incremental SHA-256 tracking**: Detects modifications and deletes outdated chunks via vector IDs. |
| **PDF Extraction** | Crashes on scanned or image PDFs. | **Three-tier extraction**: Digital text extraction -> Sufficiency heuristic -> OCR rendering fallback. |
| **Retrieval Architecture** | Single generic vector index. | **Dual-Stream RAG**: Keeps user patient data isolated from authoritative medical reference literature. |
| **Clinical Safety** | Relies on LLM's internal weights; susceptible to hallucination. | **Strictly Grounded + Guardrailed**: Demarcated context feeds with non-diagnostic system constraints. |

---

## 📄 License

Developed for educational and demonstration purposes.