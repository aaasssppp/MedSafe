"""
MedSafe: AI-Powered Medical Report Understanding Assistant.
Flask application server.
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from core.rag import medical_rag_service

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("medsafe.app")

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_current_uploaded_files():
    """Retrieve list of files currently in uploads directory with sizes."""
    files = []
    if os.path.exists(Config.UPLOAD_FOLDER):
        for fname in sorted(os.listdir(Config.UPLOAD_FOLDER)):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(Config.UPLOAD_FOLDER, fname)
                try:
                    size_bytes = os.path.getsize(fpath)
                    files.append({
                        "filename": fname,
                        "size_bytes": size_bytes,
                        "size_kb": round(size_bytes / 1024, 1),
                    })
                except OSError:
                    continue
    return files


@app.route("/")
def index():
    """Render main application interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return system readiness, knowledge base state, and uploaded document inventory."""
    status = medical_rag_service.get_status()
    status["uploaded_files"] = get_current_uploaded_files()
    return jsonify(status)


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Handle one or more PDF report uploads.
    Saves to data/uploads/ folder without indexing yet.
    Indexing occurs when the user clicks 'Process Reports'.
    """
    if "pdf" not in request.files:
        # Check alternative key
        files = request.files.getlist("files") or request.files.getlist("pdf[]")
    else:
        files = request.files.getlist("pdf")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"success": False, "error": "No files selected for upload."}), 400

    saved_files = []
    errors = []

    for file in files:
        if file and file.filename:
            if not allowed_file(file.filename):
                errors.append(f"'{file.filename}' is not a supported PDF document.")
                continue

            safe_name = secure_filename(file.filename)
            if not safe_name:
                safe_name = f"report_{len(saved_files) + 1}.pdf"

            destination = os.path.join(Config.UPLOAD_FOLDER, safe_name)
            try:
                file.save(destination)
                size_bytes = os.path.getsize(destination)
                saved_files.append({
                    "filename": safe_name,
                    "size_bytes": size_bytes,
                    "size_kb": round(size_bytes / 1024, 1),
                })
                logger.info("Saved uploaded report: %s (%d bytes)", safe_name, size_bytes)
            except Exception as e:
                errors.append(f"Failed to save '{file.filename}': {e}")

    return jsonify({
        "success": len(saved_files) > 0,
        "uploaded": saved_files,
        "errors": errors,
        "all_files": get_current_uploaded_files(),
    })


@app.route("/delete-file", methods=["POST"])
def delete_file():
    """Delete a specific uploaded report."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")

    if not filename:
        return jsonify({"success": False, "error": "No filename specified."}), 400

    safe_name = secure_filename(os.path.basename(filename))
    filepath = os.path.join(Config.UPLOAD_FOLDER, safe_name)

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info("Deleted uploaded file: %s", safe_name)
            # Reset user report index when files change
            medical_rag_service.clear_user_reports()
            return jsonify({
                "success": True,
                "deleted": safe_name,
                "remaining_files": get_current_uploaded_files(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": f"Could not delete file: {e}"}), 500

    return jsonify({"success": False, "error": "File not found."}), 404


@app.route("/clear-uploads", methods=["POST"])
def clear_uploads():
    """Clear all uploaded user reports and reset the user report index."""
    medical_rag_service.clear_user_reports()
    deleted_count = 0

    if os.path.exists(Config.UPLOAD_FOLDER):
        for fname in os.listdir(Config.UPLOAD_FOLDER):
            fpath = os.path.join(Config.UPLOAD_FOLDER, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted_count += 1
            except Exception as e:
                logger.warning("Error deleting file %s: %e", fname, e)

    return jsonify({"success": True, "deleted_count": deleted_count})


@app.route("/process", methods=["POST"])
def process_reports():
    """
    Process all currently uploaded medical reports:
    1. Extracts text with automatic OCR fallback
    2. Chunks and indexes patient report into user vector store
    3. Generates initial structured medical summary
    """
    current_files = get_current_uploaded_files()
    if not current_files:
        return jsonify({
            "success": False,
            "error": "No medical reports uploaded yet. Please upload at least one PDF report first.",
        }), 400

    filepaths = [os.path.join(Config.UPLOAD_FOLDER, f["filename"]) for f in current_files]
    logger.info("Processing %d medical report(s)...", len(filepaths))

    result = medical_rag_service.process_user_reports(filepaths)

    if result.get("status") == "error":
        return jsonify({"success": False, "error": result.get("message")}), 500

    return jsonify({
        "success": True,
        "summary": result.get("summary", ""),
        "files": result.get("files", []),
        "total_chunks": result.get("total_chunks", 0),
        "message": result.get("message", "Reports processed successfully."),
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint for patient questions.
    Uses dual-RAG (Patient Report + Verified Knowledge Base) and safety guardrails.
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "response": "Please enter a question about your medical report.",
            "sources": [],
        }), 400

    result = medical_rag_service.query(message)
    return jsonify({
        "response": result.get("response", ""),
        "sources": result.get("sources", []),
        "status": result.get("status", "success"),
    })


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🩺 Starting MedSafe Medical Report Assistant Server")
    print(f"Knowledge Base Loaded: {'✅ YES' if medical_rag_service.is_knowledge_base_ready() else '⚠️ NO (run ingestion/ingest_knowledge_base.py)'}")
    print("Web Interface URL:     http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
