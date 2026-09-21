/**
 * MedSafe - Client Application JavaScript
 * Coordinates report uploads, OCR processing, clinical analysis display, and chat interaction.
 */

// Escape utility to prevent XSS
function escapeHTML(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Render markdown to HTML safely with fallback
function renderMarkdown(md) {
    if (typeof marked !== "undefined" && marked.parse) {
        return marked.parse(md);
    }
    // Simple fallback if CDN is offline
    return escapeHTML(md).replace(/\n/g, "<br>");
}

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    fetchStatus();

    // Setup file input listener
    const fileInput = document.getElementById("pdf-upload");
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files);
            }
        });
    }

    // Setup drag and drop
    const dropZone = document.getElementById("drop-zone");
    if (dropZone) {
        ["dragenter", "dragover"].forEach(evt => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            });
        });

        dropZone.addEventListener("drop", (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files);
            }
        });

        dropZone.addEventListener("click", () => {
            fileInput.click();
        });
    }

    // Setup Enter key on chat input
    const msgInput = document.getElementById("message-input");
    if (msgInput) {
        msgInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
}

/**
 * Fetch application status (knowledge base status, uploaded files)
 */
async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();

        // Update KB status card
        const kbDot = document.getElementById("kb-dot");
        const kbDetails = document.getElementById("kb-details");

        if (data.kb_indexed) {
            kbDot.classList.add("active");
            kbDetails.innerHTML = `🟢 Ready &bull; ${data.kb_chunks_count} knowledge passages`;
        } else {
            kbDot.classList.remove("active");
            kbDetails.innerHTML = `⚠️ Not built &bull; Run <code>python ingestion/ingest_knowledge_base.py</code>`;
        }

        // Update uploaded files list
        renderUploadedFiles(data.uploaded_files || []);

        // Update session badge
        const badge = document.getElementById("session-badge");
        if (data.user_reports_processed) {
            badge.classList.add("ready");
            badge.innerText = `Report Indexed (${data.user_chunks_count} chunks)`;
        } else if (data.uploaded_files && data.uploaded_files.length > 0) {
            badge.classList.remove("ready");
            badge.innerText = `${data.uploaded_files.length} report(s) uploaded - click Process`;
        } else {
            badge.classList.remove("ready");
            badge.innerText = "No active report";
        }

    } catch (err) {
        console.error("Status fetch error:", err);
    }
}

/**
 * Render list of uploaded medical reports in the sidebar
 */
function renderUploadedFiles(files) {
    const list = document.getElementById("pdf-list");
    const countBadge = document.getElementById("report-count");
    if (!list) return;

    countBadge.innerText = files.length;

    if (files.length === 0) {
        list.innerHTML = `
            <div class="empty-state" id="empty-docs-msg">
                No reports uploaded yet. Upload a laboratory or clinical PDF report to begin.
            </div>`;
        return;
    }

    list.innerHTML = "";
    files.forEach((file) => {
        const item = document.createElement("div");
        item.className = "pdf-item";
        item.innerHTML = `
            <div class="pdf-item-info">
                <span>📄</span>
                <div>
                    <div class="pdf-item-name" title="${escapeHTML(file.filename)}">${escapeHTML(file.filename)}</div>
                    <div class="pdf-item-size">${file.size_kb} KB</div>
                </div>
            </div>
            <button class="btn-delete" title="Delete document" onclick="deleteDocument('${escapeHTML(file.filename)}')">
                🗑️
            </button>
        `;
        list.appendChild(item);
    });
}

/**
 * Handle file upload via input or drag-and-drop
 */
async function handleFileUpload(fileList) {
    const formData = new FormData();
    let hasPdf = false;

    for (let file of fileList) {
        if (file.name.toLowerCase().endsWith(".pdf")) {
            formData.append("pdf", file);
            hasPdf = true;
        }
    }

    if (!hasPdf) {
        alert("Please select PDF medical reports only.");
        return;
    }

    const dropZone = document.getElementById("drop-zone");
    const origText = dropZone.innerHTML;
    dropZone.innerHTML = `<span>⏳ Uploading medical reports...</span>`;

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        if (data.success) {
            renderUploadedFiles(data.all_files || []);
            document.getElementById("session-badge").innerText = "Ready to process";
        } else {
            alert("Upload failed: " + (data.error || data.errors?.join(", ") || "Unknown error"));
        }
    } catch (err) {
        alert("Network error while uploading: " + err.message);
    } finally {
        dropZone.innerHTML = origText;
        document.getElementById("pdf-upload").value = "";
    }
}

/**
 * Delete a specific uploaded report
 */
async function deleteDocument(filename) {
    if (!confirm(`Remove "${filename}" from your session?`)) return;

    try {
        const res = await fetch("/delete-file", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: filename }),
        });

        const data = await res.json();
        if (data.success) {
            renderUploadedFiles(data.remaining_files || []);
            fetchStatus();
        } else {
            alert("Error deleting file: " + (data.error || "Unknown"));
        }
    } catch (err) {
        alert("Network error: " + err.message);
    }
}

/**
 * Process uploaded reports: triggers OCR fallback if needed, builds user FAISS index,
 * and renders initial structured analysis.
 */
async function processReports() {
    const processBtn = document.getElementById("process-btn");
    const spinner = document.getElementById("process-spinner");
    const btnText = document.getElementById("process-btn-text");

    processBtn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.innerText = "Extracting & Analyzing...";

    try {
        const res = await fetch("/process", { method: "POST" });
        const data = await res.json();

        if (data.success) {
            // Append Initial Analysis Card to Chat
            renderInitialAnalysis(data.summary, data.files, data.total_chunks);
            fetchStatus();
        } else {
            alert("Report Processing Notice: " + (data.error || "Unable to extract report."));
        }
    } catch (err) {
        alert("Processing request failed: " + err.message);
    } finally {
        processBtn.disabled = false;
        spinner.classList.add("hidden");
        btnText.innerText = "⚡ Process Reports";
    }
}

/**
 * Render the Initial MedSafe Report Analysis card in the main area
 */
function renderInitialAnalysis(summaryMarkdown, files, totalChunks) {
    const chatBox = document.getElementById("chat-box");

    // Extraction badge breakdown
    const extractionBadges = files.map(f => {
        const methodTag = f.method === "ocr" ? "🔍 OCR" : "📄 Digital Text";
        return `<span class="source-pill">${escapeHTML(f.filename)} (${methodTag}, ${f.chunks} chunks)</span>`;
    }).join(" ");

    const htmlContent = `
        <div class="bot-msg-wrapper">
            <div class="bot-avatar">📋</div>
            <div class="bot-msg" style="border-left: 3px solid var(--primary-hover); width: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-weight:700; color:var(--primary-hover); font-size:15px;">
                        Initial MedSafe Report Analysis
                    </span>
                    <span class="badge" style="background:var(--primary-light); color:var(--primary-hover);">
                        ${totalChunks} Chunks Indexed
                    </span>
                </div>
                <div style="margin-bottom:12px;">
                    ${extractionBadges}
                </div>
                <div class="bot-content">
                    ${renderMarkdown(summaryMarkdown)}
                </div>
            </div>
        </div>
    `;

    chatBox.insertAdjacentHTML("beforeend", htmlContent);
    chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Send user chat message and receive dual-RAG grounded response
 */
async function sendMessage() {
    const input = document.getElementById("message-input");
    const chatBox = document.getElementById("chat-box");
    const text = input.value.trim();

    if (!text) return;

    // Render user message
    const userHtml = `
        <div class="user-msg-wrapper">
            <div class="user-msg">${escapeHTML(text)}</div>
        </div>
    `;
    chatBox.insertAdjacentHTML("beforeend", userHtml);
    input.value = "";

    // Render loading indicator
    const loadingId = "loading-" + Date.now();
    const loadingHtml = `
        <div class="bot-msg-wrapper" id="${loadingId}">
            <div class="bot-avatar">🩺</div>
            <div class="bot-msg" style="color:var(--text-muted);">
                <span class="spinner" style="vertical-align:middle; margin-right:8px;"></span>
                MedSafe is consulting your report and medical literature...
            </div>
        </div>
    `;
    chatBox.insertAdjacentHTML("beforeend", loadingHtml);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        const data = await res.json();
        const loadingElem = document.getElementById(loadingId);

        let sourcesHtml = "";
        if (data.sources && data.sources.length > 0) {
            const pills = data.sources.map(s => `<span class="source-pill">${escapeHTML(s)}</span>`).join("");
            sourcesHtml = `<div class="sources-container"><strong>Grounding Sources:</strong> ${pills}</div>`;
        }

        const botResponseHtml = `
            <div class="bot-avatar">🩺</div>
            <div class="bot-msg">
                <div class="bot-content">
                    ${renderMarkdown(data.response || "No response generated.")}
                </div>
                ${sourcesHtml}
            </div>
        `;

        if (loadingElem) {
            loadingElem.innerHTML = botResponseHtml;
        }

    } catch (err) {
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) {
            loadingElem.innerHTML = `
                <div class="bot-avatar">⚠️</div>
                <div class="bot-msg" style="color:salmon;">
                    Error processing question: ${escapeHTML(err.message)}
                </div>
            `;
        }
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Handle quick suggestion chip click
 */
function askSuggestion(questionText) {
    const input = document.getElementById("message-input");
    input.value = questionText;
    sendMessage();
}

/**
 * Clear chat history and restore welcome message
 */
function clearChat() {
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = `
        <div class="bot-msg-wrapper">
            <div class="bot-avatar">🩺</div>
            <div class="bot-msg">
                <div class="bot-content">
                    <h3>Conversation Reset</h3>
                    <p>Your chat history has been cleared. You can ask further questions about your uploaded report or verified medical references below.</p>
                </div>
            </div>
        </div>
    `;
}
