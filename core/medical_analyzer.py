"""
Initial Medical Report Analyzer for MedSafe.
Generates an educational, non-diagnostic overview of processed patient reports.
"""

import logging
import re
from typing import Dict, Any
from google import genai

from config import Config

logger = logging.getLogger("medsafe.analyzer")

ANALYSIS_SYSTEM_PROMPT = """You are MedSafe, a professional AI medical report assistant designed to help patients understand their health documents in plain, accessible language.

CRITICAL MEDICAL SAFETY RULES:
1. NEVER diagnose the patient with a condition or disease (e.g. NEVER say "You have diabetes", instead say "The report shows an HbA1c value of 7.2%, which is higher than the standard laboratory reference range of 4.0-5.6%").
2. NEVER prescribe, change, or advise discontinuing any medication.
3. NEVER make definitive prognostic statements.
4. Distinguish between patient report findings and general medical educational context.
5. Emphasize that all lab tests must be interpreted by the patient's healthcare provider in the context of their full clinical history.
"""

ANALYSIS_USER_PROMPT = """Please review the following extracted medical report text and produce an initial educational summary.

EXTRACTED REPORT TEXT:
\"\"\"
{report_text}
\"\"\"

Please organize your response into the following clear markdown sections:

### 📋 Report Overview
- Identify the type of document (e.g. Complete Blood Count, Lipid Panel, Metabolic Profile, Prescription, Scanned Clinical Notes).
- Note patient details if visible (e.g. report date, ordering clinic; omit private identifiers like phone numbers).

### 🔬 Key Laboratory Findings
- Present a clean markdown table or bulleted list of key tests found in the report with their:
  - Test Name
  - Observed Value & Unit
  - Reference Range (as printed in the report)
  - General Status (Within range / Elevated / Low)

### ⚠️ Observations & Noteworthy Parameters
- Highlight parameters that are outside reference intervals in an objective, non-alarmist manner.
- Explain in simple terms what that specific biological marker measures.

### 💬 Constructive Questions to Ask Your Doctor
- Provide 3 to 4 specific, empowering questions the user can bring to their physician's appointment based on these specific findings.

### 🛡️ Medical Safety Note
- Conclude with a brief reminder that this analysis is educational and only a qualified medical practitioner can provide clinical diagnosis and treatment plans.
"""


def generate_initial_report_analysis(report_text: str) -> Dict[str, Any]:
    """
    Generate an initial structured educational analysis of uploaded medical reports.
    """
    if not report_text or len(report_text.strip()) < 20:
        return {
            "summary": (
                "⚠️ **Report Notice**: The uploaded document contains insufficient or unreadable text. "
                "Please verify that the uploaded PDF contains clear laboratory results or legible clinical notes."
            ),
            "status": "insufficient_text",
        }

    # Truncate text if excessively long to prevent token overflow
    truncated_text = report_text[:15000]

    api_key = Config.GEMINI_API_KEY
    if not api_key:
        # Fallback heuristic summary when API key is missing
        return {
            "summary": _generate_fallback_summary(report_text),
            "status": "fallback_no_api_key",
        }

    try:
        client = genai.Client(api_key=api_key)
        prompt = ANALYSIS_SYSTEM_PROMPT + "\n\n" + ANALYSIS_USER_PROMPT.format(report_text=truncated_text)

        response = client.models.generate_content(
            model=Config.LLM_MODEL,
            contents=prompt,
        )

        return {
            "summary": response.text,
            "status": "success",
        }
    except Exception as e:
        logger.error("Error generating medical analysis via Gemini: %s", e)
        # Fallback to local heuristic summary rather than crashing
        fallback = _generate_fallback_summary(report_text)
        return {
            "summary": f"{fallback}\n\n*(Note: AI-enhanced analysis was unavailable: {e})*",
            "status": "partial_fallback",
        }


def _generate_fallback_summary(text: str) -> str:
    """
    Heuristic rule-based summary extractor used if Gemini API is temporarily offline.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview = "\n".join(lines[:12])

    return f"""### 📋 Medical Report Processed Successfully

**Extracted Report Preview:**
```
{preview}
```

### 🔬 Document Status
- The report text has been extracted and indexed into your active session.
- You can now ask specific questions about the values, terms, or reference ranges in the chat below.

### 🛡️ Medical Safety Reminder
*MedSafe provides informational assistance to help you understand your medical reports. Always discuss any laboratory findings, abnormal values, or treatment plans directly with your licensed physician.*
"""
