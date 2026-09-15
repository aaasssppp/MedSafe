# MedSafe 🩺

### AI-Powered Medical Report Understanding Assistant

> Understand your health. Make informed conversations with your doctor.

MedSafe is an AI-powered healthcare assistant designed to help individuals understand complex medical reports and prescriptions in simple, easy-to-understand language.

By combining OCR, structured data extraction, Retrieval-Augmented Generation (RAG), Generative AI, and safety guardrails, MedSafe transforms complicated medical documents into clear and understandable information.

MedSafe is designed for educational purposes and does not replace professional medical advice, diagnosis, or treatment.

---

## 🚨 Problem Statement

Medical reports often contain complex terminology, abbreviations, and numerical values that are difficult for patients to understand.

For many people:

- Medical terminology is confusing.
- Abnormal test values may cause unnecessary anxiety.
- Patients may not understand what their reports indicate.
- Important information can be overlooked.
- Access to understandable medical information is limited.

Patients frequently search online for interpretations of their reports, which can sometimes lead to misinformation or inaccurate self-diagnosis.

---

## 💡 Our Solution

MedSafe acts as an intelligent medical information assistant that helps users understand their medical reports before consulting a healthcare professional.

Users can upload a blood test report, laboratory report, or prescription, and MedSafe extracts relevant information, explains medical terminology, identifies values outside the provided reference ranges, and provides educational context from verified medical sources.

### Core Philosophy

> MedSafe does not diagnose diseases or prescribe medications. It helps users understand medical information safely and responsibly.

---

## ✨ Key Features

### 1. Medical Report Upload

Upload medical reports in supported formats such as:

- PDF
- JPG
- PNG
- Scanned documents

---

### 2. OCR-Based Text Extraction

MedSafe uses Optical Character Recognition (OCR) to extract text from medical reports.

This allows the system to process:

- Blood test reports
- Laboratory reports
- Prescriptions
- Scanned medical documents

---

### 3. Structured Medical Data Extraction

Extracted information is converted into structured data.

Example:

```json
{
  "test_name": "HbA1c",
  "value": 8.2,
  "unit": "%",
  "reference_range": "4.0 - 5.6%"
}