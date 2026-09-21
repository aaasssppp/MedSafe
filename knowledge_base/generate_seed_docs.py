"""
Seed Medical Knowledge Base PDF Generator.
Generates authoritative, standard clinical reference documents for offline indexing.
"""

import os


def generate_pdf(filepath: str, title: str, sections: list):
    """Generates standard PDF 1.4 document with Helvetica fonts."""
    objects = []

    def add_object(content):
        objects.append(content)
        return len(objects)

    objects.append(None)  # 1: Catalog
    objects.append(None)  # 2: Pages
    f1 = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_objs = []
    lines = []
    lines.append(("TITLE", title))
    lines.append(("BLANK", ""))

    for heading, text_block in sections:
        lines.append(("HEADING", heading))
        for line in text_block.strip().split("\n"):
            line = line.strip()
            if line:
                lines.append(("TEXT", line))
            else:
                lines.append(("BLANK", ""))
        lines.append(("BLANK", ""))

    # Paginate (approx 38-42 lines per page)
    pages_lines = []
    curr = []
    for item in lines:
        curr.append(item)
        if len(curr) >= 40:
            pages_lines.append(curr)
            curr = []
    if curr:
        pages_lines.append(curr)

    for p_items in pages_lines:
        stream_cmds = ["BT", "50 740 Td", "14 TL"]

        for kind, val in p_items:
            safe_val = val.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if kind == "TITLE":
                stream_cmds.append("/F2 16 Tf")
                stream_cmds.append(f"({safe_val}) Tj")
                stream_cmds.append("T*")
                stream_cmds.append("T*")
            elif kind == "HEADING":
                stream_cmds.append("/F2 12 Tf")
                stream_cmds.append(f"({safe_val}) Tj")
                stream_cmds.append("T*")
            elif kind == "TEXT":
                stream_cmds.append("/F1 10 Tf")
                stream_cmds.append(f"({safe_val}) Tj")
                stream_cmds.append("T*")
            elif kind == "BLANK":
                stream_cmds.append("T*")

        stream_cmds.append("ET")
        stream_data = "\n".join(stream_cmds).encode("latin-1", errors="replace")

        c_obj = add_object(
            f"<< /Length {len(stream_data)} >>\nstream\n"
            + stream_data.decode("latin-1")
            + "\nendstream"
        )
        p_obj = add_object(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> /Contents {c_obj} 0 R >>"
        )
        page_objs.append(p_obj)

    kids_str = " ".join(f"{p} 0 R" for p in page_objs)
    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_objs)} >>"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, obj in enumerate(objects):
            offsets.append(f.tell())
            f.write(f"{i + 1} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

        xref_offset = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode("latin-1"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode("latin-1"))

        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1"))
        f.write(b"startxref\n")
        f.write(f"{xref_offset}\n".encode("latin-1"))
        f.write(b"%%EOF\n")


def build_all_seed_docs():
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # 1. diabetes.pdf
    generate_pdf(
        os.path.join(kb_dir, "diabetes.pdf"),
        "Clinical Reference: Diabetes Mellitus & Glycemic Markers",
        [
            (
                "1. Overview of Glycemic Control and Evaluation",
                "Diabetes mellitus is a metabolic disorder characterized by persistent hyperglycemia.\n"
                "Assessment relies primarily on Glycated Hemoglobin (HbA1c) and blood glucose metrics.\n"
                "Proper interpretation requires evaluating both acute and long-term glycemic markers.",
            ),
            (
                "2. Glycated Hemoglobin (HbA1c) Reference Intervals",
                "Normal / Non-diabetic range: Below 5.7%\n"
                "Prediabetes (Increased risk for diabetes): 5.7% to 6.4%\n"
                "Diabetes diagnostic threshold: 6.5% or greater on two separate occasions.\n"
                "Target for most non-pregnant adults with diabetes: Generally below 7.0%.\n"
                "HbA1c reflects average circulating plasma glucose concentration over 8 to 12 weeks.",
            ),
            (
                "3. Plasma Glucose Testing Thresholds",
                "Fasting Blood Glucose (FPG):\n"
                "- Normal: 70 to 99 mg/dL (3.9 to 5.5 mmol/L)\n"
                "- Impaired Fasting Glucose (Prediabetes): 100 to 125 mg/dL (5.6 to 6.9 mmol/L)\n"
                "- Diabetic Fasting Threshold: 126 mg/dL or higher (>= 7.0 mmol/L)\n"
                "Random Blood Glucose: 200 mg/dL or higher accompanied by symptoms (polyuria, polydipsia).",
            ),
            (
                "4. Clinical Interpretation and Recommendations",
                "Elevated HbA1c or FPG readings warrant comprehensive clinical evaluation.\n"
                "Factors affecting HbA1c accuracy include hemoglobinopathies, severe anemia, and chronic renal disease.\n"
                "Patients should discuss lifestyle, diet, exercise, and potential pharmacotherapy with their physician.",
            ),
        ],
    )

    # 2. hypertension.pdf
    generate_pdf(
        os.path.join(kb_dir, "hypertension.pdf"),
        "Clinical Reference: Hypertension & Blood Pressure Guidelines",
        [
            (
                "1. Blood Pressure Classification (ACC/AHA Guidelines)",
                "Normal Blood Pressure: Systolic less than 120 mmHg AND Diastolic less than 80 mmHg.\n"
                "Elevated Blood Pressure: Systolic 120-129 mmHg AND Diastolic less than 80 mmHg.\n"
                "Stage 1 Hypertension: Systolic 130-139 mmHg OR Diastolic 80-89 mmHg.\n"
                "Stage 2 Hypertension: Systolic 140 mmHg or higher OR Diastolic 90 mmHg or higher.\n"
                "Hypertensive Crisis: Systolic exceeding 180 mmHg and/or Diastolic exceeding 120 mmHg.",
            ),
            (
                "2. Measurement Protocols and Variability",
                "Blood pressure exhibits circadian variation and responsiveness to physical or emotional stress.\n"
                "White-coat hypertension involves elevated clinic readings with normal ambulatory readings.\n"
                "Diagnosis requires multiple calibrated readings across separate visits or 24-hour ambulatory monitoring.",
            ),
            (
                "3. Cardiovascular Risk Factors & Target Organ Assessment",
                "Chronic hypertension contributes to atherosclerosis, left ventricular hypertrophy, and nephrosclerosis.\n"
                "Diagnostic workup should include assessment of renal function (serum creatinine, eGFR), electrolytes, and urinalysis.\n"
                "Primary lifestyle interventions include dietary sodium reduction (DASH diet), physical activity, and stress management.",
            ),
            (
                "4. Clinical Discussion Guidance",
                "Patients with readings above 130/80 mmHg should maintain a blood pressure log.\n"
                "Any abrupt systolic spike over 180 mmHg or chest pain requires emergency medical evaluation.\n"
                "Medication initiation or adjustment must strictly be managed by the treating clinician.",
            ),
        ],
    )

    # 3. blood_tests.pdf
    generate_pdf(
        os.path.join(kb_dir, "blood_tests.pdf"),
        "Clinical Reference: Complete Blood Count & Chemistry Panels",
        [
            (
                "1. Complete Blood Count (CBC) Reference Intervals",
                "Hemoglobin (Hgb):\n"
                "- Adult Males: 13.8 to 17.2 g/dL\n"
                "- Adult Females: 12.1 to 15.1 g/dL\n"
                "- Low values indicate anemia; elevated values may suggest polycythemia or dehydration.\n"
                "White Blood Cell Count (WBC): 4,500 to 11,000 cells/mcL.\n"
                "- Elevated (Leukocytosis): Infection, inflammatory state, or physiological stress.\n"
                "- Decreased (Leukopenia): Bone marrow suppression, viral illness, or autoimmune conditions.\n"
                "Platelet Count: 150,000 to 450,000 /mcL (Thrombocytopenia vs Thrombocytosis).",
            ),
            (
                "2. Comprehensive Metabolic Panel (CMP) & Renal Function",
                "Serum Creatinine: 0.7 to 1.3 mg/dL (adult males), 0.5 to 1.1 mg/dL (adult females).\n"
                "eGFR (estimated Glomerular Filtration Rate): >= 90 mL/min/1.73m2 (Normal).\n"
                "Blood Urea Nitrogen (BUN): 7 to 20 mg/dL.\n"
                "Electrolytes: Sodium (135-145 mEq/L), Potassium (3.5-5.0 mEq/L), Chloride (96-106 mEq/L).",
            ),
            (
                "3. Lipid Profile & Cardiovascular Risk Ratios",
                "Total Cholesterol: Desirable below 200 mg/dL; Borderline high 200-239 mg/dL; High >= 240 mg/dL.\n"
                "LDL Cholesterol ('bad' cholesterol): Optimal < 100 mg/dL; Borderline 130-159 mg/dL; High 160-189 mg/dL.\n"
                "HDL Cholesterol ('good' protective cholesterol): Desirable >= 40 mg/dL (males), >= 50 mg/dL (females).\n"
                "Triglycerides: Normal < 150 mg/dL; Borderline high 150-199 mg/dL; High >= 200 mg/dL.",
            ),
            (
                "4. Patient Guidance on Laboratory Deviations",
                "A value slightly outside reference intervals does not automatically indicate illness.\n"
                "Physicians evaluate trends over time alongside physical exams and clinical symptoms.\n"
                "Fasting status, hydration, and acute stressors can temporarily alter lab values.",
            ),
        ],
    )

    # 4. medications.pdf
    generate_pdf(
        os.path.join(kb_dir, "medications.pdf"),
        "Clinical Reference: Pharmacology Overview & Safe Medication Use",
        [
            (
                "1. Common Cardiometabolic Drug Classes",
                "Biguanides (e.g., Metformin): First-line insulin-sensitizing agent for type 2 diabetes.\n"
                "ACE Inhibitors (e.g., Lisinopril) & ARBs (e.g., Losartan): First-line antihypertensive therapy.\n"
                "HMG-CoA Reductase Inhibitors (Statins e.g., Atorvastatin, Rosuvastatin): Lipid-lowering agents.\n"
                "Beta-Blockers (e.g., Metoprolol): Regulate heart rate and myocardial oxygen demand.",
            ),
            (
                "2. Adherence, Safety, and Monitoring",
                "Medications must be taken strictly as prescribed by a licensed healthcare professional.\n"
                "Do not abruptly discontinue cardiovascular or metabolic medications without medical consultation.\n"
                "Regular monitoring of renal function, electrolytes, and hepatic enzymes is standard clinical practice.",
            ),
            (
                "3. Adverse Effects and Warning Symptoms",
                "Common side effects: Gastrointestinal upset with Metformin, dry cough with ACE inhibitors, myalgia with statins.\n"
                "Severe adverse reactions (e.g. angioedema, jaundice, severe hypoglycemia) require immediate medical care.\n"
                "Inform healthcare providers of all over-the-counter supplements and herbal products.",
            ),
            (
                "4. Patient-Doctor Communication Essentials",
                "Maintain an up-to-date medication list including dosages and schedules.\n"
                "Report any suspected adverse reactions or difficulty affording prescriptions promptly to the provider.\n"
                "MedSafe does not recommend changing, adjusting, or initiating any pharmaceutical regimen.",
            ),
        ],
    )

    print("Generated all 4 seed medical reference PDFs in:", kb_dir)


if __name__ == "__main__":
    build_all_seed_docs()
