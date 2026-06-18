import base64
import io
import json
import os
import re

import anthropic
import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

# ── CSI division names ─────────────────────────────────────────────────────
DIVISION_NAMES = {
    "00": "Procurement & Contracting",
    "01": "General Requirements",
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics & Composites",
    "07": "Thermal & Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC",
    "25": "Integrated Automation",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety & Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
}

# ── In-memory session store ────────────────────────────────────────────────
doc_store: dict = {
    "spec": {"text": None, "filename": None},
    "drawings": {"text": None, "filename": None, "page_count": 0},
}

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PDF text extraction (spec) ─────────────────────────────────────────────
def extract_pdf_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(p for p in pages if p.strip())


# ── Drawing PDF extraction (text + page images) ────────────────────────────
MAX_DRAWING_IMAGE_PAGES = 15


def extract_drawing_pages(data: bytes) -> tuple[str, list[dict], int]:
    """Returns (page-annotated full text, list of {page_num, b64_png}, total_pages)."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total = len(pdf.pages)
        parts = []
        for i, p in enumerate(pdf.pages):
            txt = (p.extract_text() or "").strip()
            parts.append(f"[PAGE {i + 1}]\n{txt}" if txt else f"[PAGE {i + 1} — no extractable text]")
    full_text = "\n\n".join(parts)

    doc = fitz.open(stream=data, filetype="pdf")
    page_images = []
    for i in range(min(MAX_DRAWING_IMAGE_PAGES, len(doc))):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(0.75, 0.75))
        page_images.append({
            "page_num": i + 1,
            "b64_png": base64.b64encode(pix.tobytes("png")).decode(),
        })
    doc.close()

    return full_text, page_images, total


# ── Section splitting (spec) ───────────────────────────────────────────────
SECTION_RE = re.compile(
    r"(?m)^[ \t]*(?:SECTION\s+)?(\d{2}[ \t]+\d{2}[ \t]+\d{2})[ \t]*[-–—]?[ \t]*([^\n]*)",
    re.IGNORECASE,
)


def split_sections(text: str) -> list:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [{"number": "FULL DOC", "title": "Full Document", "text": text}]
    sections = []
    for i, m in enumerate(matches):
        num = re.sub(r"[ \t]+", " ", m.group(1).strip())
        title = m.group(2).strip().strip("-–— ")
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"number": num, "title": title, "text": text[start:end].strip()})
    return sections


# ── Clause parsing ─────────────────────────────────────────────────────────
CLAUSE_RE = re.compile(
    r"(?m)^[ \t]*((?:\d+\.)+(?:\d+|[A-Z]))\.?[ \t]+(.{1,600})"
)


def parse_clauses(section_text: str) -> list:
    clauses = []
    for m in CLAUSE_RE.finditer(section_text):
        cid = m.group(1).rstrip(".")
        ctext = " ".join(m.group(2).split())[:500]
        clauses.append({"id": cid, "text": ctext})
    return clauses[:300]


# ── Spec analysis prompt ───────────────────────────────────────────────────
SPEC_ANALYSIS_PROMPT = """\
You are an expert construction specification reviewer with deep knowledge of \
CSI MasterFormat, ACI, ASTM, AISC, and related standards used in architectural \
and structural specifications.

Analyze the specification text below and return ONLY valid JSON — no markdown \
fences, no commentary outside the JSON object.

Return this exact shape:

{
  "issues": [
    {
      "id": "issue-1",
      "type": "error",
      "section_number": "03 30 00",
      "clause_id": "1.02.B",
      "description": "One-line issue summary",
      "excerpt": "Short quoted text from the clause",
      "detail": "Full explanation and recommended action"
    }
  ],
  "grade": {
    "overall_score": 74,
    "letter": "C+",
    "metrics": [
      {
        "name": "Cross-section consistency",
        "category": "Coordination & Cross-References",
        "max_pts": 30,
        "score": 18,
        "detail": "What was checked and how scores were assigned",
        "findings": [
          { "type": "error", "text": "Finding description", "pts_delta": -8 }
        ]
      }
    ]
  }
}

Issue types:
  "error"   — definitive conflicts, mismatches, or technical errors (must fix)
  "warning" — potential issues or ambiguities that need review
  "note"    — observations for human review; no score deduction

Grading rubric — use EXACTLY these 6 metrics in this order:
  1. Cross-section consistency          30 pts  (category: Coordination & Cross-References)
  2. Internal cross-references valid    15 pts  (category: Coordination & Cross-References)
  3. Standards & edition currency       20 pts  (category: Technical Accuracy)
  4. Spec-to-drawing coordination       20 pts  (category: Technical Accuracy)
  5. CSI 3-Part format compliance       10 pts  (category: Completeness)
  6. Submittal requirements defined      5 pts  (category: Completeness)

Letter grades: A=90-100, B=80-89, C=70-79, D=60-69, F<60. Use + and - for borderline scores.

SPECIFICATION TEXT:
"""

# ── Drawing analysis prompt ────────────────────────────────────────────────
DRAWING_ANALYSIS_PROMPT = """\
You are an expert construction document reviewer with deep knowledge of \
architectural and engineering drawing standards, CSI MasterFormat, AIA document \
conventions, and building codes (IBC, ADA, NFPA).

You are analyzing a PDF drawing set. You have been provided with:
  1. Extracted text from all pages (title block data, general notes, schedules)
  2. Visual images of the first sheets for inspection

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.

Return this exact shape:

{
  "sheets": [
    {
      "id": "sheet-1",
      "number": "A-001",
      "title": "Cover Sheet",
      "discipline": "Architectural",
      "revision": "0",
      "issue_counts": { "errors": 0, "warnings": 0, "notes": 0 }
    }
  ],
  "issues": [
    {
      "id": "draw-issue-1",
      "type": "error",
      "sheet_number": "A-001",
      "description": "One-line issue summary",
      "detail": "Full explanation and recommended action"
    }
  ],
  "grade": {
    "overall_score": 82,
    "letter": "B",
    "metrics": [
      {
        "name": "Sheet index completeness",
        "category": "Completeness & Organization",
        "max_pts": 25,
        "score": 20,
        "detail": "Explanation of scoring",
        "findings": [
          { "type": "error", "text": "Finding description", "pts_delta": -5 }
        ]
      }
    ]
  }
}

Issue types:
  "error"   — definitive conflicts, missing required items, or technical errors
  "warning" — potential issues or ambiguities requiring review
  "note"    — observations for human review; no score deduction

For "sheets": extract every sheet number and title visible in the drawing index or title \
blocks, listed in document order. If no formal index exists, list sheets identified from \
visible content. Always initialize issue_counts to { "errors": 0, "warnings": 0, "notes": 0 }.
Discipline values: Architectural, Structural, Civil, Mechanical, Electrical, Plumbing, \
Landscape, Other.

Grading rubric — use EXACTLY these 5 metrics in this order:
  1. Sheet index completeness     25 pts  (category: Completeness & Organization)
  2. Title block compliance       20 pts  (category: Completeness & Organization)
  3. Spec cross-reference         30 pts  (category: Coordination & Cross-References)
  4. Code compliance notations    15 pts  (category: Technical Accuracy)
  5. Drawing standards            10 pts  (category: Technical Accuracy)

Metric guidance:
  Sheet index completeness: Does a drawing index exist? Do all listed sheets appear in the \
  document? Are sheets present but missing from the index?
  Title block compliance: Do all sheets carry: project name, sheet number, title, revision, \
  date, and firm name? Flag blank or inconsistent fields.
  Spec cross-reference: Do drawing notes cite specific spec sections (e.g., \
  "See Spec Section 03 30 00")? Are references plausible for the shown content? Are notes \
  that should cite specs but don't flagged?
  Code compliance notations: Are required notations present — IBC occupancy classification, \
  fire ratings on rated assemblies, egress path calculations, ADA compliance notes on \
  accessible routes?
  Drawing standards: Scale indicated on each sheet? North arrow on plan views? Dimensions \
  present? Symbols defined in a legend? Consistent line weights?

Letter grades: A=90-100, B=80-89, C=70-79, D=60-69, F<60. Use + and - for borderline scores.

EXTRACTED TEXT FROM DRAWING SET:
"""


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ── Spec upload endpoint ───────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()

    try:
        text = extract_pdf_text(data)
    except Exception as exc:
        raise HTTPException(400, f"PDF parse error: {exc}")

    if not text.strip():
        raise HTTPException(
            400,
            "No extractable text found. Make sure this is a text-based PDF "
            "(not a scanned image). Try a DOCX export if available.",
        )

    doc_store["spec"]["text"] = text
    doc_store["spec"]["filename"] = file.filename

    sections = split_sections(text)

    analysis_input = SPEC_ANALYSIS_PROMPT + text[:175_000]
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            messages=[{"role": "user", "content": analysis_input}],
        )
        raw_json = strip_fences(msg.content[0].text)
        result = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Claude returned malformed JSON: {exc}")
    except anthropic.APIError as exc:
        raise HTTPException(502, f"Claude API error: {exc}")

    issues: list = result.get("issues", [])

    counts: dict[str, dict] = {}
    for iss in issues:
        sn = iss.get("section_number", "")
        if sn not in counts:
            counts[sn] = {"errors": 0, "warnings": 0, "notes": 0}
        t = iss.get("type", "note")
        if t == "error":
            counts[sn]["errors"] += 1
        elif t == "warning":
            counts[sn]["warnings"] += 1
        else:
            counts[sn]["notes"] += 1

    section_list = []
    for s in sections:
        num = s["number"]
        div = num.split()[0] if " " in num else num[:2]
        section_list.append({
            "number": num,
            "title": s["title"],
            "division": div,
            "division_name": DIVISION_NAMES.get(div, f"Division {div}"),
            "text": s["text"],
            "clauses": parse_clauses(s["text"]),
            "issue_counts": counts.get(num, {"errors": 0, "warnings": 0, "notes": 0}),
        })

    return {
        "filename": file.filename,
        "sections": section_list,
        "issues": issues,
        "grade": result.get("grade", {}),
    }


# ── Drawing upload endpoint ────────────────────────────────────────────────
@app.post("/upload-drawings")
async def upload_drawings(file: UploadFile = File(...)):
    data = await file.read()

    try:
        full_text, page_images, total_pages = extract_drawing_pages(data)
    except Exception as exc:
        raise HTTPException(400, f"PDF parse error: {exc}")

    # Build multimodal content: text prompt + page image blocks
    content: list = [
        {"type": "text", "text": DRAWING_ANALYSIS_PROMPT + full_text[:120_000]},
    ]
    for pg in page_images:
        content.append({"type": "text", "text": f"\n[Visual — page {pg['page_num']}:]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": pg["b64_png"],
            },
        })
    content.append({
        "type": "text",
        "text": (
            f"\n\nTotal pages in document: {total_pages}. "
            f"Visual samples shown: {len(page_images)} pages."
        ),
    })

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            messages=[{"role": "user", "content": content}],
        )
        raw_json = strip_fences(msg.content[0].text)
        result = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Claude returned malformed JSON: {exc}")
    except anthropic.APIError as exc:
        raise HTTPException(502, f"Claude API error: {exc}")

    sheets: list = result.get("sheets", [])
    issues: list = result.get("issues", [])

    # Tally issue counts per sheet
    sheet_counts: dict[str, dict] = {}
    for iss in issues:
        sn = iss.get("sheet_number", "")
        if sn not in sheet_counts:
            sheet_counts[sn] = {"errors": 0, "warnings": 0, "notes": 0}
        t = iss.get("type", "note")
        if t == "error":
            sheet_counts[sn]["errors"] += 1
        elif t == "warning":
            sheet_counts[sn]["warnings"] += 1
        else:
            sheet_counts[sn]["notes"] += 1

    for sheet in sheets:
        sn = sheet.get("number", "")
        sheet["issue_counts"] = sheet_counts.get(sn, {"errors": 0, "warnings": 0, "notes": 0})

    doc_store["drawings"]["text"] = full_text
    doc_store["drawings"]["filename"] = file.filename
    doc_store["drawings"]["page_count"] = total_pages

    return {
        "filename": file.filename,
        "total_pages": total_pages,
        "sheets": sheets,
        "issues": issues,
        "grade": result.get("grade", {}),
    }


# ── Chat endpoint ──────────────────────────────────────────────────────────
class ChatBody(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(body: ChatBody):
    spec = doc_store["spec"]
    drawings = doc_store["drawings"]

    if not spec["text"] and not drawings["text"]:
        raise HTTPException(400, "No documents loaded. Upload a spec or drawing set first.")

    system = (
        "You are an expert construction document reviewer. "
        "Answer questions concisely and technically. "
        "Cite spec clause IDs (e.g. §1.02.B) and section numbers, or drawing sheet numbers, when relevant."
    )
    if spec["text"]:
        system += f"\n\nSPECIFICATION DOCUMENT:\n{spec['text'][:100_000]}"
    if drawings["text"]:
        system += f"\n\nDRAWING SET (extracted text):\n{drawings['text'][:75_000]}"

    messages = body.history[-20:] + [{"role": "user", "content": body.message}]

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return {"response": resp.content[0].text}
    except anthropic.APIError as exc:
        raise HTTPException(502, str(exc))


# ── Serve the HTML app ─────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("qaqc.html")
