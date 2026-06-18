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
You are a QA/QC reviewer for R3A Architecture. Analyze the provided drawing set \
against the firm's official per-sheet deliverable checklist below. You have:
  1. Extracted text from all pages (title blocks, general notes, schedules, tags)
  2. Visual images of the first sheets for direct inspection

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.

Return this exact shape:
{
  "sheets": [
    {
      "id": "sheet-1",
      "number": "A-001",
      "title": "Abbreviations, Symbols and Standard Mounting Heights",
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
      "checklist_ref": "G.2-001",
      "description": "One-line issue summary",
      "detail": "Full explanation with recommended corrective action"
    }
  ],
  "grade": {
    "overall_score": 82,
    "letter": "B",
    "metrics": [
      {
        "name": "Title Block & Sheet Organization",
        "category": "Completeness & Organization",
        "max_pts": 20,
        "score": 16,
        "detail": "Explanation of scoring",
        "findings": [{ "type": "error", "text": "Finding", "pts_delta": -4 }]
      }
    ]
  }
}

Issue types:
  "error"   — required item missing, definitive standard violation, or technical conflict
  "warning" — potential issue or item that needs human confirmation
  "note"    — observation or reminder; no score impact

checklist_ref: reference the checklist section and item number (e.g. "G.1-G.5", "G.2-001", "1.2-009").

For "sheets": list every sheet found in the drawing index or title blocks in document order.
Discipline values: Architectural, Structural, Civil, Mechanical, Electrical, Plumbing, Landscape, General, Other.

═══════════════════════════════════════════
R3A FIRM STANDARDS — PER-SHEET QA CHECKLIST
═══════════════════════════════════════════

GENERAL ITEMS (apply to every sheet in the set):
G.1  Confirm design documents comply with scope of work narrative and project goals.
G.2  Building Code Review completed — egress, accessibility, construction type, variances.
G.3  Zoning / Municipal Code Review completed.
G.4  Plumbing count review completed.
G.5  All plans drawn at consistent scale within type (Life Safety same scale; Floor Plans same scale, etc.).
G.6  All plans in the same orientation.
G.7  Consistent terminology between plans and specifications.
G.8  No vague notes such as "see architectural" or "see structural" as the only descriptor.
G.9  Avoid match-lines where possible.
G.10 Avoid the word "new" — use specific scope language instead.
G.11 Views organized top-left to right-and-down; view numbers match order.
G.12 View titles in ALL CAPS.
G.13 Grid used as non-plottable guide to organize sheets.
G.14 North arrow to the left of view number on floor plan views.
G.15 Client Name and Address correct.
G.16 Project Name and Address correct.
G.17 Project Directory updated with Owner, Architect, and Consultant contacts.
G.18 All text (general notes, reference notes, detail notes) in Arial 3/32" all-caps, left-justified.
G.19 Note leaders centered on top row (leader on left) or bottom row (leader on right).
G.20 Room tags centered in room; numbering clockwise from entry, sequential by floor (100s, 200s…).
G.21 Dimensions locate all walls and doors; use 1/4" or larger, single strings.
G.22 No overlapping information — tags and notes must not obscure each other.
G.23 All typical and general notes reviewed for applicability, spelling, and grammar.

TITLE BLOCK (apply to every sheet):
TB-1  Client Name and Address correct and consistent on cover sheet and title block.
TB-2  Project Name and Address correct on cover sheet and title block.
TB-3  Project number correct.
TB-4  Document/Deliverable Phase accurate to submission.
TB-5  Issuance date correct.
TB-6  Logo and firm information graphically correct.
TB-7  Sheet name text is ALL CAPS.
TB-8  Sheet number correct per sheet series.
TB-9  Permit stamp applied when required; correct seal for the state.

G-001 — COVER SHEET:
001  Deliverable Issuance Phase and Date updated.
002  Project Location Map shown — building identified with hatch and "PROJECT LOCATION" note.
003  Site Plan included.
004  Project Directory shows current Owner, Architect, and Consultant contacts.
005  Project Rendering included — indicative "money shot" of the project.
006  Sheet List updated including consultant drawings for submission.

G-010 — CODE SUMMARY:
001  Code Summary PDF linked at 300 ppi.
002  Code Summary information coordinated and current with design.
003  Occupant load indicated and matches Life Safety sheets.
004  Fire ratings for construction type indicated and coordinated with Life Safety and floor plans.
005  Work area on Code Summary coordinated with Life Safety plan.
006  IECC requirements included — opaque doors, windows, thermal envelope (walls, roof).
007  Plumbing counts shown (existing vs. proposed) and coordinated with plans.

G-100s — LIFE SAFETY PLANS:
001  Correct view template used — plan halftoned.
002  Legend and notes reviewed for applicability; AHJ referenced correctly.
003  Fire-rated assemblies identified with correct line type (1-hour, 2-hour, etc.).
004  Occupant loads identified per area or occupancy room schedule confirmed.
005  Limit-of-Work line type and note used for renovation work area; coordinated with Code Summary.
006  Exit width tags on exit doors to grade and stair doors; parameters edited.
007  Exit signs shown at required exits; directional arrows shown; coordinated with RCP and Electrical.
008  Fire extinguishers shown; location meets code distance/area; FEC signage and light noted if recessed.
009  Exit travel distance path shown with EP linetype; note reads "EXIT TRAVEL DISTANCE = 00'-00"".
010  Accessible route noted from work area to elevators/exits.
011  Clear floor areas and turning radii shown.
012  Stairs identified with rating and diagonal hatch per legend.
013  Occupant load noted at exit doors with text note and oval border.

G-200s — UL ASSEMBLIES:
001  UL assembly PDFs linked into drafting view at correct resolution; organized left-to-right, top-to-bottom.
002  UL assemblies coordinated with Wall Types legend.

G-300s — SIGNAGE PLANS:
001  Correct view template used — plan halftoned similar to Life Safety.
002  Elevation legend of signage types provided.
003  Signage tagged at doors and other locations per scope.
004  Signage schedule complete with legend, sizes, and materials.
005  General notes include electrical lighting requirements for illuminated signage.

A-001 — ABBREVIATIONS, SYMBOLS, MOUNTING HEIGHTS:
001  Sheet largely unchanged/unedited from standard template.
002  Standard Mounting Heights reviewed and dimensions confirmed against current codes.

A-002 — GENERAL NOTES AND WALL TYPES:
003  General Construction and Demolition Notes edited for project scope.
004  Wall type details added for initial wall selections.
005  General Wall Type Notes edited for project scope.
006  General Referenced Construction and Demolition Notes edited for project scope.

A-010s — ARCHITECTURAL SITE PLANS:
001  Building footprint shown (not floor plan) — thick perimeter line with grey hatch.
002  Site context and adjacent buildings shown.
003  Street labels shown.
004  Survey or CAD background linked (black, non-critical text removed).
005  New hardscape/sidewalks shown with hatching distinguishing landscape from hardscape.
006  Design elements shown to quantify/qualify scope.
007  Dimension strings for critical clearances; continuous strings off existing buildings and column grids.
008  Exterior Elevation tags shown.
009  Section/Detail tags for exterior site elements.
010  Callouts for enlarged site plans and floor plans.
011  Correct view template used.
012  Notes identify exterior stairs, ramps, hardscape, landscape, and scope.

A-100s — FLOOR PLANS:
001  Walls showing correct thickness per actual wall types (interior/exterior).
002  Windows correct width, height, sill height.
003  Doors and storefronts correct dimensions and location.
004  Code compliance confirmed — airlocks, egress door clearances.
005  Toilet rooms — correct fixture counts, layouts, plumbing clearances.
006  Room tags with name and number (101, 201… clockwise from entry).
007  Casework/millwork shown with countertops, base/wall cabinets, sinks, appliances.
008  Door and storefront tags placed correctly (leaf center, overhead to side, storefront centered).
009  Dimensions — column grid, exterior face of walls, interior strings locating all walls and doors.
010  Column grids shown and coordinated.
011  Correct view template used.
012  Scope of work notes provided (reference notes or text notes).

A-110s — REFLECTED CEILING PLANS:
001  Dimensions — column grid and interior strings.
002  Column grids shown and coordinated.
003  Ceiling layouts shown with coordinated types and elevations.
004  Light fixtures, supply/return/exhaust diffusers, and sprinkler heads shown.
005  Correct view template used.
006  Ceiling access panels shown.
007  Scope of work notes provided.

A-120s — ROOF PLANS:
001  Dimensions — column grid, exterior face of walls, interior strings.
002  Column grids shown and coordinated.
003  Mechanical units and penetrations shown with size and location.
004  Roof drainage shown — gutters, downspouts, drains, scuppers, slopes indicated.
005  Roof assembly modeled; slopes and direction identified.
006  Roof access and walkway paths to units shown.
007  Typical Wall Section tags shown.
008  Roof edge conditions identified and detailed — coping, gravel stop, etc.
009  Correct view template used.
010  Scope of work notes provided.

A-200s — EXTERIOR ELEVATIONS:
001  Materials proposed identified graphically with Basis of Design notes.
002  Correct view template used.
003  Exterior glazing identified with glazing "ticks."
004  Typical Wall Section and Building Section tags visible and aligned.
005  Ground plane shown with masking region (extra heavy line).
006  Foundations shown with dashed lines.
007  Vertical dimensions provided — heights, floor references, horizontal/vertical material changes.

A-300s — BUILDING SECTIONS:
001  Wall section and roof assembly drafted/modeled accurately with glazing.
002  Ground plane and hatches for earth/gravel shown.
003  Foundations and perimeter drains shown if needed.
004  Correct view template used.
005  2D information added — sections cannot be unedited straight-from-Revit.
006  Room tags provided for rooms visible in section.
007  Vertical dimensions — heights, floor references, material changes.

A-310s — WALL SECTIONS:
001  Detailed notes describing each wall assembly element; notes left-justified with leaders per standard.
002  Wall section and roof assembly drafted/modeled accurately with glazing.
003  Ground plane and hatches for earth/gravel shown.
004  Foundations and perimeter drains shown if needed.
005  Correct view template used.
006  2D information added; sections cannot be unedited straight-from-Revit.
007  Vertical dimensions — heights, floor references, material changes.
008  Callouts for exterior details at larger scale.

A-400s — ENLARGED FLOOR PLANS & INTERIOR ELEVATIONS:
001  Walls correct thickness per actual wall types.
002  Windows correct width, height, sill height.
003  Doors and storefronts correct dimensions and location.
004  Code compliance confirmed — egress doors, clearances.
005  Toilet rooms — correct counts, layouts, plumbing clearances.
006  Room tags with name and number.
007  Casework/millwork shown (overall dimensions only in plan; details in elevations).
008  Door and storefront tags placed correctly.
009  Dimensions — column grid, exterior/interior strings.
010  Interior elevation tags shown; all required elevations tagged and on sheet.
011  Correct view template used.
012  Scope of work notes provided.

A-500s — DETAILS:
001  Detailed notes for each element; left-justified with leaders per standard.
002  Correct view template used.
003  Detail components used to illustrate elements (metal studs, sheathing, etc.).
004  Critical dimensions provided where needed (alignments, offsets).
005  Hatches correctly reflect materials noted in details.
006  2D information added — details cannot be unedited straight-from-Revit.
007  Callout/tags to corresponding details at larger scale.

A-600s — DOOR SCHEDULES AND DETAILS:
001  Door Type drafting view edited to show door slabs in project.
002  Frame Type drafting view edited to show frames in project.
003  Hardware sets noted in schedule.
004  Head, jamb, and sill details coordinated with project conditions.
005  Storefront details and system elevations added and coordinated with Door Schedule.

A-610s — STOREFRONTS AND WINDOWS:
001  Storefront Type Elevations show dimensions, glazing/frame types, mullion layouts, Head/Jamb/Sill details.
002  Frame/mullion visually correct for specified system (2" typical, curtainwall 2.5", frame depth 4.5"–7"+).
003  Frame type and glazing types noted in elevations and general notes/legend.
004  Head, jamb, and sill details coordinated with project conditions.
005  Storefront details coordinated with Door Schedule.

A-700s — STAIRS, RAMPS, ELEVATORS:
001  Enlarged floor plans for stairs, elevators, ramps match typical enlarged plan standard.
002  Correct view template used.
003  Stair design and details — 7" max riser, 11" min tread, 1" inset/overlap at toe.
004  Railing design and details — max ramp slope 1:12; railings required at 1:20+.
005  Elevator details included per project type.
006  Stair, elevator, and ramp sections with 2D information, notes, and dimensions.

A-800s — EXTERIOR DETAILS:
001  Details organized vertically and horizontally where related.
002  Sheet organized with grid.
003  Notes left-justified — leader from bottom on right, from top on left.
004  Notes specific to condition, indicating contractor scope and material references (no product names in drawings).
005  Notes are sentences without periods at end; acceptable punctuation: - / ,
006  Vertical dimensions and column grids/levels shown in applicable views.
007  Correct view template and drawing scale used.

AF001 — FINISH SCHEDULE:
001  Wall and Flooring Schedules with finish selections provided.
002  Correct schedule view template used.
003  Finish General Notes and project-specific notes reviewed for applicability.
004  Schedule view template correct.

AF101 — FINISH PLANS:
001  Correct view template used (no building grid).
002  Finish tags on plans; separate plans for walls/surfaces vs. floors for clarity.
003  Finish tags correspond to schedules on AF001.
004  Dimensions not used on Finish Plans (unless separate dimensioned plans directed by PM).
005  Material alignment identified; tag transitions coordinated.

AQ001 — EQUIPMENT SCHEDULE:
001  Equipment Schedule with tag, description, and required utilities.
002  Supply/install responsibility noted — OFOI, OFCI, CFCI, or CFOI.
003  Equipment General Notes reviewed for applicability.
004  Correct view template used.

AQ101 — EQUIPMENT PLANS:
001  Correct view template used.
002  Equipment locations and layouts shown with specialty clearances noted.
003  Tags per piece of equipment with leaders for clarity.
004  Equipment General Notes reviewed for applicability.

═══════════════════════════════════════════
GRADING RUBRIC — use EXACTLY these 5 metrics:
  1. Title Block & Sheet Organization    20 pts  (category: Completeness & Organization)
  2. General Standards Compliance        25 pts  (category: General Standards)
  3. Sheet-Specific Content              30 pts  (category: Sheet Content)
  4. Code & Accessibility Notations      15 pts  (category: Code Compliance)
  5. Coordination & Cross-References     10 pts  (category: Coordination)

Letter grades: A=90-100, B=80-89, C=70-79, D=60-69, F<60. Use + and - for borderline scores.

INSTRUCTIONS:
- For each sheet found, identify its sheet series and apply the matching checklist above.
- Apply General Items (G.1–G.23) and Title Block checks (TB-1–TB-9) to every sheet.
- Report every violation as an issue with the checklist_ref from the checklist above.
- Focus on what is verifiable from extracted text and visual images — do not speculate about Revit model internals.
- Be specific: quote actual text from the drawing when flagging an issue.

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
