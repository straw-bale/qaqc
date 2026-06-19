import asyncio
import base64
import io
import json
import os
import re
import tempfile
from typing import List

import anthropic
import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    "documents": {"text": None, "filenames": [], "page_count": 0},
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
# 2× scale = 144 DPI effective — legible for fine text and hairline work.
# JPEG quality 90 minimises compression artefacts on line drawings.
# All pages are rendered; batching in the endpoint handles context limits.
RENDER_SCALE = 2.0
JPEG_QUALITY = 90
PAGES_PER_BATCH = 15  # pages per Claude vision call at 2× scale


def extract_drawing_pages(pdf_path: str) -> tuple[str, list[dict], int]:
    """Returns (page-annotated full text, list of {page_num, b64, media_type}, total_pages).
    Accepts a file path so neither pdfplumber nor fitz need to copy bytes into RAM."""
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        parts = []
        for i, p in enumerate(pdf.pages):
            txt = (p.extract_text() or "").strip()
            parts.append(f"[PAGE {i + 1}]\n{txt}" if txt else f"[PAGE {i + 1} — no extractable text]")
    full_text = "\n\n".join(parts)

    # fitz opens from disk — no in-memory PDF copy, minimal RAM overhead.
    doc = fitz.open(pdf_path)
    page_images = []
    try:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
            page_images.append({
                "page_num": i + 1,
                "b64": base64.b64encode(pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)).decode(),
                "media_type": "image/jpeg",
            })
            del pix  # free each pixmap immediately after encoding
    finally:
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
CSI MasterFormat and the standards used in Pennsylvania architectural and structural \
specifications. Projects in Pennsylvania fall under the Pennsylvania Uniform \
Construction Code (UCC), which adopts the following current editions:
  - IBC 2018 (International Building Code, effective PA 2022)
  - IFC 2018 (International Fire Code)
  - IPC 2018 (International Plumbing Code)
  - IMC 2018 (International Mechanical Code)
  - IECC 2018 / ASHRAE 90.1-2016 (energy)
  - NEC 2017 / NFPA 70 (electrical)
  - ICC/ANSI A117.1-2017 (accessibility, referenced by IBC)
  - ADA 2010 Standards for Accessible Design
  - ASCE 7-16 (structural loads, referenced by IBC 2018)
  - ACI 318-14 (concrete, referenced by IBC 2018)
  - AISC 360-16 (steel, referenced by IBC 2018)
  - AWC NDS 2018 (wood, referenced by IBC 2018)
Flag any specification references to superseded editions of these standards as errors.

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

# ── Document batch analysis prompt ─────────────────────────────────────────
# Used for each batch of pages from any uploaded construction document.
# Files may be drawing sets, specification books, or combined documents.
DRAWING_BATCH_PROMPT = """\
You are a QA/QC reviewer for R3A Architecture. Analyze the provided batch of construction \
document pages against the firm's official per-sheet deliverable checklist below. You have:
  1. Extracted text from ALL pages across all uploaded files (title blocks, notes, schedules, tags, spec sections)
  2. High-resolution visual images of THIS BATCH of pages for direct inspection

Pages may contain drawing sheets, specification sections, general notes, or a mix. \
Treat specification section pages as sheets with discipline "Specification" and use the \
CSI section number (e.g. "03 30 00") as the sheet number and the section title as the title.

This project is in Pennsylvania. Apply the Pennsylvania Uniform Construction Code (UCC), \
which adopts these current editions:
  - IBC 2018 (effective PA 2022), IFC 2018, IPC 2018, IMC 2018
  - IECC 2018 / ASHRAE 90.1-2016, NEC 2017 / NFPA 70
  - ICC/ANSI A117.1-2017, ADA 2010 Standards
  - ASCE 7-16, ACI 318-14, AISC 360-16, AWC NDS 2018

IMPORTANT — pages marked "[PAGE X — no extractable text]" are image-based (rasterized) \
sheets. Use the provided visual images to inspect those pages. Do NOT list them as blank \
or skip them — analyze their visible content from the images.

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
Do NOT include a "grade" field — grading is performed after all batches are merged.

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
  ]
}

Issue types:
  "error"   — required item missing, definitive standard violation, or technical conflict
  "warning" — potential issue or item that needs human confirmation
  "note"    — observation or reminder; no score impact

checklist_ref: reference the checklist section and item number (e.g. "G.1-G.5", "G.2-001", "1.2-009").

For "sheets": list ONLY the sheets/pages whose visual images appear in this batch.
Discipline values: Architectural, Structural, Civil, Mechanical, Electrical, Plumbing, Landscape, General, Specification, Other.

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

INSTRUCTIONS:
- List in "sheets" ONLY the sheets visually present in this batch's page images.
- Report in "issues" ALL issues identifiable in this batch — from both visual inspection and cross-referencing the extracted text.
- Cross-sheet issues detectable from the extracted text (inconsistent terminology, mismatched tags, numbering errors) should be reported even if the affected sheets are not in this batch.
- For pages marked "[PAGE X — no extractable text]", use the visual image — never mark as blank or skip.
- Be specific: quote actual text from the drawing when flagging an issue.
- Flag any code references not matching the current Pennsylvania UCC editions listed above.
- Focus on what is verifiable from extracted text and visual images — do not speculate about Revit model internals.

EXTRACTED TEXT FROM ENTIRE DRAWING SET (use for cross-reference context):
"""

# ── Drawing grade prompt ───────────────────────────────────────────────────
# Separate text-only call after all batches are merged.
DRAWING_GRADE_PROMPT = """\
You are completing a QA/QC review of an architectural drawing set for R3A Architecture. \
Based on the full sheet inventory and all issues identified by visual inspection of every page, \
assign overall grades using the rubric below.

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.

Return this exact shape:
{
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

GRADING RUBRIC — use EXACTLY these 5 metrics in this order:
  1. Title Block & Sheet Organization    20 pts  (category: Completeness & Organization)
  2. General Standards Compliance        25 pts  (category: General Standards)
  3. Sheet-Specific Content              30 pts  (category: Sheet Content)
  4. Code & Accessibility Notations      15 pts  (category: Code Compliance)
  5. Coordination & Cross-References     10 pts  (category: Coordination)

Letter grades: A=90-100, B=80-89, C=70-79, D=60-69, F<60. Use + and - for borderline scores.

Grade ONLY the disciplines present. Do not penalise for absent consultant disciplines — \
an architectural-only set should be graded as a complete architectural submission.

"""


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ── Unified document upload endpoint ──────────────────────────────────────
# Accepts one or more PDFs (drawings, specs, or mixed). All files are rendered
# page-by-page and analyzed together so specs embedded in a drawing set are
# caught without requiring the user to classify files first.
@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    # Stream each file to a temp path, extract pages, then clean up.
    all_text_parts: list[str] = []
    all_page_images: list[dict] = []
    filenames: list[str] = []
    global_page_offset = 0

    for file in files:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp:
                while chunk := await file.read(1024 * 1024):
                    tmp.write(chunk)
            try:
                file_text, page_images, _ = await asyncio.to_thread(
                    extract_drawing_pages, tmp_path
                )
            except Exception as exc:
                raise HTTPException(400, f"PDF parse error ({file.filename}): {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Offset page numbers so they remain unique across all uploaded files.
        for pg in page_images:
            pg["page_num"] += global_page_offset
        global_page_offset += len(page_images)

        all_text_parts.append(f"=== FILE: {file.filename} ===\n{file_text}")
        all_page_images.extend(page_images)
        filenames.append(file.filename)

    full_text = "\n\n".join(all_text_parts)
    total_pages = len(all_page_images)

    # Make extracted text available for chat immediately, before streaming starts.
    doc_store["documents"]["text"] = full_text
    doc_store["documents"]["filenames"] = filenames
    doc_store["documents"]["page_count"] = total_pages

    batches = [
        all_page_images[i : i + PAGES_PER_BATCH]
        for i in range(0, len(all_page_images), PAGES_PER_BATCH)
    ]
    total_batches = len(batches)

    async def generate():
        all_sheets: list = []
        all_issues: list = []
        issue_counter = 0
        prior_context = ""

        for batch_idx, batch in enumerate(batches):
            content: list = [
                {"type": "text", "text": DRAWING_BATCH_PROMPT + full_text[:120_000]},
            ]
            for pg in batch:
                content.append({"type": "text", "text": f"\n[Visual — page {pg['page_num']}:]"})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": pg["media_type"],
                        "data": pg["b64"],
                    },
                })

            tail = (
                f"\n\nTotal pages across all uploaded files: {total_pages}. "
                f"This batch covers pages {batch[0]['page_num']}–{batch[-1]['page_num']} "
                f"(batch {batch_idx + 1} of {total_batches}). "
                f"Files uploaded: {', '.join(filenames)}."
            )
            if prior_context:
                tail += prior_context
            content.append({"type": "text", "text": tail})

            try:
                msg = await asyncio.to_thread(
                    client.messages.create,
                    model="claude-sonnet-4-6",
                    max_tokens=24000,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": content}],
                )
                raw_text = next(b.text for b in msg.content if b.type == "text")
                batch_result = json.loads(strip_fences(raw_text))
            except Exception as exc:
                yield json.dumps({"event": "error", "message": str(exc)}) + "\n"
                return

            batch_sheets = batch_result.get("sheets", [])
            batch_issues = batch_result.get("issues", [])

            for iss in batch_issues:
                issue_counter += 1
                iss["id"] = f"issue-{issue_counter}"

            batch_counts: dict[str, dict] = {}
            for iss in batch_issues:
                sn = iss.get("sheet_number", "")
                if sn not in batch_counts:
                    batch_counts[sn] = {"errors": 0, "warnings": 0, "notes": 0}
                t = iss.get("type", "note")
                if t == "error":
                    batch_counts[sn]["errors"] += 1
                elif t == "warning":
                    batch_counts[sn]["warnings"] += 1
                else:
                    batch_counts[sn]["notes"] += 1
            for s in batch_sheets:
                sn = s.get("number", "")
                s["issue_counts"] = batch_counts.get(sn, {"errors": 0, "warnings": 0, "notes": 0})

            all_sheets.extend(batch_sheets)
            all_issues.extend(batch_issues)

            yield json.dumps({
                "event": "batch",
                "batch_num": batch_idx + 1,
                "total_batches": total_batches,
                "page_start": batch[0]["page_num"],
                "page_end": batch[-1]["page_num"],
                "total_pages": total_pages,
                "filenames": filenames,
                "sheets": batch_sheets,
                "issues": batch_issues,
            }) + "\n"

            if all_sheets or all_issues:
                sheet_summary = ", ".join(
                    f"{s.get('number', '?')} — {s.get('title', '?')}"
                    for s in all_sheets[:40]
                )
                issue_lines = "\n".join(
                    f"  [{i.get('checklist_ref', '')}] {i.get('sheet_number', '')} — {i.get('description', '')}"
                    for i in all_issues[-80:]
                )
                prior_context = (
                    "\n\nCONTEXT FROM PRIOR BATCHES — use for cross-document awareness. "
                    "Do not re-report these exact issues verbatim; only flag them again "
                    "if the same problem recurs on a new page with distinct evidence:\n"
                    f"Sheets/sections already logged: {sheet_summary}\n"
                    f"Issues already identified (most recent {min(80, len(all_issues))}):\n"
                    f"{issue_lines}"
                )

        # Deduplicate sheets (first occurrence wins).
        seen: dict = {}
        for s in all_sheets:
            sn = s.get("number", "")
            if sn and sn not in seen:
                seen[sn] = s
        sheets: list = list(seen.values())

        # Final grading call — text-only, no images.
        grade: dict = {}
        try:
            grade_input = (
                DRAWING_GRADE_PROMPT
                + "SHEETS/SECTIONS FOUND:\n"
                + json.dumps(sheets, indent=2)[:20_000]
                + "\n\nALL ISSUES FOUND:\n"
                + json.dumps(all_issues, indent=2)[:60_000]
            )
            grade_msg = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": grade_input}],
            )
            grade = json.loads(strip_fences(grade_msg.content[0].text)).get("grade", {})
        except Exception:
            pass

        yield json.dumps({
            "event": "done",
            "filenames": filenames,
            "total_pages": total_pages,
            "grade": grade,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Chat endpoint ──────────────────────────────────────────────────────────
class ChatBody(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(body: ChatBody):
    docs = doc_store["documents"]

    if not docs["text"]:
        raise HTTPException(400, "No documents loaded. Upload files first.")

    system = (
        "You are an expert construction document reviewer. "
        "Answer questions concisely and technically. "
        "Cite spec clause IDs (e.g. §1.02.B) and section numbers, or drawing sheet numbers, when relevant."
        f"\n\nCONSTRUCTION DOCUMENTS (extracted text):\n{docs['text'][:175_000]}"
    )

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
