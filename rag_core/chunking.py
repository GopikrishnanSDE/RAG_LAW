"""
chunking.py — section-aware chunking for the Income Tax Act corpus.

Why not just RecursiveCharacterTextSplitter(chunk_size=500):
A fixed-size token window has no idea where a section, sub-section or clause
ends. It will happily cut "...the deduction under this section shall not
exceed" away from the number that follows it. The resulting chunk reads like
fluent English and embeds fine — it just answers the wrong question later,
which is exactly the silent-failure mode this project exists to catch.

Strategy: split on the document's own structure first (section > sub-section
> clause), and fall back to size-based splitting only *inside* a boundary
that's still too big for the embedding window — never *across* one.

Pipeline
--------
raw pages (from ingestion_pipeline.load_documents)
  -> concatenate into one string, keeping a page-offset index (for citations)
  -> track structural context as we scan: CHAPTER, Part ("B.—Deductions in
     respect of certain payments"), SCHEDULE
  -> detect section boundaries: a short title line immediately followed by
     "<number>.  " (two+ spaces) — e.g. "123.  An individual shall be
     allowed..."
  -> guard against false positives with a monotonic section-number check,
     because Schedules re-use the same "<number>.  " paragraph style with
     their own 1, 2, 3... numbering
  -> each section is a candidate chunk. If it's still under max_tokens, done.
  -> otherwise split along sub-section boundaries "(1)", "(2)", ... — and if
     a single sub-section is *still* too big, along clause boundaries
     "(a)", "(b)", ... — and only as an absolute last resort, a
     RecursiveCharacterTextSplitter with legal-aware separators.
  -> every emitted chunk gets a one-line header re-stating its section
     number/chapter/title, so chunk 2 of 3 is still self-describing to the embedder
     and to a human reading a retrieval trace.
  -> the Schedules after the last section ("SCHEDULE XV / [See section
     123]") are chunked as their own units, split on their "1.", "2."
     paragraphs. Without this, all 16 Schedules were absorbed into the last
     section (536, "Repeal and savings") as ~470 mislabelled chunks — so
     Schedule XV's list of what qualifies under Section 123 (tuition fees,
     PF, ...) was unreachable as "Section 123" content. Each Schedule
     chunk's header names the section it belongs to, so both retrieval
     signals connect it back.

Known limitations (documented, not hidden):
  - The monotonic-number heuristic assumes the Act's sections are scanned in
    order and increase (they do, with occasional 80A/80B-style suffixes).
    It will misfire on a corpus that's been reordered.
  - Amendment/effective-date attachment (`extract_amendment_notes`) is
    best-effort: it catches the common "<n>. Inserted/Substituted/Omitted by
    the Finance Act, <year>, w.e.f. <date>" footnote pattern, not every
    drafting style.
  - Table-heavy content (rate schedules, TDS tables) is chunked as ordinary
    text; if the eval set includes table lookups, that needs a dedicated
    table extractor, not this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # tiktoken missing, or its encoding file couldn't be fetched — fall back to a rough estimate

    def count_tokens(text: str) -> int:
        return len(text) // 4


# --------------------------------------------------------------------------
# Structural regexes
# --------------------------------------------------------------------------

# "CHAPTER VIII" on its own line
CHAPTER_RE = re.compile(r"^[ \t]*CHAPTER\s+([IVXLC]+)[ \t]*$", re.MULTILINE)

# "SCHEDULE XV" / "THE FIRST SCHEDULE" on its own line
SCHEDULE_RE = re.compile(r"^[ \t]*(?:THE\s+)?SCHEDULE[ \t]*([IVXLC]*)[ \t]*$", re.MULTILINE)

# "[See section 123]" — the line under a Schedule heading naming its section
SCHEDULE_REF_RE = re.compile(r"^\[See sections? ([^\]]+)\]$", re.IGNORECASE)

# Schedule paragraphs: "1. For any tax year, the following amounts..."
PARAGRAPH_RE = re.compile(r"(?m)^[ \t]*\d{1,3}\.[ \t]+(?=\S)")

# The PDF's rupee glyph extracts as a backtick ("`  1,50,000"); every one of
# the corpus's 228 backticks precedes an amount. Swapped 1:1 so character
# offsets (and therefore page mapping) are unchanged.
RUPEE_GLYPH = "`"

# "B.—Deductions in respect of certain payments"
PART_RE = re.compile(r"^[ \t]*([A-Z])\.—(.+)$", re.MULTILINE)

# Section marker: number, period, one-or-more spaces, then real text.
# e.g. "123.  An individual or a Hindu undivided family..." (two spaces)
# e.g. "10. If a husband and wife are governed..." (one space — PDF kerning
# makes the run of spaces after "." inconsistent, so we can't require 2+;
# the monotonic-number guard in find_section_boundaries() is what actually
# keeps Schedule/table paragraph numbers, which use the same "<n>. text"
# style, from being mistaken for sections).
SECTION_MARKER_RE = re.compile(r"(?m)^[ \t]*(\d{1,3}[A-Z]{0,2})\.[ \t]+(?=\S)")

# Sub-section: "(1)", "(2)", ... at the start of a line
SUBSECTION_RE = re.compile(r"(?m)^[ \t]*\((\d{1,3})\)[ \t]")

# Clause: "(a)", "(b)" — PDF extraction sometimes inserts a space before the
# letter ("( a)"), so tolerate that.
CLAUSE_RE = re.compile(r"(?m)^[ \t]*\([ \t]?([a-z])\)[ \t]")

# Any parenthetical list marker at all — clause (a)/(b), sub-clause roman
# numerals (i)/(ii)/(iii), capital sub-sub-clauses (A)/(B), or a bare digit
# (1)/(2). Used only to recognize "this line is still inside a list", so
# title extraction doesn't walk back into the previous section's body.
ANY_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*\([ \t]?([a-zA-Z]{1,4}|\d{1,3})\)[ \t]")

# Amendment footnote, e.g. "82. Inserted by the Finance Act, 2026, w.e.f. 1-4-2026."
AMENDMENT_RE = re.compile(
    r"(?m)^[ \t]*(\d{1,3})\.[ \t]+"
    r"(Inserted|Substituted|Omitted|Amended)\b.*?"
    r"w\.e\.f\.[ \t]+([\d]{1,2}-[\d]{1,2}-[\d]{4})"
)

# Inline marker referencing a footnote, e.g. "82[or any co-operative society...]"
INLINE_AMENDMENT_REF_RE = re.compile(r"(\d{1,3})\[")

MAX_TITLE_LOOKBACK_CHARS = 300


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Step 1: concatenate pages, keep an offset -> page-number index
# --------------------------------------------------------------------------

def concat_pages(docs: list) -> tuple[str, list[tuple[int, int, int]]]:
    """Join LangChain Documents into one string; return (text, page_spans).

    page_spans is a list of (start_char, end_char, page_number) so any later
    character offset can be mapped back to the PDF page it came from.
    """
    parts = []
    page_spans = []
    cursor = 0
    for i, doc in enumerate(docs):
        content = doc.page_content
        start = cursor
        parts.append(content)
        cursor += len(content)
        page_num = doc.metadata.get("page", i)
        page_spans.append((start, cursor, page_num))
        # separator between pages so section markers never fuse across a
        # page break (e.g. "...end of page69.  Next section" )
        parts.append("\n")
        cursor += 1
    return "".join(parts), page_spans


def _page_for_offset(page_spans: list[tuple[int, int, int]], offset: int) -> Optional[int]:
    for start, end, page_num in page_spans:
        if start <= offset < end:
            return page_num
    return page_spans[-1][2] if page_spans else None


# --------------------------------------------------------------------------
# Step 2: find section boundaries (with monotonic-number false-positive guard)
# --------------------------------------------------------------------------

def _section_sort_key(num: str) -> tuple[int, str]:
    """'80C' -> (80, 'C') so 80 < 80A < 80C < 81 sorts correctly."""
    m = re.match(r"(\d+)([A-Z]*)", num)
    return (int(m.group(1)), m.group(2))


def find_section_boundaries(text: str) -> list[dict]:
    """Return confirmed section boundaries as dicts with number/title/start/end.

    A candidate "<number>.  " marker is accepted as a real section start only
    if its number is >= the previous confirmed section's number (monotonic).
    This is what filters out Schedule/table paragraph numbering, which also
    uses the "<n>.  text" style but restarts from 1.
    """
    candidates = list(SECTION_MARKER_RE.finditer(text))
    confirmed = []
    last_key = (0, "")

    for m in candidates:
        num = m.group(1)
        key = _section_sort_key(num)
        # allow small equal/step-back tolerance for suffixed sections
        # (80, 80A, 80B... vs a fresh "1." that resets a schedule)
        if key < last_key and not (key[0] == last_key[0]):
            continue  # looks like a restart -> schedule/table paragraph, skip
        last_key = key

        # heading = the short block of text right before this marker
        lookback_start = max(0, m.start() - MAX_TITLE_LOOKBACK_CHARS)
        preceding = text[lookback_start : m.start()]
        title, consumed = _extract_title(preceding)
        title_start = m.start() - consumed

        confirmed.append(
            {
                "number": num,
                "title": title,
                "marker_start": m.start(),
                "title_start": title_start,
                "body_start": m.end(),
            }
        )

    # fill in each section's end. Use the *next* section's title_start, not
    # its marker_start — the next section's heading text sits in the stream
    # before its "<n>. " marker, so ending at marker_start would pull that
    # heading into the tail of *this* section's chunk instead.
    for i, sec in enumerate(confirmed):
        if i + 1 < len(confirmed):
            nxt = confirmed[i + 1]
            sec["end"] = nxt["title_start"] if nxt["title"] else nxt["marker_start"]
        else:
            sec["end"] = len(text)

    return confirmed


def _extract_title(preceding_text: str) -> tuple[str, int]:
    """Pull the heading line(s) directly before a section marker.

    Headings are short (usually one sentence, sometimes wrapped over two
    lines) and end with a period, e.g.:
        "Deduction in respect of employer and assessee contribution to
         pension scheme of Central Government."
    We take trailing non-blank lines up to the first blank line or up to a
    clause/sub-section marker (which would mean we've walked back into the
    previous section's body, not a heading).

    Returns (title, consumed_chars): consumed_chars is how many characters,
    counting back from the end of preceding_text, the heading occupies — the
    caller uses this to know exactly where the heading starts, so the
    *previous* section's chunk can be cut off before it instead of after it.
    """
    raw_lines = preceding_text.splitlines(keepends=True)
    nonblank_idxs = [i for i, ln in enumerate(raw_lines) if ln.strip()]
    if not nonblank_idxs:
        return "", 0

    nearest_idx = nonblank_idxs[-1]
    nearest = raw_lines[nearest_idx].strip()
    if ANY_LIST_MARKER_RE.match(nearest):
        return "", 0  # walked straight into the previous section's list body

    use_idxs = [nearest_idx]
    # a heading ends with terminal punctuation on its last physical line; if
    # the line immediately before it does NOT end with terminal punctuation,
    # that line is a wrap-continuation of the same heading and belongs with
    # it. If it DOES end with terminal punctuation, it's a separate, already
    # -complete sentence (almost always the tail of the previous section's
    # body) and must not be pulled in.
    if len(nonblank_idxs) >= 2:
        prev_idx = nonblank_idxs[-2]
        prev_line = raw_lines[prev_idx].strip()
        if (
            not ANY_LIST_MARKER_RE.match(prev_line)
            and prev_line
            and prev_line[-1] not in ".:;"
        ):
            use_idxs.insert(0, prev_idx)

    title = " ".join(raw_lines[i].strip() for i in use_idxs)
    consumed = sum(len(raw_lines[i]) for i in range(use_idxs[0], len(raw_lines)))
    return title, consumed


def find_schedules(text: str) -> list[dict]:
    """Return each Schedule as {number, title, ref, start, body_start, end}.

    Heading layout in the corpus:
        SCHEDULE XV
        [See section 123]
        DEDUCTION IN RESPECT OF LIFE INSURANCE PREMIA,
         CONTRIBUTION TO PROVIDENT FUND, ...
        Sums qualifying as deduction.
        1. For any tax year, ...
    The title is the run of upper-case lines after the "[See section]" line.
    """
    matches = list(SCHEDULE_RE.finditer(text))
    schedules = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        lines = text[m.end() : end].splitlines(keepends=True)

        ref, title_lines, consumed = None, [], 0
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                consumed += len(ln)
                continue
            ref_m = SCHEDULE_REF_RE.match(stripped)
            if ref_m and ref is None and not title_lines:
                ref = ref_m.group(1).strip()
            elif stripped.upper() == stripped and any(c.isalpha() for c in stripped):
                title_lines.append(stripped)
            else:
                break
            consumed += len(ln)

        title = " ".join(title_lines).rstrip(",")
        title = re.sub(r"\s+", " ", title).capitalize() if title else ""
        schedules.append(
            {
                "number": f"Schedule {m.group(1)}",
                "title": title,
                "ref": ref,
                "start": m.start(),
                "body_start": m.end() + consumed,
                "end": end,
            }
        )
    return schedules


# --------------------------------------------------------------------------
# Step 3: structural context (chapter / part / schedule) per offset
# --------------------------------------------------------------------------

def build_context_index(text: str) -> list[tuple[int, str, str]]:
    """Return a sorted list of (offset, kind, value) markers for chapter/part."""
    markers = []
    for m in CHAPTER_RE.finditer(text):
        markers.append((m.start(), "chapter", f"Chapter {m.group(1)}"))
    for m in PART_RE.finditer(text):
        markers.append((m.start(), "part", f"{m.group(1)}.—{m.group(2).strip()}"))
    markers.sort(key=lambda x: x[0])
    return markers


def _context_at(markers: list[tuple[int, str, str]], offset: int) -> dict:
    chapter, part = None, None
    for m_offset, kind, value in markers:
        if m_offset > offset:
            break
        if kind == "chapter":
            chapter, part = value, None  # a new chapter resets the part
        elif kind == "part":
            part = value
    return {"chapter": chapter, "part": part}


# --------------------------------------------------------------------------
# Step 4: amendment notes (best-effort)
# --------------------------------------------------------------------------

def extract_amendment_notes(text: str) -> dict[str, dict]:
    """Map footnote number -> {change_type, effective_date} for the whole doc."""
    notes = {}
    for m in AMENDMENT_RE.finditer(text):
        footnote_num, change_type, eff_date = m.groups()
        notes[footnote_num] = {"change_type": change_type, "effective_date": eff_date}
    return notes


def _amendments_in(section_text: str, notes_by_footnote: dict[str, dict]) -> list[dict]:
    found = []
    for ref_num in set(m.group(1) for m in INLINE_AMENDMENT_REF_RE.finditer(section_text)):
        if ref_num in notes_by_footnote:
            found.append({"footnote": ref_num, **notes_by_footnote[ref_num]})
    return found


# --------------------------------------------------------------------------
# Step 5: size-bounded splitting inside a section, never across a boundary
# --------------------------------------------------------------------------

_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n(", "\n ( ", ". ", "\n", " "],
)


def _split_by_marker(text: str, marker_re: re.Pattern) -> list[str]:
    """Split text at each marker match, keeping the marker with the piece that follows."""
    matches = list(marker_re.finditer(text))
    if not matches:
        return [text]
    pieces = []
    if matches[0].start() > 0:
        pieces.append(text[: matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pieces.append(text[m.start() : end])
    return [p for p in pieces if p.strip()]


def _split_section_body(body: str, max_tokens: int, hard_max_tokens: int) -> list[str]:
    """Sub-section, then clause, then fallback — in that order of preference."""
    if count_tokens(body) <= max_tokens:
        return [body]

    pieces = _split_by_marker(body, SUBSECTION_RE)
    if len(pieces) == 1:  # no sub-sections found — try clauses directly
        pieces = _split_by_marker(body, CLAUSE_RE)

    result = []
    for piece in pieces:
        if count_tokens(piece) <= hard_max_tokens:
            result.append(piece)
            continue
        # still too big: try clause-level split within this sub-section
        sub_pieces = _split_by_marker(piece, CLAUSE_RE)
        if len(sub_pieces) > 1:
            for sp in sub_pieces:
                if count_tokens(sp) <= hard_max_tokens:
                    result.append(sp)
                else:
                    result.extend(_FALLBACK_SPLITTER.split_text(sp))
        else:
            result.extend(_FALLBACK_SPLITTER.split_text(piece))
    return result


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def chunk_documents(
    docs: list,
    act_name: str,
    source_file: str,
    max_tokens: int = 700,
    hard_max_tokens: int = 1200,
) -> list[Chunk]:
    """Turn loaded page-level Documents into section-aware chunks with metadata."""
    text, page_spans = concat_pages(docs)
    text = text.replace(RUPEE_GLYPH, "₹")
    schedules = find_schedules(text)
    first_schedule = schedules[0]["start"] if schedules else len(text)
    sections = [
        s for s in find_section_boundaries(text) if s["marker_start"] < first_schedule
    ]
    for sec in sections:
        sec["end"] = min(sec["end"], first_schedule)
    context_markers = build_context_index(text)
    amendment_notes = extract_amendment_notes(text)

    chunks: list[Chunk] = []
    for sec in sections:
        full_section_text = text[sec["marker_start"] : sec["end"]].strip()
        pieces = _split_section_body(full_section_text, max_tokens, hard_max_tokens)
        ctx = _context_at(context_markers, sec["marker_start"])
        page_start = _page_for_offset(page_spans, sec["marker_start"])
        page_end = _page_for_offset(page_spans, max(sec["end"] - 1, sec["marker_start"]))

        total = len(pieces)
        for idx, piece in enumerate(pieces, start=1):
            header = f"Section {sec['number']}"
            # Name the chapter: provisions say "this Chapter", while users
            # (and competing sections elsewhere in the Act) say "Chapter
            # VIII" — without it, 122(2) lost to s.198 on that exact phrase.
            if ctx["chapter"]:
                header += f" ({ctx['chapter']})"
            if sec["title"]:
                header += f" — {sec['title']}"
            if total > 1:
                header += f" (part {idx}/{total})"
            chunk_text = f"{header}\n\n{piece.strip()}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "act_name": act_name,
                        "source_file": source_file,
                        "section_number": sec["number"],
                        "section_title": sec["title"],
                        "chapter": ctx["chapter"],
                        "part": ctx["part"],
                        "chunk_index": idx,
                        "chunk_count": total,
                        "page_start": page_start,
                        "page_end": page_end,
                        "amendments": _amendments_in(piece, amendment_notes) or None,
                    },
                )
            )

    for sch in schedules:
        body = text[sch["body_start"] : sch["end"]].strip()
        if not body:
            continue
        pieces = []
        for para in _split_by_marker(body, PARAGRAPH_RE):
            pieces.extend(_split_section_body(para, max_tokens, hard_max_tokens))
        page_start = _page_for_offset(page_spans, sch["start"])
        page_end = _page_for_offset(page_spans, max(sch["end"] - 1, sch["start"]))

        total = len(pieces)
        for idx, piece in enumerate(pieces, start=1):
            header = sch["number"]
            if sch["ref"]:
                header += f" (see Section {sch['ref']})"
            if sch["title"]:
                header += f" — {sch['title']}"
            if total > 1:
                header += f" (part {idx}/{total})"

            chunks.append(
                Chunk(
                    text=f"{header}\n\n{piece.strip()}",
                    metadata={
                        "act_name": act_name,
                        "source_file": source_file,
                        "section_number": sch["number"],
                        "section_title": sch["title"],
                        "chapter": None,
                        "part": f"See section {sch['ref']}" if sch["ref"] else None,
                        "chunk_index": idx,
                        "chunk_count": total,
                        "page_start": page_start,
                        "page_end": page_end,
                        "amendments": _amendments_in(piece, amendment_notes) or None,
                    },
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Debug helper — run standalone to sanity-check boundary detection
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from rag_core.ingestion_pipeline import DEFAULT_PDF, load_documents

    docs = load_documents(str(DEFAULT_PDF))
    chunks = chunk_documents(
        docs,
        act_name="The Income-tax Act, 2025 (as amended by FA Act 2026)",
        source_file=DEFAULT_PDF.name,
    )

    print(f"{len(chunks)} chunks from {len(docs)} pages\n")
    seen_sections = []
    for c in chunks:
        if c.metadata["chunk_index"] == 1:
            seen_sections.append(c.metadata["section_number"])

    print("First 20 detected section numbers, in order:")
    print(seen_sections[:20])

    token_counts = [count_tokens(c.text) for c in chunks]
    print(f"\ntoken count: min={min(token_counts)} max={max(token_counts)} "
          f"avg={sum(token_counts)//len(token_counts)}")

    # print one multi-part section as a spot check, if any
    for c in chunks:
        if c.metadata["chunk_count"] > 1:
            print("\n--- sample multi-part chunk ---")
            print(c.metadata)
            print(c.text[:400])
            break
