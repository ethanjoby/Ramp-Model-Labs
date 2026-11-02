"""Turns what finance teams actually have (emails, spreadsheets, PDFs, CSV/JSON exports, pasted text) into rows.

Every parser is pure Python and read-only: no formulas are evaluated, no macros run, nothing is fetched.
A row is {"text": str, "label": str | None, "meta": {...}}; meta keeps the original cells so a workflow
can hand results back in the shape they arrived in.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import email
from email import policy
import html
import io
import json
import re
from typing import Any

MAX_ROWS = 5000
MAX_CELL = 600
MAX_PDF_PAGES = 60
TEXT_HINTS = ("text", "description", "memo", "merchant", "vendor", "narrative", "details", "input", "message", "subject",
              "body", "line item", "line_item", "item", "note", "notes", "payee", "name", "transaction", "descriptor")
LABEL_HINTS = ("label", "category", "class", "target", "gl", "gl code", "gl_code", "account", "intent", "decision",
               "status", "type", "queue", "verdict", "outcome")
AMOUNT_HINTS = ("amount", "total", "value", "sum", "usd", "price", "debit", "credit")


@dataclass
class Parsed:
    kind: str                                  # email | spreadsheet | csv | json | pdf | text
    rows: list[dict[str, Any]]
    columns: list[str] = field(default_factory=list)
    text_column: str | None = None
    label_column: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def labeled(self) -> int:
        return sum(1 for r in self.rows if r.get("label"))

    def labels(self) -> list[str]:
        return sorted({r["label"] for r in self.rows if r.get("label")})[:50]

    def as_jsonl(self) -> str:
        """What the lab pipeline consumes: parse_examples() understands JSONL with text/label."""
        return "\n".join(json.dumps({"text": r["text"], "label": r.get("label")}) for r in self.rows)

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "count": len(self.rows), "labeled": self.labeled, "labels": self.labels(),
                "columns": self.columns, "text_column": self.text_column, "label_column": self.label_column,
                "notes": self.notes, "preview": [{"text": r["text"], "label": r.get("label")} for r in self.rows[:5]]}


def _clean(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return re.sub(r"\s+", " ", str(v)).strip()[:MAX_CELL]


def _row(text: str, label: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"text": text, "label": (label or None), "meta": meta or {}}


# ------------------------------------------------------------------ dispatch

def parse_upload(filename: str, data: bytes, content_type: str = "") -> Parsed:
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext in ("xlsx", "xlsm", "xltx", "xltm") or "spreadsheetml" in content_type or data[:2] == b"PK" and ext != "zip":
        return parse_spreadsheet(data)
    if ext == "pdf" or data[:5] == b"%PDF-":
        return parse_pdf(data)
    text = data.decode("utf-8", errors="replace")
    if ext == "eml" or content_type == "message/rfc822" or looks_like_email(text):
        return parse_email(data)
    if ext in ("json", "jsonl") or text.lstrip()[:1] in ("[", "{"):
        return parse_json(text)
    if ext in ("csv", "tsv") or looks_like_delimited(text):
        return parse_delimited(text)
    return parse_text(text)


# ------------------------------------------------------------------ tabular

def _infer_columns(headers: list[str], rows: list[list[Any]]) -> tuple[str | None, str | None, str | None, list[str]]:
    """Which column is the text, which the label, which the amount. Hints first, then shape of the data."""
    notes: list[str] = []
    low = [h.lower().strip() for h in headers]

    def by_hint(hints):
        for i, h in enumerate(low):
            if any(h == k or h.startswith(k) or k in h.split() for k in hints):
                return headers[i]
        return None

    def col(name):
        i = headers.index(name)
        return [_clean(r[i]) if i < len(r) else "" for r in rows]

    text_col = by_hint(TEXT_HINTS)
    if not text_col:
        best, best_len = None, 0.0
        for h in headers:
            vals = [v for v in col(h) if v]
            if not vals:
                continue
            words = sum(len(v.split()) for v in vals) / len(vals)
            if words > best_len:
                best, best_len = h, words
        text_col = best
        if text_col:
            notes.append(f"No obvious text column; using '{text_col}' because its cells have the most words.")

    label_col = by_hint(LABEL_HINTS)
    if label_col == text_col:
        label_col = None
    if not label_col:
        for h in headers:
            if h == text_col:
                continue
            vals = [v for v in col(h) if v]
            if len(vals) < max(4, len(rows) // 2):
                continue
            distinct = {v.lower() for v in vals}
            numeric = sum(bool(re.fullmatch(r"[-+$]?[\d,]*\.?\d+%?", v)) for v in vals) / len(vals)
            if 2 <= len(distinct) <= min(50, len(vals) // 2) and numeric < 0.2 and all(len(v) <= 40 for v in vals):
                label_col = h
                notes.append(f"Treating '{h}' as the label column: {len(distinct)} distinct short values.")
                break
    amount_col = by_hint(AMOUNT_HINTS)
    if amount_col in (text_col, label_col):
        amount_col = None
    return text_col, label_col, amount_col, notes


def _tabular(kind: str, headers: list[str], rows: list[list[Any]], extra_notes: list[str] | None = None) -> Parsed:
    headers = [_clean(h) or f"column_{i + 1}" for i, h in enumerate(headers)]
    rows = rows[:MAX_ROWS]
    text_col, label_col, amount_col, notes = _infer_columns(headers, rows)
    out: list[dict[str, Any]] = []
    if text_col:
        ti, li, ai = headers.index(text_col), headers.index(label_col) if label_col else None, headers.index(amount_col) if amount_col else None
        for n, r in enumerate(rows, start=2):
            cells = {h: _clean(r[i]) if i < len(r) else "" for i, h in enumerate(headers)}
            text = cells[text_col]
            if not text:
                continue
            if ai is not None and cells[amount_col]:
                amt = cells[amount_col]
                text = f"{text} | {'$' + amt if re.fullmatch(r'[\d,]*\.?\d+', amt) else amt}"
            out.append(_row(text, cells[label_col] if li is not None else None, {"row": n, **{k: v for k, v in cells.items() if v}}))
    if not out:
        notes.append("Couldn't find a usable text column. Name one 'text', 'description' or 'memo'.")
    return Parsed(kind, out, headers, text_col, label_col, (extra_notes or []) + notes)


def parse_spreadsheet(data: bytes) -> Parsed:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)  # cached values only; formulas never evaluate
    try:
        for ws in wb.worksheets:
            raw = []
            for row in ws.iter_rows(values_only=True):
                if any(c not in (None, "") for c in row):
                    raw.append(list(row))
                if len(raw) > MAX_ROWS + 20:
                    break
            if len(raw) < 2:
                continue
            # header: first row with at least two short strings
            hi = next((i for i, r in enumerate(raw[:10]) if sum(isinstance(c, str) and 0 < len(c) <= 60 for c in r) >= 2), 0)
            notes = [f"Sheet '{ws.title}'"] + ([f"Skipped {hi} row(s) above the header."] if hi else [])
            if len(wb.worksheets) > 1:
                notes.append(f"{len(wb.worksheets)} sheets in the file; only the first one with data is used.")
            return _tabular("spreadsheet", raw[hi], raw[hi + 1:], notes)
    finally:
        wb.close()
    return Parsed("spreadsheet", [], notes=["No sheet with data."])


def looks_like_delimited(text: str) -> bool:
    lines = [l for l in text.splitlines()[:20] if l.strip()]
    if len(lines) < 2:
        return False
    for d in (",", "\t", ";", "|"):
        counts = [l.count(d) for l in lines]
        if counts[0] >= 1 and len(set(counts)) <= 2:
            return True
    return False


def parse_delimited(text: str) -> Parsed:
    sample = "\n".join(text.splitlines()[:30])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    reader = list(csv.reader(io.StringIO(text), dialect))
    reader = [r for r in reader if any(c.strip() for c in r)]
    if not reader:
        return Parsed("csv", [], notes=["Empty file."])
    if len(reader[0]) == 1 and all(len(r) <= 1 for r in reader):
        return parse_text(text)
    return _tabular("csv", reader[0], reader[1:])


def parse_json(text: str) -> Parsed:
    try:
        data = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    except json.JSONDecodeError:
        return parse_text(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    dicts = [d for d in data if isinstance(d, dict)][:MAX_ROWS]
    if not dicts:
        return Parsed("json", [_row(_clean(x), None, {"row": i}) for i, x in enumerate(data[:MAX_ROWS]) if _clean(x)], notes=["Array of values; each one is a row."])
    headers = list(dict.fromkeys(k for d in dicts for k in d))
    return _tabular("json", headers, [[d.get(h) for h in headers] for d in dicts])


# ------------------------------------------------------------------ email

def looks_like_email(text: str) -> bool:
    head = text.lstrip()[:600].lower()
    return head.startswith(("from:", "subject:", "delivered-to:", "received:", "return-path:")) and "subject:" in head


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s)


def parse_email(data: bytes) -> Parsed:
    msg = email.message_from_bytes(data, policy=policy.default)
    subject, sender, date = _clean(msg.get("subject")), _clean(msg.get("from")), _clean(msg.get("date"))
    body = ""
    attachments: list[Parsed] = []
    for part in msg.walk():
        ctype = part.get_content_type()
        if part.get_content_disposition() == "attachment":
            fname = part.get_filename() or ""
            if fname.lower().rsplit(".", 1)[-1] in ("csv", "tsv", "xlsx", "json", "jsonl", "txt", "pdf"):
                try:
                    attachments.append(parse_upload(fname, part.get_payload(decode=True) or b"", ctype))
                except Exception:  # a bad attachment shouldn't sink the email
                    pass
            continue
        if ctype == "text/plain" and not body:
            body = part.get_content()
        elif ctype == "text/html" and not body:
            body = _strip_html(part.get_content())
    body = re.sub(r"\n{3,}", "\n\n", body or "").strip()
    quoted = re.search(r"\n(On .{10,120} wrote:|-{3,} ?Original Message|From: .+\nSent: )", body)
    if quoted:
        body = body[:quoted.start()].strip()  # keep the newest message, drop the quoted thread
    notes = [f"From {sender or 'unknown'}" + (f", {date}" if date else "")]
    rows = [_row(f"Subject: {subject}\n{body[:2000]}".strip(), None, {"subject": subject, "sender": sender, "date": date})]
    for a in attachments:
        if a.rows:
            notes.append(f"Attachment parsed as {a.kind}: {len(a.rows)} rows.")
            rows = a.rows if len(a.rows) > 1 else rows + a.rows
    kind = "email" if not attachments else "email+attachment"
    return Parsed(kind, rows[:MAX_ROWS], notes=notes)


# ------------------------------------------------------------------ pdf and text

def parse_pdf(data: bytes, granularity: str = "auto") -> Parsed:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES]]
    notes = [f"{len(reader.pages)} page(s)" + (f", first {MAX_PDF_PAGES} read" if len(reader.pages) > MAX_PDF_PAGES else "")]
    whole = re.sub(r"[ \t]+", " ", "\n".join(pages)).strip()
    if not whole:
        return Parsed("pdf", [], notes=notes + ["No extractable text (scanned image?). OCR is not available offline."])
    if granularity == "document" or (granularity == "auto" and len(pages) <= 2 and len(whole) < 4000):
        return Parsed("pdf", [_row(whole[:4000], None, {"pages": len(pages)})], notes=notes + ["Whole document as one row."])
    rows = []
    for pn, page in enumerate(pages, start=1):
        for line in page.splitlines():
            line = _clean(line)
            if len(line.split()) >= 3:
                rows.append(_row(line, None, {"page": pn}))
    return Parsed("pdf", rows[:MAX_ROWS], notes=notes + ["One row per line with three or more words."])


def parse_text(text: str) -> Parsed:
    rows = []
    lines = text.splitlines()
    if len(lines) > 1 and lines[0].strip().lower() in TEXT_HINTS:
        lines = lines[1:]  # a one-column export still has its header; "text" is not a row
    for i, line in enumerate(lines, start=1):
        line = line.strip(" -•\t")
        if not line:
            continue
        if "=>" in line:
            t, l = line.rsplit("=>", 1)
            rows.append(_row(_clean(t), _clean(l) or None, {"line": i}))
        else:
            rows.append(_row(_clean(line), None, {"line": i}))
    return Parsed("text", rows[:MAX_ROWS], notes=["One row per line."] if len(rows) > 1 else [])
