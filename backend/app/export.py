"""Writes workflow results back in the shape people work in: a spreadsheet with the original columns intact
and the model's decision, confidence and review flag added at the end."""
from __future__ import annotations

import csv
import io
from typing import Any

RESULT_COLUMNS = ("prediction", "confidence", "needs_review", "runner_up")


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        for k in (r.get("meta") or {}):
            if k != "row":
                seen.setdefault(k, None)
    if not seen:
        seen["text"] = None
    return list(seen)


def _record(row: dict[str, Any], pred: dict[str, Any], columns: list[str]) -> list[Any]:
    meta = row.get("meta") or {}
    base = [meta.get(c, row["text"] if c == "text" else "") for c in columns]
    top = pred.get("top") or []
    runner_up = top[1]["label"] if len(top) > 1 else ""
    conf = pred.get("confidence")
    return base + [pred.get("label", ""), round(conf, 4) if conf is not None else "", "yes" if pred.get("needs_review") else "", runner_up]


def to_csv(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> bytes:
    columns = _columns(rows)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns + list(RESULT_COLUMNS))
    for r, p in zip(rows, preds):
        w.writerow(_record(r, p, columns))
    return buf.getvalue().encode()


def to_xlsx(rows: list[dict[str, Any]], preds: list[dict[str, Any]], title: str = "Results") -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    columns = _columns(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = title[:31] or "Results"
    header = columns + list(RESULT_COLUMNS)
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    review_fill = PatternFill("solid", fgColor="FFF3C4")
    for r, p in zip(rows, preds):
        ws.append(_record(r, p, columns))
        if p.get("needs_review"):
            for cell in ws[ws.max_row]:
                cell.fill = review_fill
    for i, name in enumerate(header, start=1):
        width = max(len(str(name)), *(len(str(ws.cell(row=j, column=i).value or "")) for j in range(2, min(ws.max_row, 200) + 1)))
        ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, width + 2))
    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
