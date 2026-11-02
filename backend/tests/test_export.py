import csv
import io

from openpyxl import load_workbook

from app.export import to_csv, to_xlsx

ROWS = [{"text": "UBER *TRIP SF | $23.1", "label": None, "meta": {"row": 2, "Date": "2026-10-01", "Merchant": "UBER *TRIP SF", "Amount": "23.1"}},
        {"text": "SWEETGREEN | $14.5", "label": None, "meta": {"row": 3, "Date": "2026-10-02", "Merchant": "SWEETGREEN", "Amount": "14.5"}}]
PREDS = [{"label": "Travel", "confidence": 0.97, "top": [{"label": "Travel", "p": 0.97}, {"label": "Meals", "p": 0.02}]},
         {"label": "Meals", "confidence": 0.41, "needs_review": True, "top": [{"label": "Meals", "p": 0.41}, {"label": "Office", "p": 0.39}]}]


def test_csv_keeps_original_columns_and_appends_results():
    rows = list(csv.reader(io.StringIO(to_csv(ROWS, PREDS).decode())))
    assert rows[0] == ["Date", "Merchant", "Amount", "prediction", "confidence", "needs_review", "runner_up"]
    assert rows[1] == ["2026-10-01", "UBER *TRIP SF", "23.1", "Travel", "0.97", "", "Meals"]
    assert rows[2][3:] == ["Meals", "0.41", "yes", "Office"]


def test_xlsx_highlights_review_rows():
    ws = load_workbook(io.BytesIO(to_xlsx(ROWS, PREDS, "Expenses"))).active
    assert ws.title == "Expenses" and ws["D1"].value == "prediction" and ws["D2"].value == "Travel"
    assert ws["F3"].value == "yes" and ws["F3"].fill.fgColor.rgb.endswith("FFF3C4") and ws.freeze_panes == "A2"


def test_rows_without_meta_fall_back_to_text_column():
    rows = list(csv.reader(io.StringIO(to_csv([{"text": "hello", "meta": {}}], [{"label": "x", "confidence": None}]).decode())))
    assert rows[0][0] == "text" and rows[1][:2] == ["hello", "x"]
