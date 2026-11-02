import io

from openpyxl import Workbook

from app.ingest import parse_upload


def xlsx(rows, sheet="Sheet1"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_spreadsheet_infers_text_label_and_amount_columns():
    data = xlsx([["Date", "Merchant", "Amount", "GL Code"], ["2026-10-01", "UBER *TRIP SF", 23.1, "Travel"],
                 ["2026-10-02", "SWEETGREEN", 14.5, "Meals"], ["2026-10-03", "GITHUB.COM", 4, "Software"], ["2026-10-04", "DELTA AIR", 412, "Travel"]])
    p = parse_upload("expenses.xlsx", data)
    assert p.kind == "spreadsheet" and p.text_column == "Merchant" and p.label_column == "GL Code"
    assert p.rows[0]["text"] == "UBER *TRIP SF | $23.1" and p.rows[0]["label"] == "Travel"
    assert p.rows[0]["meta"]["Date"] == "2026-10-01" and p.labeled == 4 and p.labels() == ["Meals", "Software", "Travel"]
    assert p.as_jsonl().count("\n") == 3 and "UBER *TRIP SF" in p.as_jsonl()


def test_spreadsheet_without_hints_falls_back_to_shape():
    data = xlsx([["a", "b", "c"], ["x", "card-not-present $12 at LUXE WATCHES 03:00 new device", "fraud"],
                 ["y", "contactless $4 at STARBUCKS 08:00 recurring", "legitimate"], ["z", "wire $9,000 to new payee first time", "fraud"],
                 ["w", "contactless $7 at WHOLE FOODS 12:00", "legitimate"]])
    p = parse_upload("mystery.xlsx", data)
    assert p.text_column == "b" and p.label_column == "c" and any("most words" in n for n in p.notes)


def test_csv_tsv_and_json_share_the_tabular_path():
    assert parse_upload("x.csv", b"text,label\nUBER TRIP,travel\n").rows[0]["label"] == "travel"
    tsv = parse_upload("x.tsv", b"description\tcategory\nDELTA AIR\ttravel\nSWEETGREEN\tmeals\n")
    assert tsv.kind == "csv" and tsv.rows[1]["text"] == "SWEETGREEN"
    js = parse_upload("x.jsonl", b'{"memo": "GITHUB", "gl": "software"}\n{"memo": "HERTZ", "gl": "travel"}')
    assert js.kind == "json" and js.rows[1]["label"] == "travel"


def test_email_keeps_newest_message_and_parses_csv_attachment():
    raw = b"""From: ap@vendor.com\r\nTo: billing@ramp.com\r\nSubject: Invoice INV-7781 overdue\r\nDate: Thu, 30 Oct 2026 10:00:00 -0700\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary="b1"\r\n\r\n--b1\r\nContent-Type: text/plain\r\n\r\nHi team, invoice INV-7781 for $3,867.82 is 12 days past due. Please advise.\r\n\r\nOn Oct 20, 2026, Someone <x@y.com> wrote:\r\n> earlier thread\r\n--b1\r\nContent-Type: text/csv\r\nContent-Disposition: attachment; filename="lines.csv"\r\n\r\ntext,label\r\nNorthwind INV-1 $100; no PO,needs_review\r\nAcme INV-2 $50; matches PO,approve\r\n--b1--\r\n"""
    p = parse_upload("msg.eml", raw)
    assert p.kind == "email+attachment" and len(p.rows) == 2 and p.rows[0]["label"] == "needs_review"
    single = parse_upload("msg.eml", raw.split(b"--b1\r\nContent-Type: text/csv")[0] + b"--b1--\r\n")
    assert single.kind == "email" and single.rows[0]["text"].startswith("Subject: Invoice INV-7781 overdue")
    assert "earlier thread" not in single.rows[0]["text"] and single.rows[0]["meta"]["sender"] == "ap@vendor.com"


def test_pasted_email_text_is_detected():
    p = parse_upload("", b"From: cardholder@ramp.com\nSubject: card declined at Delta\n\nMy card was declined twice at Delta this morning, help?")
    assert p.kind == "email" and "card declined" in p.rows[0]["text"]


def test_pdf_single_page_is_one_row_and_long_pdfs_split_into_lines():
    def pdf(lines_per_page, pages):
        import pypdf
        w = pypdf.PdfWriter()
        for pn in range(pages):
            content = "BT /F1 12 Tf 40 760 Td 14 TL " + " ".join(f"(Line item {pn}-{i} widget qty 2 $45.00) Tj T*" for i in range(lines_per_page)) + " ET"
            page = w.add_blank_page(width=612, height=792)
            from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(content.encode())
            page[NameObject("/Contents")] = w._add_object(stream)
        buf = io.BytesIO()
        w.write(buf)
        return buf.getvalue()

    one = parse_upload("invoice.pdf", pdf(3, 1))
    assert one.kind == "pdf" and len(one.rows) == 1 and "Line item 0-0" in one.rows[0]["text"]
    many = parse_upload("statement.pdf", pdf(10, 4))
    assert len(many.rows) == 40 and many.rows[-1]["meta"]["page"] == 4


def test_plain_text_lines_and_arrow_labels():
    p = parse_upload("notes.txt", b"UBER *TRIP => travel\n- SWEETGREEN\n\nHERTZ RENT A CAR => travel")
    assert [(r["text"], r["label"]) for r in p.rows] == [("UBER *TRIP", "travel"), ("SWEETGREEN", None), ("HERTZ RENT A CAR", "travel")]
