from app.agents.data import parse_examples


def test_csv_with_aliased_columns():
    rows = parse_examples("memo,category\nUBER TRIP,travel\nSWEETGREEN,meals\n")
    assert rows == [{"text": "UBER TRIP", "label": "travel"}, {"text": "SWEETGREEN", "label": "meals"}]


def test_csv_without_label_column_gives_unlabeled_rows():
    rows = parse_examples("text\nUBER TRIP\n")
    assert rows == [{"text": "UBER TRIP", "label": None}]


def test_jsonl_and_json_array():
    assert parse_examples('{"text": "a", "label": "x"}\n{"input": "b", "category": "y"}') == [
        {"text": "a", "label": "x"}, {"text": "b", "label": "y"}]
    assert parse_examples('[{"description": "c", "label": "z"}]') == [{"text": "c", "label": "z"}]


def test_arrow_lines_and_blank_input():
    assert parse_examples("DELTA AIR => travel\nno label here =>") == [
        {"text": "DELTA AIR", "label": "travel"}, {"text": "no label here", "label": None}]
    assert parse_examples("   ") == []


def test_row_cap():
    big = "text,label\n" + "\n".join(f"row {i},a" for i in range(6000))
    assert len(parse_examples(big)) == 5000
