from forgeai.loader import chunk, load_folder


def test_loads_text_and_csv_and_keeps_csv_rows_whole(tmp_path):
    (tmp_path / "notes.txt").write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "id,note\n1,alpha\n2,beta\n3,gamma\n",
        encoding="utf-8",
    )

    docs = load_folder(str(tmp_path))

    assert [doc.doc_id for doc in docs] == ["notes.txt", "records.csv"]
    text_doc, csv_doc = docs
    assert text_doc.text == "First paragraph.\n\nSecond paragraph."
    assert csv_doc.text == (
        "id: 1 | note: alpha\n"
        "id: 2 | note: beta\n"
        "id: 3 | note: gamma"
    )

    csv_chunks = chunk(csv_doc, size=24, overlap=0)
    assert [line for part in csv_chunks for line in part.text.splitlines()] == [
        "id: 1 | note: alpha",
        "id: 2 | note: beta",
        "id: 3 | note: gamma",
    ]


def test_text_chunks_carry_overlap(tmp_path):
    text_path = tmp_path / "long.txt"
    text_path.write_text(
        "abcdefghij klmnopqrst uvwxyz0123 456789ABCDEFGHIJ KLMNOPQRSTUVWXYZ "
        "abcdefghijklmnopqrstuvwxyz",
        encoding="utf-8",
    )
    doc = load_folder(str(tmp_path))[0]

    chunks = chunk(doc, size=30, overlap=8)

    assert len(chunks) > 1
    assert chunks[1].text.startswith(chunks[0].text[-8:])
