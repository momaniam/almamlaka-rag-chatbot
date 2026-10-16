from conftest import make_chunk
from rag.formatting import citation_label, normalise_digits, safe_html


def test_citation_label_single_page():
    c = make_chunk("Doc_A", 3, "Key Details", "x", page=1, doc_title="Project Overview Document")
    assert citation_label(c) == "Project Overview Document, p. 1, §3 Key Details"


def test_citation_label_page_range():
    c = make_chunk("Doc_A", 2, "Notes", "x", page=1, doc_title="Doc")
    c.page_end = 2
    assert citation_label(c) == "Doc, p. 1–2, §2 Notes"


def test_safe_html_escapes_script_and_sets_rtl():
    out = safe_html('<script>alert("x")</script>\nسطر', "ar")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert out.startswith('<div dir="rtl" lang="ar" class="rtl">')
    assert "<br>" in out


def test_safe_html_ltr_and_unknown_lang():
    assert safe_html("hi", "en").startswith('<div dir="ltr" lang="en" class="ltr">')
    assert 'dir="ltr"' in safe_html("hi", 'x" onclick="evil')


def test_normalise_digits():
    assert normalise_digits("١٥ مارس ٢٠٢٧") == "15 مارس 2027"
    assert normalise_digits("۲۴") == "24"
