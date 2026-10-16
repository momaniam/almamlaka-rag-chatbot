from rag.language import detect_language


def test_english():
    assert detect_language("What is the total budget?") == "en"


def test_arabic():
    assert detect_language("متى موعد الإطلاق؟") == "ar"


def test_mixed_arabic_majority_with_latin_terms():
    assert detect_language("ما هي ميزانية Q2 للتسويق؟") == "ar"


def test_mixed_mostly_english():
    assert detect_language("What does مرحبا mean in this project document context?") == "en"


def test_no_letters_uses_fallback():
    assert detect_language("2027?", "ar") == "ar"
    assert detect_language("???") == "en"
    assert detect_language("", "en") == "en"
    assert detect_language("١٢٣", "ar") == "ar"
