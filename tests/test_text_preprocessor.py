import modules.text_preprocessor as tp


def test_clean_text_removes_sensitive_patterns_and_keeps_money_token():
    text = "Visit https://example.com XX and email me@x.com. Paid $1,234 in 2023!!!"
    cleaned = tp.clean_text(text)

    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "xx" not in cleaned
    assert "2023" not in cleaned
    assert "moneytok" in cleaned


def test_preprocess_text_lemmatization_pipeline(monkeypatch):
    monkeypatch.setattr(tp, "tokenize", lambda s: s.split())
    monkeypatch.setattr(
        tp,
        "remove_stopwords",
        lambda tokens, extra_stopwords=None: [t for t in tokens if t not in {"and", "the"}],
    )
    monkeypatch.setattr(
        tp, "lemmatize_tokens", lambda tokens: [t[:-1] if t.endswith("s") else t for t in tokens]
    )

    out = tp.preprocess_text("Cats and dogs.")
    assert out == "cat dog"


def test_preprocess_batch_preserves_order(monkeypatch):
    def fake_preprocess_text(text, **kwargs):
        return f"clean::{text}"

    monkeypatch.setattr(tp, "preprocess_text", fake_preprocess_text)
    out = tp.preprocess_batch(["a", "b", "c"], n_jobs=1)
    assert out == ["clean::a", "clean::b", "clean::c"]
