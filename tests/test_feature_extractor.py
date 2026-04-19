import numpy as np
import scipy.sparse as sp

from modules.feature_extractor import (
    build_bow,
    build_tfidf,
    load_features,
    load_vectorizer,
    save_features,
    save_vectorizer,
)


def test_build_bow_outputs_sparse_matrix_and_vocab():
    train = ["red car", "blue car", "red bike"]
    test = ["red car"]
    x_train, x_test, vec = build_bow(train, test, max_features=20, min_df=1, binary=True)

    assert sp.issparse(x_train)
    assert x_train.shape[0] == 3
    assert x_test.shape[0] == 1
    assert "car" in vec.vocabulary_
    assert "red" in vec.vocabulary_


def test_build_tfidf_bigram_contains_expected_ngram():
    train = ["red car", "blue car", "red bike"]
    test = ["red car"]
    _, _, vec = build_tfidf(train, test, max_features=50, min_df=1, ngram_range=(1, 2))
    feats = set(vec.get_feature_names_out())
    assert "red car" in feats


def test_save_and_load_sparse_features_roundtrip(tmp_path):
    x_train = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    x_test = sp.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float32))
    y_train = np.array([0, 1], dtype=np.int64)
    y_test = np.array([1], dtype=np.int64)

    save_features(str(tmp_path), "toy_sparse", x_train, x_test, y_train, y_test)
    xtr2, xte2, ytr2, yte2 = load_features(str(tmp_path), "toy_sparse", sparse=True)

    assert sp.issparse(xtr2) and sp.issparse(xte2)
    assert xtr2.shape == x_train.shape
    assert xte2.shape == x_test.shape
    np.testing.assert_array_equal(ytr2, y_train)
    np.testing.assert_array_equal(yte2, y_test)


def test_save_and_load_vectorizer_roundtrip(tmp_path):
    train = ["red car", "blue car", "red bike"]
    test = ["red car"]
    _, _, vec = build_bow(train, test, max_features=20, min_df=1)

    path = tmp_path / "vec.pkl"
    save_vectorizer(vec, str(path))
    loaded = load_vectorizer(str(path))

    transformed = loaded.transform(["red car"])
    assert transformed.shape[1] == len(loaded.vocabulary_)
