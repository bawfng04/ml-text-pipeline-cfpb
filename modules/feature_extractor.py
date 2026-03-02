"""
Module trích xuất đặc trưng văn bản.
Hỗ trợ: BoW, TF-IDF, n-gram (truyền thống) và Word2Vec, GloVe, BERT (hiện đại).
"""

import os
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ─────────────────────────────────────────────
# Phần 1: Traditional features
# ─────────────────────────────────────────────


def build_bow(
    train_texts,
    test_texts,
    max_features: int = 50_000,
    min_df: int = 2,
    binary: bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix, CountVectorizer]:
    """
    Bag of Words.
    binary=True thì chỉ đánh dấu có/không xuất hiện từ.
    """
    vec = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        binary=binary,
        dtype=np.float32,
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec


def build_tfidf(
    train_texts,
    test_texts,
    max_features: int = 50_000,
    min_df: int = 2,
    ngram_range: tuple = (1, 1),
    sublinear_tf: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """
    TF-IDF vectorizer. sublinear_tf=True thì dùng log(TF) - thường tốt hơn với text dài.
    ngram_range=(1,2) để thêm bigrams.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec


def build_ngram(
    train_texts,
    test_texts,
    max_features: int = 50_000,
    ngram_range: tuple = (2, 2),
    use_tfidf: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Wrapper tiện lợi cho n-gram đơn thuần."""
    if use_tfidf:
        X_train, X_test, vec = build_tfidf(
            train_texts,
            test_texts,
            max_features=max_features,
            ngram_range=ngram_range,
        )
    else:
        X_train, X_test, vec = build_bow(
            train_texts,
            test_texts,
            max_features=max_features,
        )
    return X_train, X_test, vec


# ─────────────────────────────────────────────
# Phần 2: Word2Vec embeddings
# ─────────────────────────────────────────────


def train_word2vec(
    tokenized_texts: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
):
    """
    Huấn luyện Word2Vec từ đầu trên corpus.
    tokenized_texts: list of list of tokens (đã preprocess).
    """
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    return model


def text_to_avg_vector(
    tokens: list[str], w2v_model, vector_size: int = 100
) -> np.ndarray:
    """Biểu diễn một văn bản bằng trung bình vector của các từ."""
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def build_w2v_features(
    tokenized_train: list[list[str]],
    tokenized_test: list[list[str]],
    vector_size: int = 100,
    w2v_model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Xây dựng ma trận features từ Word2Vec.
    Nếu w2v_model=None thì tự train, ngược lại dùng model truyền vào (pretrained).
    """
    if w2v_model is None:
        w2v_model = train_word2vec(tokenized_train, vector_size=vector_size)

    X_train = np.vstack(
        [text_to_avg_vector(t, w2v_model, vector_size) for t in tokenized_train]
    )
    X_test = np.vstack(
        [text_to_avg_vector(t, w2v_model, vector_size) for t in tokenized_test]
    )
    return X_train, X_test, w2v_model


# ─────────────────────────────────────────────
# Phần 3: BERT embeddings (sentence-transformers)
# ─────────────────────────────────────────────


def build_bert_features(
    train_texts: list[str],
    test_texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trích xuất sentence embeddings dùng sentence-transformers.
    Mặc định dùng 'all-MiniLM-L6-v2' vì nhỏ, nhanh, vẫn tốt.
    Truncate về max_length=128 vì complaint text thường dài.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length

    print(f"  Encoding {len(train_texts)} train samples...")
    X_train = model.encode(
        train_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"  Encoding {len(test_texts)} test samples...")
    X_test = model.encode(
        test_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return X_train.astype(np.float32), X_test.astype(np.float32)


# ─────────────────────────────────────────────
# Phần 4: Lưu / load features
# ─────────────────────────────────────────────


def save_features(
    features_dir: str,
    name: str,
    X_train,
    X_test,
    y_train=None,
    y_test=None,
):
    """
    Lưu features ra file .npz (sparse) hoặc .npy (dense).
    Cũng lưu luôn labels nếu truyền vào.
    """
    out = Path(features_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sp.issparse(X_train):
        sp.save_npz(str(out / f"{name}_train.npz"), X_train.astype(np.float32))
        sp.save_npz(str(out / f"{name}_test.npz"), X_test.astype(np.float32))
    else:
        np.save(str(out / f"{name}_train.npy"), X_train.astype(np.float32))
        np.save(str(out / f"{name}_test.npy"), X_test.astype(np.float32))

    if y_train is not None:
        np.save(str(out / "y_train.npy"), y_train)
    if y_test is not None:
        np.save(str(out / "y_test.npy"), y_test)

    print(f"  Saved: {name} -> {out}")


def load_features(features_dir: str, name: str, sparse: bool = False):
    """Load features đã lưu."""
    out = Path(features_dir)
    if sparse:
        X_train = sp.load_npz(str(out / f"{name}_train.npz"))
        X_test = sp.load_npz(str(out / f"{name}_test.npz"))
    else:
        X_train = np.load(str(out / f"{name}_train.npy"))
        X_test = np.load(str(out / f"{name}_test.npy"))

    y_train, y_test = None, None
    if (out / "y_train.npy").exists():
        y_train = np.load(str(out / "y_train.npy"))
    if (out / "y_test.npy").exists():
        y_test = np.load(str(out / "y_test.npy"))

    return X_train, X_test, y_train, y_test


def save_vectorizer(vec, path: str):
    with open(path, "wb") as f:
        pickle.dump(vec, f)


def load_vectorizer(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
