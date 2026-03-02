"""
Module tiền xử lý văn bản cho CFPB Consumer Complaints dataset.
"""

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Tải các resource NLTK cần thiết (chỉ lần đầu)
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "punkt_tab", "omw-1.4"]
    for r in resources:
        nltk.download(r, quiet=True)


# Pattern đặc trưng của CFPB dataset: tên/số bị redact thành XX...
_REDACT_PATTERN = re.compile(r"\bx{2,}\b", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\w+")
_DOLLAR_PATTERN = re.compile(r"\$[\d,]+\.?\d*")
_NUMBER_PATTERN = re.compile(r"\b\d+\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, remove_numbers: bool = True) -> str:
    """
    Làm sạch text thô: bỏ URL, email, ký tự đặc biệt, số, redacted tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.lower()

    # Bỏ URL và email
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)

    # Thay số tiền bằng token đặc biệt (giữ ngữ nghĩa "có đề cập đến tiền")
    text = _DOLLAR_PATTERN.sub(" moneytok ", text)

    # Bỏ token bị redact (XX, XXX, XXXX...) - đặc thù của dataset này
    text = _REDACT_PATTERN.sub(" ", text)

    if remove_numbers:
        text = _NUMBER_PATTERN.sub(" ", text)

    # Bỏ dấu câu
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Chuẩn hóa whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text thành list các tokens."""
    return word_tokenize(text) if text else []


def remove_stopwords(
    tokens: list[str], extra_stopwords: set | None = None
) -> list[str]:
    """Lọc stopwords tiếng Anh. Có thể truyền thêm custom stopwords."""
    sw = set(stopwords.words("english"))
    if extra_stopwords:
        sw |= extra_stopwords
    return [t for t in tokens if t not in sw and len(t) > 1]


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    remove_numbers: bool = True,
    extra_stopwords: set | None = None,
) -> str:
    """
    Pipeline tiền xử lý đầy đủ: clean -> tokenize -> remove stopwords -> stem/lemma.

    Trả về chuỗi đã xử lý (dạng string, không phải list) để dùng với sklearn vectorizers.
    """
    text = clean_text(text, remove_numbers=remove_numbers)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, extra_stopwords=extra_stopwords)

    if use_stemming:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def preprocess_batch(
    texts,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    remove_numbers: bool = True,
    extra_stopwords: set | None = None,
    n_jobs: int = -1,
) -> list[str]:
    """
    Xử lý một mảng/series texts, hỗ trợ song song bằng joblib.
    Dùng n_jobs=-1 để tận dụng hết CPU cores.
    """
    from joblib import Parallel, delayed

    def _process(t):
        return preprocess_text(
            t,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
            remove_numbers=remove_numbers,
            extra_stopwords=extra_stopwords,
        )

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(_process)(t) for t in texts)
