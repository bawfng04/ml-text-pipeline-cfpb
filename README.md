# ML Text Pipeline - CFPB Consumer Complaints

**Môn học:** Học Máy (Machine Learning)
**Học kỳ:** HK2, Năm học 2025-2026

---

## Thông tin nhóm

| Họ tên             | MSSV   | Email   |
| ------------------ | ------ | ------- |
| [Tên thành viên 1] | [MSSV] | [email] |
| [Tên thành viên 2] | [MSSV] | [email] |

**Giảng viên hướng dẫn (GVHD):** [Tên GV]

---

## Mục tiêu

Xây dựng pipeline học máy để phân loại khiếu nại người tiêu dùng tài chính (CFPB Consumer Complaints) theo danh mục sản phẩm. Pipeline bao gồm:

- **EDA:** Phân tích phân phối nhãn, độ dài văn bản, tần suất từ, xu hướng theo thời gian
- **Tiền xử lý:** Làm sạch text, loại stopwords, lemmatization (song song với joblib)
- **Trích xuất đặc trưng truyền thống:** Bag-of-Words, TF-IDF unigram, TF-IDF bigram
- **Trích xuất đặc trưng hiện đại:** Word2Vec (tự huấn luyện), BERT sentence embeddings (`all-MiniLM-L6-v2`)
- **Huấn luyện & đánh giá:** Naive Bayes (Complement + Multinomial), Logistic Regression, LinearSVC, **Random Forest** — ~18 tổ hợp
- **Hyperparameter tuning:** 3-fold CV sweep C ∈ [0.01, 10] cho LR và SVM
- **So sánh toàn diện:** Bảng kết quả, grouped bar chart, confusion matrix, per-class F1, trade-off chart

---

## Dataset

**Nguồn:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/#get-the-data)
**Download URL:** `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
**Kích thước gốc:** ~7.8 GB (13.7M rows)
**Sample sử dụng:** ~14,000 rows (7 classes × 2,000 mẫu, stratified)

Dataset được download tự động trong notebook — **không cần mount Drive**.

---

## Cấu trúc thư mục

```
ml-text-pipeline-cfpb/
├── data/
│   ├── complaints.csv          # Dataset gốc (download tự động)
│   └── df_clean.parquet        # Cache sau preprocessing (tạo tự động)
├── docs/
│   └── assignment-reqs.txt     # Yêu cầu đề bài
├── features/                   # Features đã trích xuất (tạo tự động)
│   ├── bow_train.npz / bow_test.npz
│   ├── tfidf_uni_train.npz / tfidf_uni_test.npz
│   ├── tfidf_bigram_train.npz / tfidf_bigram_test.npz
│   ├── w2v_train.npy / w2v_test.npy
│   ├── bert_train.npy / bert_test.npy
│   ├── y_train.npy / y_test.npy
│   ├── word2vec.model
│   └── vec_*.pkl               # Vectorizers (BoW, TF-IDF)
├── modules/
│   ├── __init__.py
│   ├── text_preprocessor.py    # Làm sạch text, tokenize, lemmatize
│   ├── feature_extractor.py    # BoW, TF-IDF, Word2Vec, BERT
│   └── trainer.py              # Train, evaluate, summarize
├── notebooks/
│   └── main_pipeline.ipynb     # Notebook chính (EDA → kết quả)
├── reports/
│   └── experiment_results.csv  # Bảng kết quả tổng hợp (tạo tự động)
├── requirements.txt
└── README.md
```

---

## Hướng dẫn chạy

### Trên Google Colab (khuyến nghị)

1. Mở link Colab: **[link]**
2. Chọn **Runtime → Run all**
   - Notebook tự clone repo, cài thư viện, download dataset và chạy toàn bộ pipeline
   - Không cần thao tác thêm

### Chạy local

```bash
# Clone repo
git clone https://github.com/bawfng04/ml-text-pipeline-cfpb.git
cd ml-text-pipeline-cfpb

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Cài dependencies
pip install -r requirements.txt

# Chạy notebook
jupyter notebook notebooks/main_pipeline.ipynb
```

**Yêu cầu:** Python 3.10+, RAM ≥ 16 GB (khuyến nghị 32 GB)

---

## Thư viện chính

| Thư viện              | Phiên bản tối thiểu | Mục đích                          |
| --------------------- | ------------------- | --------------------------------- |
| scikit-learn          | ≥ 1.3               | Vectorizer, models, metrics       |
| nltk                  | ≥ 3.8               | Tokenize, stopwords, lemmatize    |
| gensim                | ≥ 4.3               | Word2Vec                          |
| sentence-transformers | ≥ 2.6               | BERT sentence embeddings          |
| torch                 | ≥ 2.0               | Backend cho sentence-transformers |
| pandas / numpy        | ≥ 2.0 / ≥ 1.24      | Data manipulation                 |
| matplotlib / seaborn  | ≥ 3.7 / ≥ 0.12      | Visualization                     |
| wordcloud             | ≥ 1.9               | Word Cloud EDA                    |
| pyarrow               | ≥ 12.0              | Parquet cache I/O                 |
| joblib                | ≥ 1.3               | Song song hóa preprocessing       |

---

## Kết quả tóm tắt

Xem chi tiết trong `reports/experiment_results.csv` và notebook Section 9.

**Best model: TF-IDF Bigram + LinearSVC (C=0.5) — F1 Weighted = 0.8631**

| Feature               | Model                       | F1 Weighted |
| --------------------- | --------------------------- | ----------- |
| TF-IDF Bigram (C=0.5) | LinearSVC (tuned)           | **0.8631**  |
| TF-IDF Bigram (C=5.0) | Logistic Regression (tuned) | 0.8597      |
| TF-IDF Unigram        | Logistic Regression         | 0.8590      |
| TF-IDF Bigram         | LinearSVC                   | 0.8599      |
| TF-IDF Bigram         | Logistic Regression         | 0.8573      |
| BERT (MiniLM)         | LinearSVC                   | 0.8241      |
| Word2Vec              | LinearSVC                   | 0.8244      |
| TF-IDF Bigram         | Random Forest               | 0.8270      |
| BoW                   | Complement NB               | 0.8010      |

**Nhận xét nhanh:**

- TF-IDF Bigram + LinearSVC (tuned C=0.5) là best overall với F1 = 0.8631
- BERT (0.8241) bất ngờ **không vượt** TF-IDF — do sample nhỏ (13,981 rows) và chỉ dùng làm feature extractor (không fine-tune)
- Word2Vec (0.8244) tương đương BERT nhưng train nhanh hơn nhiều
- Random Forest (0.827) kém LinearSVC ~3.3% trên sparse features, đúng như dự đoán
- Không có class nào có F1 < 0.7; khó nhất là Credit card (0.806)

---

## Links

- **Notebook Colab:** [link]
- **Báo cáo PDF:** `reports/report.pdf`
- **GitHub:** https://github.com/bawfng04/ml-text-pipeline-cfpb
