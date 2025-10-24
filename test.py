"""
generate_synthetic_10k_fixed.py
Phiên bản ổn định không phụ thuộc vào dữ liệu NLTK.
- Không yêu cầu sdv hoặc nltk_data
- Tự động fallback nếu thiếu thư viện
- Sinh 10,000 hàng synthetic từ dữ liệu crawl về tiểu đường
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ======================
# 1️⃣ Cấu hình ban đầu
# ======================
DATA_DIR = Path("data")
INPUT_TEXT = DATA_DIR / "diabetes_text.csv"
INPUT_FEATURES = DATA_DIR / "diabetes_text_features.csv"
OUT_NUMERIC = DATA_DIR / "diabetes_synthetic_numeric.csv"
OUT_TEXT = DATA_DIR / "diabetes_synthetic_text.csv"
OUT_FULL = DATA_DIR / "diabetes_synthetic_full.csv"

N_SYNTH = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ======================
# 2️⃣ Đọc dữ liệu gốc
# ======================
if not INPUT_TEXT.exists() or not INPUT_FEATURES.exists():
    raise FileNotFoundError("❌ Thiếu file input: hãy đảm bảo có diabetes_text.csv và diabetes_text_features.csv trong thư mục data/")

df_feats = pd.read_csv(INPUT_FEATURES)
df_text = pd.read_csv(INPUT_TEXT)

if "title" in df_feats.columns and "title" in df_text.columns:
    df = pd.merge(df_feats, df_text[["title", "content"]], on="title", how="left")
else:
    df = df_feats.copy()
    df["content"] = df_text["content"] if "content" in df_text.columns else ""

exclude = {"title", "content"}
numeric_cols = [c for c in df.columns if c not in exclude]
X_real = df[numeric_cols].fillna(0)

print(f"✅ Đọc {len(df)} hàng gốc, {len(numeric_cols)} cột numeric.")

# ======================
# 3️⃣ Sinh dữ liệu numeric 10,000 hàng
# ======================
print("🧮 Đang sinh dữ liệu numeric (bootstrap + jitter)...")
rows = []
for i in range(N_SYNTH):
    idx = np.random.randint(0, len(X_real))
    row = X_real.iloc[idx].astype(float).values.copy()
    noise = np.random.normal(0, 0.01, len(row))
    row = row + noise * (np.abs(row) + 1e-6)
    rows.append(row)
synth = pd.DataFrame(rows, columns=X_real.columns)
synth.to_csv(OUT_NUMERIC, index=False)
print(f"💾 Lưu: {OUT_NUMERIC} ({len(synth)} hàng)")

# ======================
# 4️⃣ Tạo nội dung văn bản synthetic
# ======================
print("🧠 Đang tạo nội dung văn bản synthetic...")

contents = df["content"].fillna("").astype(str).tolist()
titles = df["title"].fillna("").astype(str).tolist()
n_orig = len(contents)

# fallback tokenizer không cần nltk
def sent_tokenize(text):
    return [s.strip() for s in text.split(".") if len(s.strip()) > 0]

def synonym_replace(text):
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.05 and len(words[i]) > 4:
            words[i] = words[i][::-1]  # đảo chữ làm "nhiễu" nhẹ
    return " ".join(words)

def sentence_shuffle(text):
    sents = sent_tokenize(text)
    if len(sents) > 1 and random.random() < 0.4:
        random.shuffle(sents)
    return ". ".join(sents) + "."

# khởi tạo TF-IDF để chọn bài gốc gần nhất
vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
tfidf = vectorizer.fit_transform(df["content"].fillna("").astype(str))
nn = NearestNeighbors(n_neighbors=1).fit(tfidf)

# tạo nội dung synthetic
synth_text = []
for i in range(N_SYNTH):
    base_idx = np.random.randint(0, n_orig)
    base_title = titles[base_idx]
    base_content = contents[base_idx]

    new_text = sentence_shuffle(base_content)
    if random.random() < 0.6:
        new_text = synonym_replace(new_text)
    if random.random() < 0.05:
        other_idx = np.random.randint(0, n_orig)
        new_text += "\n\n" + contents[other_idx][:150]

    synth_text.append({
        "title": base_title + " [synthetic]" if random.random() < 0.6 else base_title,
        "content": new_text
    })

df_text_syn = pd.DataFrame(synth_text)
df_text_syn.to_csv(OUT_TEXT, index=False)
print(f"💾 Lưu: {OUT_TEXT} ({len(df_text_syn)} hàng)")

# ======================
# 5️⃣ Gộp full dataset
# ======================
df_full = pd.concat([synth.reset_index(drop=True), df_text_syn.reset_index(drop=True)], axis=1)
df_full.to_csv(OUT_FULL, index=False)
print(f"✅ Đã tạo full dataset: {OUT_FULL}")
print("🎉 Hoàn tất! Tổng cộng 10.000 hàng synthetic.")
