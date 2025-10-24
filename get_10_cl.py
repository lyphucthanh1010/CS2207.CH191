import pandas as pd
from pathlib import Path

# 1️⃣ Đường dẫn file gốc
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "diabetes_synthetic_full.csv"
OUTPUT_FILE = DATA_DIR / "diabetes_top10.csv"

# 2️⃣ Đọc file CSV
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"❌ Không tìm thấy file: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"📂 Đọc {len(df)} hàng và {len(df.columns)} cột từ {INPUT_FILE}")

# 3️⃣ Danh sách 10 cột quan trọng nhất
TOP10_COLS = [
    "a1c",
    "glucose",
    "has_obesity",
    "age",
    "has_exercise",
    "has_diet",
    "cholesterol",
    "pressure",
    "insulin",
    "weight"
]

# 4️⃣ Kiểm tra cột có tồn tại không
missing = [c for c in TOP10_COLS if c not in df.columns]
if missing:
    print(f"⚠️ Một số cột không có trong file: {missing}")
    TOP10_COLS = [c for c in TOP10_COLS if c in df.columns]

# 5️⃣ Lọc dữ liệu
df_top10 = df[TOP10_COLS]

# 6️⃣ Xuất ra file mới
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_top10.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Đã lọc và lưu 10 cột quan trọng nhất vào: {OUTPUT_FILE}")

# 7️⃣ Hiển thị thống kê cơ bản
print("\n📊 Mô tả thống kê của 10 cột:")
print(df_top10.describe())
