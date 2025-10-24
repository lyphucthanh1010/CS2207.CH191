import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = "https://www.healthline.com/health/diabetes"
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# Gửi request đến trang chính
response = requests.get(base_url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Lấy danh sách link liên quan đến 'diabetes'
links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "/health/" in href and "diabetes" in href and href.startswith("https"):
        links.append(href)
links = list(set(links))  # loại trùng lặp

print(f"🔗 Tìm thấy {len(links)} bài viết.")

# Crawl từng bài
data = []
for link in links[:20]:  # crawl thử 20 bài
    try:
        time.sleep(1)
        res = requests.get(link, headers=headers, timeout=10)
        art = BeautifulSoup(res.text, "html.parser")
        title = art.find("h1").get_text(strip=True) if art.find("h1") else ""
        content = " ".join(p.get_text(strip=True) for p in art.find_all("p"))
        if len(content) > 200:
            data.append({"url": link, "title": title, "content": content})
            print(f"✅ {title[:60]}...")
    except Exception as e:
        print(f"⚠️ Lỗi {link}: {e}")

# Lưu thành dataset mới
df = pd.DataFrame(data)
df.to_csv("diabetes_text.csv", index=False)
print(f"💾 Đã lưu {len(df)} bài viết tại diabetes_text.csv")
