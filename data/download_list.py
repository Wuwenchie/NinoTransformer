import requests
from tqdm import tqdm

# 讀取檔案內的下載連結
with open("download_list.txt", "r") as file:
    urls = [line.strip() for line in file.readlines() if line.strip()]

for url in urls:
    filename = url.split("/")[-1]
    print(f"下載中: {filename}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename, total=total_size, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"下載完成: {filename}")
