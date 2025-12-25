from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import os
import csv
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading

# ===================== CONFIG =====================
BASE_URL = "https://qipedc.moet.gov.vn"

VIDEOS_DIR = "Dataset/Videos"
TEXT_DIR = "Dataset/Text"
CSV_PATH = os.path.join(TEXT_DIR, "label.csv")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

csv_lock = threading.Lock()
seen_videos = set()   # chống trùng khi crawl

# ===================== EDGE DRIVER =====================
def create_edge_driver():
    options = Options()
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service()  # Selenium tự quản lý EdgeDriver
    return webdriver.Edge(service=service, options=options)

# ===================== CSV =====================
def csv_init():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "VIDEO", "LABEL"])

def add_to_csv(video_name, label):
    with csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row_id = sum(1 for _ in open(CSV_PATH, encoding="utf-8"))
            writer.writerow([row_id, video_name, label])

# ===================== SCRAPE =====================
def scrape_current_page(driver, results: list):
    global seen_videos

    WebDriverWait(driver, 5).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR,
             "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) > a")
        )
    )

    items = driver.find_elements(
        By.CSS_SELECTOR,
        "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) > a"
    )

    for item in items:
        try:
            label = item.find_element(By.TAG_NAME, "p").text.strip()
            thumb = item.find_element(By.TAG_NAME, "img").get_attribute("src")

            video_id = thumb.replace(
                "https://qipedc.moet.gov.vn/thumbs/", ""
            ).replace(".png", "")

            video_url = f"{BASE_URL}/videos/{video_id}.mp4"

            if video_url in seen_videos:
                continue

            seen_videos.add(video_url)

            results.append({
                "label": label,
                "video_url": video_url
            })

        except:
            continue

# ===================== CRAWL =====================
def crawl_videos():
    print("CRAWLING VIDEOS (EDGE)")
    driver = create_edge_driver()
    videos = []

    try:
        driver.get(f"{BASE_URL}/dictionary")
        print("Connected to dictionary website")

        # Trang đầu
        scrape_current_page(driver, videos)

        # Trang 2–4
        for i in range(2, 5):
            btn_index = i if i == 2 else i + 1
            driver.find_element(
                By.CSS_SELECTOR, f"button:nth-of-type({btn_index})"
            ).click()
            scrape_current_page(driver, videos)

        # Trang 5–217 (luôn click button thứ 6)
        for _ in range(5, 218):
            driver.find_element(
                By.CSS_SELECTOR, "button:nth-of-type(6)"
            ).click()
            scrape_current_page(driver, videos)

        # Trang cuối
        for i in range(218, 220):
            btn_index = 6 if i == 218 else 7
            driver.find_element(
                By.CSS_SELECTOR, f"button:nth-of-type({btn_index})"
            ).click()
            scrape_current_page(driver, videos)

    finally:
        driver.quit()

    return videos

# ===================== DOWNLOAD =====================
def download_video(video):
    url = video["video_url"]
    label = video["label"]

    filename = os.path.basename(urlparse(url).path)
    output_path = os.path.join(VIDEOS_DIR, filename)

    if os.path.exists(output_path):
        return

    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=100
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        add_to_csv(filename, label)
        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Error {filename}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)

# ===================== MAIN =====================
def main():
    csv_init()

    videos = crawl_videos()
    print(f"Total unique videos: {len(videos)}")

    if not videos:
        print("No videos found")
        return

    print("START DOWNLOADING")

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(download_video, videos)

    print("DOWNLOAD COMPLETED")

if __name__ == "__main__":
    main()
