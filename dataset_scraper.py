import requests
from bs4 import BeautifulSoup
import time
import json
import hashlib

BASE_URL = "https://www.apotheek.nl"
START_URL = f"{BASE_URL}/medicijnen?letter=a"

HEADERS = {
    "User-Agent": "MedicationRAGBot/1.0 (educational project)"
}

DELAY = 1.5  # polite delay


# -----------------------------
# Helpers
# -----------------------------
def normalize_section(name: str) -> str:
    name = name.lower()

    if "bijwerking" in name:
        return "bijwerkingen"
    if "gebruik" in name or "hoe gebruik" in name:
        return "gebruik"
    if "waarvoor" in name:
        return "indicatie"
    if "wanneer" in name:
        return "waarschuwingen"

    return "overig"


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# -----------------------------
# Step 1: Get medication links
# -----------------------------
def get_medication_links(limit=10):
    res = requests.get(START_URL, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    links = set()

    for a in soup.select("a"):
        href = a.get("href")
        if href and "/medicijnen/" in href:
            full_url = href if href.startswith("http") else BASE_URL + href
            links.add(full_url)

    links = list(links)

    # Basic filtering (avoid index/self links)
    links = [l for l in links if l.count("/") > 4]

    return links[:limit]


# -----------------------------
# Step 2: Parse medication page
# -----------------------------
def parse_medication_page(url):
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    title_tag = soup.find("h1")
    if not title_tag:
        return []

    title = title_tag.get_text(strip=True)

    chunks = []
    seen_hashes = set()

    # Sections = h2 / h3 blocks
    for section in soup.select("h2, h3"):
        section_title = section.get_text(strip=True)

        content_parts = []

        for sib in section.find_next_siblings():
            if sib.name in ["h2", "h3"]:
                break
            text = sib.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        content = " ".join(content_parts).strip()

        # Filter noise
        if len(content) < 80:
            continue

        content_hash = hash_text(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        chunks.append({
            "title": title,
            "url": url,
            "source": "apotheek.nl",
            "section_raw": section_title,
            "section": normalize_section(section_title),
            "content": content,
            "type": "drug",
            "language": "nl"
        })

    return chunks


# -----------------------------
# Step 3: Main scrape function
# -----------------------------
def scrape_dataset(limit=10):
    links = get_medication_links(limit=limit)

    print(f"Found {len(links)} medication links")

    dataset = []

    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Scraping: {link}")

        try:
            chunks = parse_medication_page(link)
            dataset.extend(chunks)
        except Exception as e:
            print(f"Error scraping {link}: {e}")

        time.sleep(DELAY)

    return dataset


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    data = scrape_dataset(limit=10)

    print(f"\nTotal chunks collected: {len(data)}")

    with open("apotheek_dataset_10.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Saved to apotheek_dataset_10.json")