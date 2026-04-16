import requests
from bs4 import BeautifulSoup
import time
import json
import string
import hashlib

BASE_URL = "https://www.apotheek.nl"
HEADERS = {
    "User-Agent": "MedicationRAGBot/1.0 (educational project)"
}
DELAY = 1.0


# -----------------------------
# Helpers
# -----------------------------
SECTION_DISPLAY = {
    "wat_is_het": "Wat is het",
    "waarvoor": "Waar is het voor",
    "bijwerkingen": "Bijwerkingen",
    "gebruik": "Hoe gebruik je het",
    "vergeten": "Vergeten dosis",
    "rijvaardigheid": "Rijvaardigheid en alcohol",
    "interacties": "Interacties met andere medicijnen",
    "zwangerschap": "Zwangerschap en borstvoeding",
    "lever_nier": "Bij lever- of nierproblemen",
    "stoppen": "Stoppen met het medicijn",
    "algemeen": "Algemene informatie",
    "overig": "Overige informatie",
}

SECTIONS_MAP = {
    "intro": "wat_is_het",
    "conditions": "waarvoor",
    "sideEffects": "bijwerkingen",
    "instructions": "gebruik",
    "forgotten": "vergeten",
    "forbidden": "rijvaardigheid",
    "interaction": "interacties",
    "pregnancy": "zwangerschap",
    "reducedKidneyLiverFunction": "lever_nier",
    "quitting": "stoppen",
    "information": "algemeen",
}


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


SECTION_TAGS = {
    "wat_is_het": ["description", "definition", "what-is-it"],
    "waarvoor": ["indications", "uses", "conditions-treated"],
    "bijwerkingen": ["side-effects", "adverse-reactions", "safety"],
    "gebruik": ["dosage", "instructions", "how-to-use"],
    "vergeten": ["missed-dose", "compliance"],
    "rijvaardigheid": ["driving", "alcohol", "lifestyle"],
    "interacties": ["interactions", "drug-interactions", "contraindications"],
    "zwangerschap": ["pregnancy", "breastfeeding", "fertility"],
    "lever_nier": ["organ-impairment", "kidney", "liver"],
    "stoppen": ["discontinuation", "withdrawal"],
    "algemeen": ["general", "availability", "market-info"],
}

SECTION_INTENT = {
    "wat_is_het": "description",
    "waarvoor": "usage",
    "bijwerkingen": "side_effects",
    "gebruik": "usage",
    "vergeten": "usage",
    "rijvaardigheid": "usage",
    "interacties": "side_effects",
    "zwangerschap": "side_effects",
    "lever_nier": "side_effects",
    "stoppen": "usage",
    "algemeen": "description",
}

CONTENT_KEYWORDS = {
    "cardiovascular": ["hart", "bloeddruk", "cholesterol", "ritme", "trombose", "vaat", "hartinfarct", "bijnier"],
    "pain": ["pijn", "pijnstiller", "hoofdpijn", "spierpijn"],
    "infection": ["infectie", "bacterie", "virus", "schimmel", "antibioticum", "ontsteking"],
    "mental_health": ["depressie", "angst", "slaap", "psych", "stemming", "epilepsie", "aanval"],
    "diabetes": ["diabetes", "bloedsuiker", "glucose", "insuline"],
    "hormonal": ["hormoon", "schildklier", "bijnier", "oestrogeen", "testosteron"],
    "respiratory": ["luchtweg", "astma", "COPD", "ademhaling", "neus", "long"],
    "digestive": ["maag", "darm", "misselijk", "braken", "diarree", "levert"],
    "skin": ["huid", "eczeem", "uitslag", "jeuk"],
    "immune": ["immuun", "allergie", "overgevoelig", "ontsteking"],
    "cancer": ["kanker", "tumor", "celdeling", "chemo"],
    "reproductive": ["zwanger", "borstvoeding", "vruchtbaar", "menstruatie", "overgang"],
    "children": ["kinderen", "baby", "jong", "geboorte"],
}


def extract_metadata_tags(content: str, section: str) -> list:
    tags = []
    
    if section in SECTION_TAGS:
        tags.extend(SECTION_TAGS[section])
    
    content_lower = content.lower()
    for category, keywords in CONTENT_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            tags.append(category)
    
    return list(set(tags))


# -----------------------------
# Step 1: Get medication links
# -----------------------------
def get_medication_links(limit=10):
    letters = list(string.ascii_lowercase) + ["0-9"]
    links = set()

    for letter in letters:
        url = f"{BASE_URL}/medicijnen?letter={letter}"
        print(f"Fetching index: {url}")

        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.select("a[href]"):
                href = a.get("href")

                if not href:
                    continue

                # Match real medication pages
                if (
                    "/medicijnen/" in href
                    # and "?" not in href
                    and "bij-kinderen" not in href
                    and "kindertekst" not in href
                    ):
                    full_url = href if href.startswith("http") else BASE_URL + href

                    # Avoid navigation links and duplicates
                    # Keep only real drug pages (2-3 slashes in relative path)
                    slash_count = full_url.count("/")
                    if slash_count >= 3 and slash_count <= 5:
                        links.add(full_url)

            time.sleep(DELAY)

            if len(links) >= limit:
                break

        except Exception as e:
            print("Error:", e)

    return list(links)[:limit]


# -----------------------------
# Step 2: Parse medication page
# -----------------------------
def parse_medication_page(url):
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    next_data = soup.find('script', {'id': '__NEXT_DATA__'})
    if not next_data:
        return []

    data = json.loads(next_data.string)
    medicine = data.get('props', {}).get('pageProps', {}).get('medicine', {})

    title = medicine.get('title', '')
    if not title:
        return []

    sections = {}

    for section_key, section_name in SECTIONS_MAP.items():
        content = medicine.get(section_key, '')
        if not content or len(content) < 50:
            continue

        content_text = BeautifulSoup(content, 'html.parser').get_text(' ', strip=True)

        if len(content_text) < 50:
            continue

        sections[section_name] = {
            "display": SECTION_DISPLAY.get(section_name, section_name),
            "content": content_text
        }

    chunks = []
    for section_name, section_data in sections.items():
        content = section_data["content"]
        chunks.append({
            "title": title,
            "section": section_name,
            "section_display": section_data["display"],
            "content": content,
            "intent": SECTION_INTENT.get(section_name, "general"),
            "search_text": f"{title}. {section_data['display']}. {content}",
            "url": url,
            "source": "apotheek.nl",
            "type": "drug",
            "language": "nl",
            "tags": extract_metadata_tags(content, section_name),
        })
    return chunks


# -----------------------------
# Step 3: Main
# -----------------------------
def scrape_dataset(limit=10000, batch_save=50):
    links = get_medication_links(limit=limit)

    print(f"\nCollected {len(links)} medication links\n")

    dataset = []
    scraped_urls = set()

    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Scraping: {link}")

        try:
            chunks = parse_medication_page(link)
            dataset.extend(chunks)
            scraped_urls.add(link)
        except Exception as e:
            print(f"Error scraping {link}: {e}")

        if (i + 1) % batch_save == 0:
            with open("apotheek_dataset_partial.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {len(dataset)} chunks so far (batch save)\n")

        time.sleep(DELAY)

    return dataset


if __name__ == "__main__":
    data = scrape_dataset()

    print(f"\nTotal chunks collected: {len(data)}")

    with open("apotheek_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Saved to apotheek_dataset.json")