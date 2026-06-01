import requests
from bs4 import BeautifulSoup
import time
import json
import string
import hashlib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

HEADERS = {
    "User-Agent": "MedicationRAGBot/1.0 (educational project)"
}
DELAY = 0.5


# ==============================
# Shared helpers
# ==============================

SECTION_TAGS = {
    "wat_is_het":    ["description", "definition", "what-is-it"],
    "waarvoor":      ["indications", "uses", "conditions-treated"],
    "bijwerkingen":  ["side-effects", "adverse-reactions", "safety"],
    "gebruik":       ["dosage", "instructions", "how-to-use"],
    "vergeten":      ["missed-dose", "compliance"],
    "rijvaardigheid":["driving", "alcohol", "lifestyle"],
    "interacties":   ["interactions", "drug-interactions", "contraindications"],
    "zwangerschap":  ["pregnancy", "breastfeeding", "fertility"],
    "lever_nier":    ["organ-impairment", "kidney", "liver"],
    "stoppen":       ["discontinuation", "withdrawal"],
    "algemeen":      ["general", "availability", "market-info"],
    "overdosering":  ["overdose", "toxicology", "emergency"],
    "eigenschappen": ["pharmacology", "kinetics", "mechanism"],
    "contra":        ["contraindications", "warnings", "safety"],
}

SECTION_INTENT = {
    "wat_is_het":    "description",
    "waarvoor":      "usage",
    "bijwerkingen":  "side_effects",
    "gebruik":       "usage",
    "vergeten":      "usage",
    "rijvaardigheid":"usage",
    "interacties":   "side_effects",
    "zwangerschap":  "side_effects",
    "lever_nier":    "side_effects",
    "stoppen":       "usage",
    "algemeen":      "description",
    "overdosering":  "side_effects",
    "eigenschappen": "description",
    "contra":        "side_effects",
}

CONTENT_KEYWORDS = {
    "cardiovascular": ["hart", "bloeddruk", "cholesterol", "ritme", "trombose", "vaat", "hartinfarct", "bijnier"],
    "pain":           ["pijn", "pijnstiller", "hoofdpijn", "spierpijn"],
    "infection":      ["infectie", "bacterie", "virus", "schimmel", "antibioticum", "ontsteking"],
    "mental_health":  ["depressie", "angst", "slaap", "psych", "stemming", "epilepsie", "aanval"],
    "diabetes":       ["diabetes", "bloedsuiker", "glucose", "insuline"],
    "hormonal":       ["hormoon", "schildklier", "bijnier", "oestrogeen", "testosteron"],
    "respiratory":    ["luchtweg", "astma", "COPD", "ademhaling", "neus", "long"],
    "digestive":      ["maag", "darm", "misselijk", "braken", "diarree", "levert"],
    "skin":           ["huid", "eczeem", "uitslag", "jeuk"],
    "immune":         ["immuun", "allergie", "overgevoelig", "ontsteking"],
    "cancer":         ["kanker", "tumor", "celdeling", "chemo"],
    "reproductive":   ["zwanger", "borstvoeding", "vruchtbaar", "menstruatie", "overgang"],
    "children":       ["kinderen", "baby", "jong", "geboorte"],
}


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_metadata_tags(content: str, section: str) -> list:
    tags = list(SECTION_TAGS.get(section, []))
    content_lower = content.lower()
    for category, keywords in CONTENT_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            tags.append(category)
    return list(set(tags))


def make_chunk(title, section, section_display, content, intent, url, source):
    return {
        "title":          title,
        "section":        section,
        "section_display": section_display,
        "content":        content,
        "intent":         intent,
        "search_text":    f"{title}. {section_display}. {content}",
        "url":            url,
        "source":         source,
        "type":           "drug",
        "language":       "nl",
        "tags":           extract_metadata_tags(content, section),
    }


# ==============================
# SOURCE 1 — apotheek.nl
# ==============================

APOTHEEK_BASE = "https://www.apotheek.nl"

APOTHEEK_SECTION_DISPLAY = {
    "wat_is_het":    "Wat is het",
    "waarvoor":      "Waar is het voor",
    "bijwerkingen":  "Bijwerkingen",
    "gebruik":       "Hoe gebruik je het",
    "vergeten":      "Vergeten dosis",
    "rijvaardigheid":"Rijvaardigheid en alcohol",
    "interacties":   "Interacties met andere medicijnen",
    "zwangerschap":  "Zwangerschap en borstvoeding",
    "lever_nier":    "Bij lever- of nierproblemen",
    "stoppen":       "Stoppen met het medicijn",
    "algemeen":      "Algemene informatie",
    "overig":        "Overige informatie",
}

APOTHEEK_SECTIONS_MAP = {
    "intro":                    "wat_is_het",
    "conditions":               "waarvoor",
    "sideEffects":              "bijwerkingen",
    "instructions":             "gebruik",
    "forgotten":                "vergeten",
    "forbidden":                "rijvaardigheid",
    "interaction":              "interacties",
    "pregnancy":                "zwangerschap",
    "reducedKidneyLiverFunction":"lever_nier",
    "quitting":                 "stoppen",
    "information":              "algemeen",
}


def get_apotheek_links(limit=10000):
    letters = list(string.ascii_lowercase) + ["0-9"]
    links = set()

    for letter in letters:
        url = f"{APOTHEEK_BASE}/medicijnen?letter={letter}"
        print(f"[apotheek.nl] Fetching index: {url}")

        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href or "/medicijnen/" not in href:
                    continue
                full_url = href if href.startswith("http") else APOTHEEK_BASE + href
                slash_count = full_url.count("/")
                if 3 <= slash_count <= 5:
                    links.add(full_url)

            time.sleep(DELAY)
            if len(links) >= limit:
                break

        except Exception as e:
            print(f"[apotheek.nl] Error: {e}")

    return list(links)[:limit]


def parse_apotheek_page(url):
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    next_data = soup.find("script", {"id": "__NEXT_DATA__"})
    if not next_data:
        return []

    data = json.loads(next_data.string)
    medicine = data.get("props", {}).get("pageProps", {}).get("medicine", {})
    title = medicine.get("title", "")
    if not title:
        return []

    chunks = []
    for api_key, section_name in APOTHEEK_SECTIONS_MAP.items():
        content = medicine.get(api_key, "")
        if not content or len(content) < 50:
            continue
        content_text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
        if len(content_text) < 50:
            continue
        chunks.append(make_chunk(
            title=title,
            section=section_name,
            section_display=APOTHEEK_SECTION_DISPLAY.get(section_name, section_name),
            content=content_text,
            intent=SECTION_INTENT.get(section_name, "general"),
            url=url,
            source="apotheek.nl",
        ))
    return chunks


def scrape_apotheek(limit=10000, batch_save=50, dataset=None, partial_path=None):
    if dataset is None:
        dataset = []
    links = get_apotheek_links(limit=limit)
    print(f"\n[apotheek.nl] Collected {len(links)} links\n")

    for i, link in enumerate(links):
        print(f"[apotheek.nl] [{i+1}/{len(links)}] {link}")
        try:
            chunks = parse_apotheek_page(link)
            dataset.extend(chunks)
        except Exception as e:
            print(f"[apotheek.nl] Error: {e}")

        if partial_path and (i + 1) % batch_save == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {len(dataset)} chunks (batch)\n")

        time.sleep(DELAY)

    return dataset


# ==============================
# SOURCE 2 — farmacotherapeutischkompas.nl
# ==============================

FK_BASE = "https://www.farmacotherapeutischkompas.nl"

# FK section headings → internal section key
FK_SECTION_MAP = {
    "advies":                    ("waarvoor",      "Advies"),
    "indicaties":                ("waarvoor",      "Indicaties"),
    "doseringen":                ("gebruik",       "Doseringen"),
    "bijwerkingen":              ("bijwerkingen",  "Bijwerkingen"),
    "interacties":               ("interacties",   "Interacties"),
    "zwangerschap":              ("zwangerschap",  "Zwangerschap"),
    "lactatie":                  ("zwangerschap",  "Lactatie"),
    "contra-indicaties":         ("contra",        "Contra-indicaties"),
    "waarschuwingen en voorzorgen": ("contra",     "Waarschuwingen en voorzorgen"),
    "overdosering":              ("overdosering",  "Overdosering"),
    "eigenschappen":             ("eigenschappen", "Eigenschappen"),
    "samenstelling":             ("algemeen",      "Samenstelling"),
}


def get_fk_links(limit=10000):
    """
    FK drug pages follow the pattern:
      /bladeren/preparaatteksten/{first_letter}/{slug}
    Each group page contains a <div id="medicine-listing"> with links to drug pages.
    We scrape the static group index, then visit each group page to harvest links
    from div#medicine-listing.
    """
    links = set()
    group_index_url = f"{FK_BASE}/bladeren/preparaatteksten/groep"

    print(f"[FK] Fetching group index: {group_index_url}")
    try:
        res = requests.get(group_index_url, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        group_hrefs = set()
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if "/bladeren/preparaatteksten/groep/" in href and "/groep" not in href.strip("/").split("/")[-1]:
                full = href if href.startswith("http") else FK_BASE + href
                full = full.split("#")[0]
                group_hrefs.add(full)

        print(f"[FK] Found {len(group_hrefs)} drug groups, now fetching each…")

        for group_url in sorted(group_hrefs):
            if len(links) >= limit:
                break
            try:
                gres = requests.get(group_url, headers=HEADERS)
                if gres.status_code != 200:
                    continue
                gsoup = BeautifulSoup(gres.text, "html.parser")
                # Drug links are inside <div id="medicine-listing">
                listing = gsoup.find(id="medicine-listing")
                if listing:
                    for a in listing.find_all("a", href=True):
                        href = a["href"]
                        full = href if href.startswith("http") else FK_BASE + href
                        full = full.split("#")[0].split("?")[0]
                        links.add(full)
                time.sleep(DELAY)
            except Exception as e:
                print(f"[FK] Error fetching group {group_url}: {e}")

    except Exception as e:
        print(f"[FK] Error fetching group index: {e}")

    return list(links)[:limit]


def parse_fk_page(url):
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # Title: the last <h1> on the page (first one is "Farmacotherapeutisch Kompas")
    h1_tags = soup.find_all("h1")
    title = h1_tags[-1].get_text(" ", strip=True) if h1_tags else ""
    if not title:
        return []

    chunks = []

    # FK pages use <h2> headings to separate sections.
    # We walk the DOM: for each h2 collect all sibling content until the next h2.
    for h2 in soup.find_all("h2"):
        heading_text = h2.get_text(" ", strip=True).lower()

        # Find matching section key
        matched_key = None
        for key in FK_SECTION_MAP:
            if key in heading_text:
                matched_key = key
                break
        if matched_key is None:
            continue

        section_name, section_display = FK_SECTION_MAP[matched_key]

        # Collect text from siblings until next h2
        content_parts = []
        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":
                break
            text = sibling.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        content = " ".join(content_parts).strip()
        if len(content) < 50:
            continue

        chunks.append(make_chunk(
            title=title,
            section=section_name,
            section_display=section_display,
            content=content,
            intent=SECTION_INTENT.get(section_name, "general"),
            url=url,
            source="farmacotherapeutischkompas.nl",
        ))

    return chunks


def scrape_fk(limit=10000, batch_save=50, dataset=None, partial_path=None):
    if dataset is None:
        dataset = []
    links = get_fk_links(limit=limit)
    print(f"\n[FK] Collected {len(links)} drug links\n")

    for i, link in enumerate(links):
        print(f"[FK] [{i+1}/{len(links)}] {link}")
        try:
            chunks = parse_fk_page(link)
            dataset.extend(chunks)
        except Exception as e:
            print(f"[FK] Error: {e}")

        if partial_path and (i + 1) % batch_save == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {len(dataset)} chunks (batch)\n")

        time.sleep(DELAY)

    return dataset


# ==============================
# Main entry point
# ==============================

def scrape_dataset(
    apotheek_limit=10000,
    fk_limit=10000,
    batch_save=50,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    partial_path = os.path.join(DATA_DIR, "combined_dataset_partial.json")

    dataset = []

    # --- Source 1: apotheek.nl ---
    print("=" * 60)
    print("Scraping source 1: apotheek.nl")
    print("=" * 60)
    dataset = scrape_apotheek(
        limit=apotheek_limit,
        batch_save=batch_save,
        dataset=dataset,
        partial_path=partial_path,
    )
    print(f"\n[apotheek.nl] Done. Total chunks so far: {len(dataset)}\n")

    # --- Source 2: farmacotherapeutischkompas.nl ---
    print("=" * 60)
    print("Scraping source 2: farmacotherapeutischkompas.nl")
    print("=" * 60)
    dataset = scrape_fk(
        limit=fk_limit,
        batch_save=batch_save,
        dataset=dataset,
        partial_path=partial_path,
    )
    print(f"\n[FK] Done. Total chunks so far: {len(dataset)}\n")

    return dataset


if __name__ == "__main__":
    data = scrape_dataset()

    print(f"\nTotal chunks collected: {len(data)}")

    out_path = os.path.join(DATA_DIR, "combined_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_path}")
