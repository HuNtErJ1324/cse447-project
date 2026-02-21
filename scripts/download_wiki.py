#!/usr/bin/env python3
"""Download multilingual text from Wikipedia API and other web sources."""
import urllib.request
import json
import os
import time

LANGUAGES = {
    'ar': 'Arabic', 'bn': 'Bengali', 'zh': 'Chinese', 'hr': 'Croatian',
    'cs': 'Czech', 'nl': 'Dutch', 'fi': 'Finnish', 'fr': 'French',
    'de': 'German', 'el': 'Greek', 'hi': 'Hindi', 'id': 'Indonesian',
    'ko': 'Korean', 'ms': 'Malay', 'no': 'Norwegian', 'pt': 'Portuguese',
    'ru': 'Russian', 'sk': 'Slovak', 'es': 'Spanish', 'sw': 'Swahili',
    'sv': 'Swedish', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'it': 'Italian', 'ha': 'Hausa', 'en': 'English',
}

TARGET_LINES = 3000
output_path = "data/multilingual_large.txt"

def get_wiki_random_articles(lang, count=50):
    """Get random articles from Wikipedia API."""
    lines = []
    for batch in range(count // 10):
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
            # Get multiple random articles
            for _ in range(10):
                req = urllib.request.Request(url, headers={'User-Agent': 'CSE447-NLP-Project/1.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    extract = data.get('extract', '')
                    if extract and len(extract) > 20:
                        # Split into sentences
                        for sent in extract.replace('\n', ' ').split('. '):
                            sent = sent.strip()
                            if len(sent) > 10:
                                lines.append(sent)
                time.sleep(0.1)
        except Exception as e:
            print(f"  Wiki API error for {lang}: {e}")
            break
        if len(lines) >= TARGET_LINES:
            break
    return lines

def get_wiki_featured(lang, count=200):
    """Get content from Wikipedia using search for common topics."""
    lines = []
    topics = [
        "history", "science", "music", "family", "education", "nature",
        "time", "love", "friendship", "language", "culture", "food",
        "sport", "city", "country", "water", "earth", "sun",
        "computer", "human", "animal", "tree", "ocean", "mountain",
    ]
    for topic in topics:
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={topic}&utf8=&format=json&srlimit=20"
            req = urllib.request.Request(url, headers={'User-Agent': 'CSE447-NLP-Project/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                results = data.get('query', {}).get('search', [])
                for r in results:
                    # Get article text
                    title = r.get('title', '')
                    if not title:
                        continue
                    text_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}&prop=extracts&explaintext=true&format=json&exlimit=1"
                    req2 = urllib.request.Request(text_url, headers={'User-Agent': 'CSE447-NLP-Project/1.0'})
                    with urllib.request.urlopen(req2, timeout=10) as resp2:
                        data2 = json.loads(resp2.read().decode('utf-8'))
                        pages = data2.get('query', {}).get('pages', {})
                        for page in pages.values():
                            extract = page.get('extract', '')
                            if extract:
                                for sent in extract.replace('\n', ' ').split('. '):
                                    sent = sent.strip()
                                    if 10 < len(sent) < 300:
                                        lines.append(sent)
                                        if len(lines) >= TARGET_LINES:
                                            return lines
            time.sleep(0.2)
        except Exception as e:
            continue
    return lines

all_lines = []
for lang, name in LANGUAGES.items():
    print(f"\n=== {name} ({lang}) ===")
    lines = get_wiki_featured(lang, count=200)
    lines = list(dict.fromkeys(lines))[:TARGET_LINES]
    print(f"  Collected {len(lines)} lines")
    all_lines.extend(lines)

print(f"\nTotal lines: {len(all_lines)}")
with open(output_path, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
print(f"Written to {output_path}")
