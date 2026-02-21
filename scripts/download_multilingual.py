#!/usr/bin/env python3
"""Download multilingual text data from HuggingFace datasets."""
import os
import sys

# Languages we need based on dev set analysis
# Map language to ISO code for various datasets
LANGUAGES = {
    'ar': 'Arabic', 'bn': 'Bengali', 'zh': 'Chinese', 'hr': 'Croatian',
    'cs': 'Czech', 'nl': 'Dutch', 'fi': 'Finnish', 'fr': 'French',
    'de': 'German', 'el': 'Greek', 'hi': 'Hindi', 'id': 'Indonesian',
    'ko': 'Korean', 'ms': 'Malay', 'no': 'Norwegian', 'pt': 'Portuguese',
    'ru': 'Russian', 'sk': 'Slovak', 'es': 'Spanish', 'sw': 'Swahili',
    'sv': 'Swedish', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'it': 'Italian', 'ha': 'Hausa', 'nb': 'Norwegian Bokmål',
    'en': 'English',
}

TARGET_LINES_PER_LANG = 5000
output_path = "data/multilingual_large.txt"
lines_written = 0

def try_mc4(lang_code):
    """Try mc4 dataset."""
    from datasets import load_dataset
    lines = []
    try:
        ds = load_dataset("mc4", lang_code, split="train", streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            text = ex.get("text", "").strip()
            if text and 10 < len(text) < 500:
                # Split into sentences
                for sent in text.replace('\n', ' ').split('. '):
                    sent = sent.strip()
                    if len(sent) > 10:
                        lines.append(sent)
                        if len(lines) >= TARGET_LINES_PER_LANG:
                            return lines
            if i > 50000:
                break
    except Exception as e:
        print(f"  mc4 failed for {lang_code}: {e}")
    return lines

def try_wiki(lang_code):
    """Try Wikipedia dataset."""
    from datasets import load_dataset
    lines = []
    try:
        wiki_code = f"20220301.{lang_code}"
        ds = load_dataset("wikipedia", wiki_code, split="train", streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            text = ex.get("text", "").strip()
            if text:
                for sent in text.replace('\n', ' ').split('. '):
                    sent = sent.strip()
                    if 10 < len(sent) < 300:
                        lines.append(sent)
                        if len(lines) >= TARGET_LINES_PER_LANG:
                            return lines
            if i > 20000:
                break
    except Exception as e:
        print(f"  wiki failed for {lang_code}: {e}")
    return lines

def try_cc100(lang_code):
    """Try CC-100 dataset."""
    from datasets import load_dataset
    lines = []
    try:
        ds = load_dataset("cc100", lang=lang_code, split="train", streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            text = ex.get("text", "").strip()
            if text and 10 < len(text) < 500:
                lines.append(text)
                if len(lines) >= TARGET_LINES_PER_LANG:
                    return lines
            if i > 50000:
                break
    except Exception as e:
        print(f"  cc100 failed for {lang_code}: {e}")
    return lines

def try_flores(lang_code):
    """Try FLORES-200 dataset for parallel sentences."""
    from datasets import load_dataset
    lines = []
    # FLORES uses different codes
    flores_map = {
        'ar': 'ara_Arab', 'bn': 'ben_Beng', 'zh': 'zho_Hans', 'hr': 'hrv_Latn',
        'cs': 'ces_Latn', 'nl': 'nld_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn',
        'de': 'deu_Latn', 'el': 'ell_Grek', 'hi': 'hin_Deva', 'id': 'ind_Latn',
        'ko': 'kor_Hang', 'ms': 'zsm_Latn', 'no': 'nob_Latn', 'nb': 'nob_Latn',
        'pt': 'por_Latn', 'ru': 'rus_Cyrl', 'sk': 'slk_Latn', 'es': 'spa_Latn',
        'sw': 'swh_Latn', 'sv': 'swe_Latn', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl',
        'ur': 'urd_Arab', 'vi': 'vie_Latn', 'it': 'ita_Latn', 'ha': 'hau_Latn',
        'en': 'eng_Latn',
    }
    fc = flores_map.get(lang_code)
    if not fc:
        return lines
    try:
        ds = load_dataset("facebook/flores", fc, split="devtest", trust_remote_code=True)
        for ex in ds:
            text = ex.get("sentence", "").strip()
            if text and len(text) > 5:
                lines.append(text)
    except Exception as e:
        print(f"  flores failed for {lang_code}: {e}")
    return lines

print("Downloading multilingual data...")
all_lines = []

for lang_code, lang_name in LANGUAGES.items():
    print(f"\n=== {lang_name} ({lang_code}) ===")
    
    # Try FLORES first (small but high quality)
    lines = try_flores(lang_code)
    print(f"  FLORES: {len(lines)} lines")
    
    # Then try CC-100
    if len(lines) < TARGET_LINES_PER_LANG:
        cc_lines = try_cc100(lang_code)
        print(f"  CC-100: {len(cc_lines)} lines")
        lines.extend(cc_lines)
    
    # Then mc4
    if len(lines) < TARGET_LINES_PER_LANG:
        mc4_lines = try_mc4(lang_code)
        print(f"  MC4: {len(mc4_lines)} lines")
        lines.extend(mc4_lines)
    
    # Deduplicate and limit
    lines = list(dict.fromkeys(lines))[:TARGET_LINES_PER_LANG]
    print(f"  Total for {lang_name}: {len(lines)} lines")
    all_lines.extend(lines)

print(f"\nTotal lines collected: {len(all_lines)}")
with open(output_path, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
print(f"Written to {output_path}")
