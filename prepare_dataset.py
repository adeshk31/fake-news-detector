import pandas as pd
import re

print("Loading raw datasets...")
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print(f"  Fake articles loaded : {len(fake)}")
print(f"  Real articles loaded : {len(true)}")


def strip_source_metadata(text):
    """
    Remove dateline/agency metadata that leaks label information.
    True.csv articles often start with patterns like:
      'WASHINGTON (Reuters) - ...'
      '(Reuters) - ...'
    Removing these prevents the model from shortcutting on agency names.
    """
    if not isinstance(text, str):
        return ""

    # Remove leading patterns: "CITY (Reuters) - " or "(Reuters) - "
    text = re.sub(r"^[A-Z\s/,\.]+ \(Reuters\)\s*[-–]\s*", "", text)
    text = re.sub(r"^\(Reuters\)\s*[-–]\s*", "", text)

    # Remove AP datelines too
    text = re.sub(r"^[A-Z\s/,\.]+ \(AP\)\s*[-–]\s*", "", text)
    text = re.sub(r"^\(AP\)\s*[-–]\s*", "", text)

    return text.strip()


# ── Label with strings (FAKE / REAL) ───────────────────────
fake["label"] = "FAKE"
true["label"] = "REAL"

# ── Strip source metadata ───────────────────────────────────
fake["text"] = fake["text"].apply(strip_source_metadata)
true["text"] = true["text"].apply(strip_source_metadata)

# ── Combine & clean ─────────────────────────────────────────
data = pd.concat([fake, true], ignore_index=True)
data = data[["text", "label"]]

# Drop rows with missing or very short text
data = data.dropna(subset=["text"])
data = data[data["text"].str.strip().str.len() > 100]

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ────────────────────────────────────────────────────
data.to_csv("data/news_dataset.csv", index=False)

print(f"\nDataset prepared successfully!")
print(f"  Total samples : {len(data)}")
print(f"  FAKE          : {(data['label'] == 'FAKE').sum()}")
print(f"  REAL          : {(data['label'] == 'REAL').sum()}")

print(f"\nSample REAL article (first 200 chars):")
print(f"  {repr(data[data['label'] == 'REAL'].iloc[0]['text'][:200])}")

print(f"\nSample FAKE article (first 200 chars):")
print(f"  {repr(data[data['label'] == 'FAKE'].iloc[0]['text'][:200])}")