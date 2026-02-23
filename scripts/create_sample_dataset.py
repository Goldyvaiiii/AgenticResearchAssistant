"""
Script to generate a small sample dataset (~2000 papers) for cloud deployment.
Run once locally: python scripts/create_sample_dataset.py
"""
import json
import random
import csv
import os

FULL_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "arxiv-metadata-oai-snapshot.csv")
SAMPLE_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "arxiv_sample.csv")
SAMPLE_SIZE = 2000
MAX_SCAN = 100_000
SEED = 42

random.seed(SEED)
reservoir = []
n = 0

print(f"Scanning up to {MAX_SCAN} records to build reservoir of {SAMPLE_SIZE}...")
with open(FULL_DATA, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= MAX_SCAN:
            break
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue

        abstract = (p.get("abstract") or "").replace("\n", " ").strip()
        title = (p.get("title") or "").replace("\n", " ").strip()
        if len(abstract) < 50:
            continue

        record = {
            "id": p.get("id", ""),
            "title": title,
            "abstract": abstract,
            "categories": p.get("categories", ""),
            "authors": p.get("authors", ""),
            "update_date": p.get("update_date", ""),
        }
        n += 1
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(record)
        else:
            j = random.randint(0, n - 1)
            if j < SAMPLE_SIZE:
                reservoir[j] = record

print(f"Writing {len(reservoir)} records to {SAMPLE_OUT}")
with open(SAMPLE_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "title", "abstract", "categories", "authors", "update_date"])
    writer.writeheader()
    writer.writerows(reservoir)

print("Done! Sample dataset created.")
