"""
Data Loader for ArXiv dataset.
Handles efficient loading and sampling from the large JSONL dataset.
"""
import json
import random
import pandas as pd
from tqdm import tqdm


def load_arxiv_sample(
    filepath: str,
    sample_size: int = 5000,
    keyword_filter: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a sample of records from the ArXiv JSONL dataset.

    Args:
        filepath: Path to the arxiv-metadata-oai-snapshot.csv (JSONL) file.
        sample_size: Maximum number of records to return.
        keyword_filter: Optional keyword to filter by title/abstract.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: id, title, abstract, categories.
    """
    random.seed(seed)
    records = []
    keyword_lower = keyword_filter.lower() if keyword_filter else None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning dataset", unit=" records"):
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            abstract = paper.get("abstract", "") or ""
            title = paper.get("title", "") or ""

            if keyword_lower:
                combined = (title + " " + abstract).lower()
                if keyword_lower not in combined:
                    continue

            records.append(
                {
                    "id": paper.get("id", ""),
                    "title": title.replace("\n", " ").strip(),
                    "abstract": abstract.replace("\n", " ").strip(),
                    "categories": paper.get("categories", ""),
                    "authors": paper.get("authors", ""),
                    "update_date": paper.get("update_date", ""),
                }
            )

            if len(records) >= sample_size:
                break

    df = pd.DataFrame(records)
    return df


def load_random_sample(
    filepath: str,
    sample_size: int = 5000,
    max_scan: int = 200000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a random sample without keyword filtering (faster scan).

    Args:
        filepath: Path to dataset.
        sample_size: Number of records to sample.
        max_scan: Max lines to scan before stopping.
        seed: Random seed.

    Returns:
        DataFrame with paper metadata.
    """
    random.seed(seed)
    reservoir = []
    n = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading sample", unit=" records", total=max_scan)):
            if i >= max_scan:
                break
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            abstract = paper.get("abstract", "") or ""
            title = paper.get("title", "") or ""

            record = {
                "id": paper.get("id", ""),
                "title": title.replace("\n", " ").strip(),
                "abstract": abstract.replace("\n", " ").strip(),
                "categories": paper.get("categories", ""),
                "authors": paper.get("authors", ""),
                "update_date": paper.get("update_date", ""),
            }

            n += 1
            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                j = random.randint(0, n - 1)
                if j < sample_size:
                    reservoir[j] = record

    df = pd.DataFrame(reservoir)
    return df
