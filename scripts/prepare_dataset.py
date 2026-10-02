"""
Dataset Preparation Script for Product Matching Demo

Creates two CSV files from ProMapEn dataset:
1. products.csv - Products for the database (product1 from match pairs)
2. queries.csv - Queries to test:
   - With match (product2 from match pairs → should find product1 in DB)
   - Without match (product2 from no-match pairs → no match in DB)

Dataset Structure (ProMapEn):
- name1, short_description1, long_description1, specification1, image1, price1, id1 → Product 1
- name2, short_description2, long_description2, specification2, image2, price2, id2 → Product 2
- match → 1 = match, 0 = no match
- image_url1, image_url2 → JSON array strings with image URLs

Usage:
    python scripts/prepare_dataset.py
"""

import os
import json
import hashlib
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
from tqdm import tqdm


# Configuration
DATASET_URL = "https://raw.githubusercontent.com/kackamac/Product-Mapping-Datasets/main/Basic%20ProMap%20Datasets/ProMapEn/promapen-train_data.csv"
OUTPUT_DIR = Path("dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
PRODUCTS_CSV = OUTPUT_DIR / "products.csv"
QUERIES_CSV = OUTPUT_DIR / "queries.csv"

TARGET_MATCH_PAIRS = 500      # Match pairs → products in DB + queries with match
TARGET_NOMATCH_QUERIES = 500  # No-match queries (product2 from no-match pairs)

MAX_WORKERS = 8
REQUEST_TIMEOUT = 15


def download_csv() -> pd.DataFrame:
    """Download the ProMapEn dataset CSV from GitHub."""
    print("Downloading ProMapEn dataset...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    # Save raw CSV temporarily
    temp_path = OUTPUT_DIR / "raw_promapen.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(response.content)

    df = pd.read_csv(temp_path)
    print(f"   Downloaded {len(df)} rows")
    return df


def parse_image_urls(url_string: str) -> List[str]:
    """Parse image URL from JSON array string. Returns first valid URL."""
    if pd.isna(url_string):
        return []

    try:
        # It's a JSON array string like '["url1", "url2"]'
        urls = json.loads(url_string)
        if isinstance(urls, list):
            return [url for url in urls if isinstance(url, str) and url.startswith("http")]
        elif isinstance(urls, str) and urls.startswith("http"):
            return [urls]
    except (json.JSONDecodeError, TypeError):
        # Maybe it's a plain URL string
        if isinstance(url_string, str) and url_string.startswith("http"):
            return [url_string]

    return []


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image from URL."""
    if save_path.exists():
        return True

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type.lower():
            return False

        save_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def get_image_filename(url: str, prefix: str) -> str:
    """Generate a unique filename for an image URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    ext = "jpg"
    if "." in url:
        potential_ext = url.split(".")[-1].split("?")[0].lower()
        if potential_ext in ["jpg", "jpeg", "png", "gif", "webp"]:
            ext = potential_ext
    return f"{prefix}_{url_hash}.{ext}"


def build_description(row: pd.Series, suffix: str) -> str:
    """Build a combined text description from available fields."""
    parts = []

    # Name
    name_col = f"name{suffix}"
    if name_col in row.index and pd.notna(row[name_col]):
        parts.append(str(row[name_col]).strip())

    # Short description
    short_col = f"short_description{suffix}"
    if short_col in row.index and pd.notna(row[short_col]):
        short_desc = str(row[short_col]).strip()
        if short_desc and short_desc not in parts:
            parts.append(short_desc)

    # Long description (truncated)
    long_col = f"long_description{suffix}"
    if long_col in row.index and pd.notna(row[long_col]):
        long_desc = str(row[long_col]).strip()
        if len(long_desc) > 500:
            long_desc = long_desc[:500] + "..."
        if long_desc and long_desc not in parts:
            parts.append(long_desc)

    return " | ".join(parts) if parts else ""


def prepare_dataset() -> None:
    """Main function to prepare the dataset."""
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download CSV
    df = download_csv()

    # Parse image URLs and filter rows with valid images
    print("\nParsing image URLs...")
    df["parsed_urls_1"] = df["image_url1"].apply(parse_image_urls)
    df["parsed_urls_2"] = df["image_url2"].apply(parse_image_urls)

    # Filter rows where both products have at least one valid image URL
    df = df[df["parsed_urls_1"].apply(len) > 0]
    df = df[df["parsed_urls_2"].apply(len) > 0]
    print(f"   Rows with valid image URLs: {len(df)}")

    # Separate match and non-match
    matches = df[df["match"] == 1].reset_index(drop=True)
    non_matches = df[df["match"] == 0].reset_index(drop=True)

    print(f"\nAvailable data:")
    print(f"   Match pairs: {len(matches)}")
    print(f"   Non-match pairs: {len(non_matches)}")

    # Sample required amounts
    n_matches = min(TARGET_MATCH_PAIRS, len(matches))
    n_non_matches = min(TARGET_NOMATCH_QUERIES, len(non_matches))

    sampled_matches = matches.sample(n=n_matches, random_state=42).reset_index(drop=True)
    sampled_non_matches = non_matches.sample(n=n_non_matches, random_state=42).reset_index(drop=True)

    print(f"\nSelected:")
    print(f"   Match pairs: {len(sampled_matches)} -> {len(sampled_matches)} DB products + {len(sampled_matches)} match queries")
    print(f"   No-match queries: {len(sampled_non_matches)}")

    # Collect all image URLs to download
    print("\nCollecting images to download...")
    download_tasks = []

    # From match pairs: product1 for DB, product2 for query
    for idx, row in sampled_matches.iterrows():
        # Product 1 (for DB) - use first URL
        urls_1 = row["parsed_urls_1"]
        if urls_1:
            url1 = urls_1[0]
            filename1 = get_image_filename(url1, f"match_{idx}_db")
            download_tasks.append((url1, IMAGES_DIR / filename1, f"match_{idx}_db"))

        # Product 2 (for query) - use first URL
        urls_2 = row["parsed_urls_2"]
        if urls_2:
            url2 = urls_2[0]
            filename2 = get_image_filename(url2, f"match_{idx}_query")
            download_tasks.append((url2, IMAGES_DIR / filename2, f"match_{idx}_query"))

    # From non-match: product2 will be query with no match in DB
    for idx, row in sampled_non_matches.iterrows():
        urls_2 = row["parsed_urls_2"]
        if urls_2:
            url2 = urls_2[0]
            filename2 = get_image_filename(url2, f"nomatch_{idx}_query")
            download_tasks.append((url2, IMAGES_DIR / filename2, f"nomatch_{idx}_query"))

    # Download images
    print(f"\nDownloading {len(download_tasks)} images...")
    image_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, url, path): (key, path.name)
            for url, path, key in download_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            key, filename = futures[future]
            try:
                success = future.result()
                if success:
                    image_results[key] = filename
            except Exception:
                pass

    print(f"   Successfully downloaded: {len(image_results)}/{len(download_tasks)}")

    # Build products.csv and queries.csv
    print("\nBuilding CSV files...")

    products = []  # DB products from match pairs
    queries = []   # Match queries + no-match queries

    valid_match_count = 0
    valid_nomatch_count = 0

    # Process match pairs
    for idx, row in sampled_matches.iterrows():
        db_key = f"match_{idx}_db"
        query_key = f"match_{idx}_query"

        # Both images must exist
        if db_key not in image_results or query_key not in image_results:
            continue

        db_id = f"db_{valid_match_count:04d}"

        # Add product1 to DB
        products.append({
            "db_id": db_id,
            "name": str(row["name1"]) if pd.notna(row["name1"]) else "",
            "description": build_description(row, "1"),
            "image": image_results[db_key],
            "price": str(row["price1"]) if pd.notna(row["price1"]) else "",
            "source_id": str(row["id1"]) if pd.notna(row["id1"]) else ""
        })

        # Add product2 as query (should find product1 in DB)
        queries.append({
            "query_id": f"q_{valid_match_count:04d}_match",
            "name": str(row["name2"]) if pd.notna(row["name2"]) else "",
            "description": build_description(row, "2"),
            "image": image_results[query_key],
            "price": str(row["price2"]) if pd.notna(row["price2"]) else "",
            "source_id": str(row["id2"]) if pd.notna(row["id2"]) else "",
            "has_match": True,
            "ground_truth_db_id": db_id  # Should find this in DB
        })

        valid_match_count += 1

    # Process non-match queries (product2 from no-match pairs)
    for idx, row in sampled_non_matches.iterrows():
        query_key = f"nomatch_{idx}_query"

        if query_key not in image_results:
            continue

        queries.append({
            "query_id": f"q_{valid_nomatch_count:04d}_nomatch",
            "name": str(row["name2"]) if pd.notna(row["name2"]) else "",
            "description": build_description(row, "2"),
            "image": image_results[query_key],
            "price": str(row["price2"]) if pd.notna(row["price2"]) else "",
            "source_id": str(row["id2"]) if pd.notna(row["id2"]) else "",
            "has_match": False,
            "ground_truth_db_id": ""  # No match in DB
        })

        valid_nomatch_count += 1

    # Create DataFrames and save
    products_df = pd.DataFrame(products)
    queries_df = pd.DataFrame(queries)

    products_df.to_csv(PRODUCTS_CSV, index=False)
    queries_df.to_csv(QUERIES_CSV, index=False)

    # Cleanup temp file
    temp_path = OUTPUT_DIR / "raw_promapen.csv"
    if temp_path.exists():
        temp_path.unlink()

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nproducts.csv:")
    print(f"   Total products in DB: {len(products_df)}")
    print(f"   Columns: {list(products_df.columns)}")
    print(f"   Path: {PRODUCTS_CSV}")

    print(f"\nqueries.csv:")
    print(f"   Total queries: {len(queries_df)}")
    match_queries = queries_df[queries_df['has_match'] == True]
    nomatch_queries = queries_df[queries_df['has_match'] == False]
    print(f"   - With match (has_match=True): {len(match_queries)}")
    print(f"   - Without match (has_match=False): {len(nomatch_queries)}")
    print(f"   Columns: {list(queries_df.columns)}")
    print(f"   Path: {QUERIES_CSV}")

    print(f"\nImages:")
    print(f"   Downloaded: {len(list(IMAGES_DIR.glob('*')))}")
    print(f"   Path: {IMAGES_DIR}")

    print("\n" + "=" * 60)
    print("\nExpected evaluation flow:")
    print("1. Index all products.csv into ChromaDB")
    print("2. For each query in queries.csv:")
    print("   - Retrieve top-K candidates from DB")
    print("   - Rerank candidates")
    print("   - If has_match=True: check if ground_truth_db_id is in top results")
    print("   - If has_match=False: all retrieved candidates should be rejected (low score)")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
