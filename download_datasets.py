"""Download datasets from Zenodo and organize them into datasets/<name>/.

Expected final layout (matches src/data/loader.py):
    datasets/<name>/
        texts.txt
        score.txt
        splits/
            split_10.pkl
            split_5.pkl
        tfidf/
            train0.gz  test0.gz
            train1.gz  test1.gz
            ...

Usage:
    python download_datasets.py              # downloads all datasets
    python download_datasets.py webkb        # downloads only webkb
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry:
#   name       : directory name under datasets/
#   zenodo_id  : Zenodo record ID (integer)
#   zip_file   : name of the zip file in the Zenodo record
DATASETS: list[dict] = [
    {
        "name": "webkb",
        "zenodo_id": "7555368",
        "zip_file": "webkb.zip",
    },
    {
        "name": "reuters90",
        "zenodo_id": "7555298",
        "zip_file": "reut90.zip",
    },
    {
        "name": "mpqa",
        "zenodo_id": "7555268",
        "zip_file": "mpqa.zip",
    }
]

# ── Helpers ───────────────────────────────────────────────────────────────────
ZENODO_FILE_URL = "https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def _progress(block_count: int, block_size: int, total: int) -> None:
    downloaded = block_count * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct:3d}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def reorganize(dataset_dir: Path) -> None:
    """Move split_*.pkl → splits/ and ensure tfidf/ exists if present."""
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    for pkl in dataset_dir.glob("split_*.pkl"):
        target = splits_dir / pkl.name
        print(f"  Moving {pkl.name} → splits/")
        shutil.move(str(pkl), target)

    tfidf_dir = dataset_dir / "tfidf"
    if not tfidf_dir.exists():
        # Check if tfidf files are at root level
        gz_files = list(dataset_dir.glob("*.gz"))
        if gz_files:
            tfidf_dir.mkdir(exist_ok=True)
            for gz in gz_files:
                print(f"  Moving {gz.name} → tfidf/")
                shutil.move(str(gz), tfidf_dir / gz.name)


def download_dataset(entry: dict, base_dir: Path) -> None:
    name = entry["name"]
    record_id = entry["zenodo_id"]
    zip_name = entry["zip_file"]

    dataset_dir = base_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_dir / zip_name
    url = ZENODO_FILE_URL.format(record_id=record_id, filename=zip_name)

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Record  : https://zenodo.org/records/{record_id}")
    print(f"  Target  : {dataset_dir}")
    print(f"{'='*60}")

    # Download
    if zip_path.exists():
        print(f"  Zip already exists, skipping download: {zip_path}")
    else:
        download_file(url, zip_path)

    # Extract
    print(f"  Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # If zip has a single top-level directory, flatten it
        top_dirs = {p.split("/")[0] for p in zf.namelist() if p.strip("/")}
        members = zf.namelist()

        if len(top_dirs) == 1 and all(m.startswith(next(iter(top_dirs))) for m in members):
            # Strip top-level directory when extracting
            top = next(iter(top_dirs)) + "/"
            for member in members:
                relative = member[len(top):]
                if not relative:
                    continue
                target = dataset_dir / relative
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            zf.extractall(dataset_dir)

    # Remove zip to save space
    zip_path.unlink()
    print(f"  Removed {zip_name}")

    # Reorganize to match loader expectations
    reorganize(dataset_dir)

    print(f"  Done. Files in {dataset_dir}:")
    for p in sorted(dataset_dir.rglob("*"))[:20]:
        print(f"    {p.relative_to(dataset_dir)}")
    if sum(1 for _ in dataset_dir.rglob("*")) > 20:
        print("    ...")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets from Zenodo.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Names of datasets to download (default: all). E.g.: webkb reuters",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Base directory for datasets (default: datasets/)",
    )
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    registry = {d["name"]: d for d in DATASETS}

    targets = args.datasets if args.datasets else list(registry.keys())

    unknown = [t for t in targets if t not in registry]
    if unknown:
        print(f"ERROR: unknown dataset(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(registry)}")
        sys.exit(1)

    for name in targets:
        download_dataset(registry[name], base_dir)

    print(f"\nAll done. Datasets saved to '{base_dir}/'.")


if __name__ == "__main__":
    main()
