
from pathlib import Path
import shutil
import kagglehub

# Identify project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download into KaggleHub's cache
cache_path = Path(
    kagglehub.dataset_download("wordsforthewise/lending-club")
)

print("KaggleHub cache:", cache_path)

# Find the accepted loans dataset
files = [p for p in cache_path.rglob("*") if p.is_file()]

source = next(
    (p for p in files
     if p.name == "accepted_2007_to_2018Q4.csv.gz"),
    None
)

if source is None:
    candidates = [
        p for p in files
        if p.name.lower().startswith("accepted_")
        and (p.name.endswith(".csv") or p.name.endswith(".csv.gz"))
    ]

    if len(candidates) == 1:
        source = candidates[0]
    else:
        print("Available dataset files:")
        for p in files:
            print(p)
        raise FileNotFoundError("Could not identify accepted loans CSV.")

# Copy into our project
destination = RAW_DIR / source.name

if destination.exists() and destination.stat().st_size == source.stat().st_size:
    print("Dataset already exists in project.")
else:
    shutil.copy2(source, destination)
    print("Dataset copied successfully!")

print("Final dataset location:", destination)
print("File size (GB):", round(destination.stat().st_size / 1e9, 2))
