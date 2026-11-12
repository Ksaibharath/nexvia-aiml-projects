"""Download and extract the MovieLens 25M dataset."""
import os
import pathlib
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(raw_dir: str | None = None, extract_only_essential: bool = False) -> pathlib.Path:
    raw = pathlib.Path(raw_dir or os.getenv("ML25M_RAW_DIR", "data/raw"))
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "ml-25m.zip"
    target_dir = raw / "ml-25m"

    if not zip_path.exists() and not (target_dir / "ratings.csv").exists():
        print(f"Downloading MovieLens 25M (~250 MB) from {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)

    if zip_path.exists() and not (target_dir / "ratings.csv").exists():
        print("Extracting archive...")
        with zipfile.ZipFile(zip_path) as zf:
            if extract_only_essential:
                members = [
                    m for m in zf.namelist()
                    if any(m.endswith(k) for k in ("ratings.csv", "movies.csv", "tags.csv", "README.txt"))
                ]
                zf.extractall(raw, members=members)
            else:
                zf.extractall(raw)

    files = sorted(p.name for p in target_dir.iterdir())
    print("done:", files)
    return target_dir


if __name__ == "__main__":
    download_movielens(
        extract_only_essential=os.getenv("ML25M_ESSENTIAL_ONLY", "0") == "1"
    )
