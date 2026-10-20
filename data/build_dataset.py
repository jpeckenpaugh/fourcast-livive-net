#!/usr/bin/env python3
"""Build 4cast's balanced cat/dog set from the small CC-BY Zenodo sample."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOWNLOADS = DATA / "downloads"
ZIP = DOWNLOADS / "cats_dogs_light.zip"
URL = "https://zenodo.org/records/5226945/files/cats_dogs_light.zip?download=1"
MD5 = "5e014163374c3bf7069c923de2d619c8"
OUT = DATA / "crops"


def fetch():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and hashlib.md5(ZIP.read_bytes()).hexdigest() != MD5:
        ZIP.unlink()
    if not ZIP.exists():
        part = ZIP.with_suffix(".zip.part")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "4cast-dataset-builder/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.md5(part.read_bytes()).hexdigest() != MD5:
                raise RuntimeError("Zenodo archive checksum mismatch")
            part.replace(ZIP)
        finally:
            part.unlink(missing_ok=True)
    return zipfile.ZipFile(ZIP)


def main():
    with fetch() as archive:
        members = archive.namelist()
        # The released archive is organized in cat/dog folders; labels come from
        # those directory names, never inferred from visual content.
        candidates = {"cat": [], "dog": []}
        for member in members:
            parts = Path(member).parts
            if len(parts) < 2:
                continue
            filename = Path(member).name.lower()
            label = next((name for name in candidates if filename.startswith(name + ".")), None)
            if label and member.lower().endswith((".jpg", ".jpeg", ".png")):
                candidates[label].append(member)
        selected = {}
        for label in candidates:
            candidates[label].sort()
            selected[label] = []
            for member in candidates[label]:
                try:
                    from io import BytesIO
                    with archive.open(member) as raw:
                        image = Image.open(BytesIO(raw.read())).convert("RGB")
                    if min(image.size) >= 128:
                        selected[label].append((member, image.copy()))
                except Exception:
                    continue
                if len(selected[label]) == 512:
                    break
            if len(selected[label]) < 512:
                raise RuntimeError(f"Expected 512 usable {label} files (at least 128px); got {len(selected[label])}")

        OUT.mkdir(parents=True, exist_ok=True)
        manifest = []
        for label in ("cat", "dog"):
            for index, (member, image) in enumerate(selected[label]):
                ident = f"zenodo5226945_{label}_{index + 1:04d}"
                width, height = image.size
                side = min(width, height)
                left, top = (width - side) // 2, (height - side) // 2
                crop = image.crop((left, top, left + side, top + side))
                crop.save(OUT / f"{ident}.jpg", "JPEG", quality=92)
                manifest.append({
                    "id": ident,
                    "file": f"data/crops/{ident}.jpg",
                    "label": label,
                    "split": "train" if index < 384 else "val" if index < 448 else "holdout",
                    "crop_rect": [left, top, side, side],
                    "source": f"zenodo:5226945/{member}",
                })
        manifest.sort(key=lambda row: row["id"])
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        live = {row["file"].split("/")[-1] for row in manifest}
        for path in OUT.glob("zenodo5226945_*.jpg"):
            if path.name not in live:
                path.unlink()
        print("Wrote 1024 crops (512 cat, 512 dog; 768 train, 128 val, 128 holdout).")


if __name__ == "__main__":
    main()
