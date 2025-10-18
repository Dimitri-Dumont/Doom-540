import os, json, csv, argparse, random, shutil
from pathlib import Path
from PIL import Image

"""
Now supports TWO input channels:
- /opt/ml/processing/json   (S3JsonUri) -> contains episode JSON files (at bucket root)
- /opt/ml/processing/images (S3ImagesUri) -> contains images under frames/
The JSON "frame_path" should be relative to the bucket root, e.g., "frames/frames/ep000_frame000012_...png".
We resolve it under /opt/ml/processing/images/<frame_path>.
"""

def collect_json_files(root: Path):
    return sorted([p for p in root.rglob("*.json")])

def load_episode(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)

def valid_record(rec):
    a = rec.get("action", [0,0,0])
    return isinstance(a, (list, tuple)) and len(a) == 3 and all(x in [0,1,True,False] for x in a)

def ensure_jpeg(src_img_path: Path, dst_img_path: Path, img_size, jpeg_quality: int):
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_img_path).convert("RGB") as im:
        im = im.resize(img_size, Image.Resampling.LANCZOS)
        im.save(dst_img_path, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)

def mirror_rel_png_to_jpg(rel_path: Path) -> Path:
    return rel_path.with_suffix(".jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=160)
    parser.add_argument("--img_width", type=int, default=120)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    args = parser.parse_args()

    json_root = Path("/opt/ml/processing/json")
    images_root_in = Path("/opt/ml/processing/images")  # contains frames/...

    out_train = Path("/opt/ml/processing/output/train")
    out_val = Path("/opt/ml/processing/output/val")
    out_meta = Path("/opt/ml/processing/output/meta")

    def empty_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Warn: could not remove {item}: {e}")

    for p in [out_train, out_val, out_meta]:
        empty_dir(p)
    json_files = collect_json_files(json_root)
    if not json_files:
        raise RuntimeError("No JSON files found under /opt/ml/processing/json.")

    all_rows = []
    total_frames = 0
    kept_frames = 0
    for jf in json_files:
        episode = load_episode(jf)
        total_frames += len(episode)
        for rec in episode:
            if not valid_record(rec):
                continue
            rel_path_from_bucket = Path(rec["frame_path"])  # e.g., frames/frames/ep000_...png
            abs_png = (images_root_in / rel_path_from_bucket).resolve()
            if not abs_png.exists():
                abs_jpg_alt = abs_png.with_suffix(".jpg")
                if not abs_jpg_alt.exists():
                    # skip if missing
                    continue
                rel_jpg = mirror_rel_png_to_jpg(rel_path_from_bucket)
                all_rows.append((abs_jpg_alt, rel_jpg, rec["action"]))
                kept_frames += 1
                continue
            rel_jpg = mirror_rel_png_to_jpg(rel_path_from_bucket)
            all_rows.append((abs_png, rel_jpg, rec["action"]))
            kept_frames += 1

    if not all_rows:
        raise RuntimeError("No valid frames found after scanning JSON & images. Check that frame_path matches images prefix.")

    random.shuffle(all_rows)
    val_ratio = 0.2
    split_idx = int(len(all_rows) * (1 - val_ratio))
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    def write_split(rows, out_dir: Path):
        # Note: we keep a mirrored structure under output/<split>/images/<frame_path>.jpg
        images_root_out = out_dir / "images"
        csv_path = out_dir / ("train.csv" if out_dir.name == "train" else "val.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path","label_left","label_right","label_attack"])
            for src_abs, rel_jpg, action in rows:
                dst_abs = (images_root_out / rel_jpg).resolve()
                ensure_jpeg(src_abs, dst_abs, (args.img_width, args.img_height), args.jpeg_quality)
                l, r, a = [int(bool(x)) for x in action]
                w.writerow([str(rel_jpg).replace("\\", "/"), l, r, a])

    write_split(train_rows, out_train)
    write_split(val_rows, out_val)

    meta = {
        "total_frames_in_json": total_frames,
        "kept_frames": kept_frames,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "image_size": [args.img_height, args.img_width],
        "jpeg_quality": args.jpeg_quality,
        "schema": {
            "csv_header": ["image_path","label_left","label_right","label_attack"],
            "label_definition": ["left","right","attack"]
        }
    }
    with open(out_meta / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✓ Preprocessing complete.")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()