import os, json, argparse, numpy as np, tensorflow as tf

def resolve_image_path(img_path, csv_path):
    import os
    if os.path.isabs(img_path):
        return img_path
    base_dir = os.path.dirname(csv_path)
    images_root = os.path.join(base_dir, "images")
    return os.path.join(images_root, img_path)


def read_csv(path):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            img = resolve_image_path(parts[0], path)
            l, r, a = map(int, parts[1:])
            rows.append((img, [l, r, a]))
    return rows

def load_dataset(rows, img_size):
    paths = tf.constant([r[0] for r in rows])
    labels = tf.constant([r[1] for r in rows], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)/255.0
        return img, y
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

def f1_precision_recall(y_true, y_pred, thresh=0.5):
    y_hat = (y_pred >= thresh).astype(np.float32)
    eps = 1e-8
    p_list, r_list, f1_list = [], [], []
    for i in range(y_true.shape[1]):
        tp = np.sum((y_true[:, i] == 1) & (y_hat[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_hat[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_hat[:, i] == 0))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        p_list.append(float(precision)); r_list.append(float(recall)); f1_list.append(float(f1))
    macro = {
        "precision": float(np.mean(p_list)),
        "recall": float(np.mean(r_list)),
        "f1": float(np.mean(f1_list)),
    }
    return p_list, r_list, f1_list, macro

from pathlib import Path
import tarfile

MODEL_MOUNT = Path("/opt/ml/processing/model")

def ensure_extracted_model_dir(mount: Path) -> Path:
    """Find or extract a real SavedModel directory from the SageMaker model mount."""
    # Already a SavedModel folder?
    for cand in [mount, mount / "1"]:
        if (cand / "saved_model.pb").exists() or (cand / "saved_model.pbtxt").exists():
            return cand

    # If there's a model.tar.gz, extract it
    tars = list(mount.glob("*.tar.gz"))
    if tars:
        extract_dir = mount / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0], "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # Check likely subfolders
        for cand in [extract_dir / "1", extract_dir]:
            if (cand / "saved_model.pb").exists():
                return cand
        for sub in extract_dir.iterdir():
            if (sub / "saved_model.pb").exists():
                return sub

    # Fallback
    for pb in mount.rglob("saved_model.pb"):
        return pb.parent

    raise FileNotFoundError(f"Could not find a SavedModel under {mount}")

def main():
    meta_path = "/opt/ml/processing/meta/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    img_h, img_w = meta.get("image_size", [160, 120])
    img_size = [img_h, img_w]

    val_csv = "/opt/ml/processing/val/val.csv"
    rows = read_csv(val_csv)
    ds = load_dataset(rows, img_size)

    model_dir = ensure_extracted_model_dir(MODEL_MOUNT)
    model = tf.keras.models.load_model(str(model_dir))
    y_true_all, y_pred_all = [], []
    for x, y in ds:
        y_pred = model.predict(x, verbose=0)
        y_true_all.append(y.numpy())
        y_pred_all.append(y_pred)
    import numpy as np
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    p_list, r_list, f1_list, macro = f1_precision_recall(y_true, y_pred, 0.5)

    evaluation = {
        "binary_classification_metrics": {
            "precision_by_label": p_list,
            "recall_by_label": r_list,
            "f1_by_label": f1_list,
            "macro_avg": macro,
            "labels": ["left", "right", "attack"],
        }
    }

    out_dir = "/opt/ml/processing/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    print(json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    main()