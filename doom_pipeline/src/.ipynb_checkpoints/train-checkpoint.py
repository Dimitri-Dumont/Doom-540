import os, argparse, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def resolve_image_path(img_path, csv_path):
    if os.path.isabs(img_path):
        return img_path
    base_dir = os.path.dirname(csv_path)
    images_root = os.path.join(base_dir, "images")
    return os.path.join(images_root, img_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--img_height", type=int, default=160)
    p.add_argument("--img_width", type=int, default=120)
    p.add_argument("--num_actions", type=int, default=3)
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()

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

def build_dataset(rows, img_height, img_width, batch_size, shuffle=False):
    paths = tf.constant([r[0] for r in rows])
    labels = tf.constant([r[1] for r in rows], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, tf.float32)/255.0
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(rows), 4096), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(img_height, img_width, num_actions):
    model = keras.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_actions, activation="sigmoid"),
    ])
    return model

def main():
    args = parse_args()

    train_csv = "/opt/ml/input/data/train/train.csv"
    val_csv   = "/opt/ml/input/data/val/val.csv"

    train_rows = read_csv(train_csv)
    val_rows   = read_csv(val_csv)

    train_ds = build_dataset(train_rows, args.img_height, args_img_width := args.img_width, args.batch_size, shuffle=True)
    val_ds   = build_dataset(val_rows, args.img_height, args_img_width, args.batch_size, shuffle=False)

    model = build_model(args.img_height, args.img_width, args.num_actions)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="bin_acc"), keras.metrics.Precision(), keras.metrics.Recall()],
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath="/opt/ml/model",
        monitor="val_bin_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False
    )
    es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlrop = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, es, rlrop],
        verbose=2
    )

    model.save("/opt/ml/model")

    os.makedirs("/opt/ml/output/data", exist_ok=True)
    with open("/opt/ml/output/data/train_summary.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

if __name__ == "__main__":
    main()