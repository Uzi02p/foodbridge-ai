"""
train_freshness.py
------------------
Trains the FRESHNESS CLASSIFIER (AI part 1) with transfer learning.

Run it with:
    python train_freshness.py              # default: 5 training rounds (epochs)
    python train_freshness.py --epochs 3   # fewer rounds = faster

What is transfer learning?
    MobileNetV2 is a model Google already trained on 1.4 million everyday
    photos (ImageNet). It already "knows" edges, colours, textures and shapes.
    We keep that knowledge FROZEN and only train one small new layer on top
    that answers our question: "is this food fresh or rotten?"
    This needs far less data and time than training from scratch.

Dataset: Kaggle "Fruits fresh and rotten for classification".
If it isn't found, this script prints exactly where to download it.

Outputs (all in models/):
    freshness.h5                       - the trained model
    freshness_confusion_matrix.png     - picture of right/wrong predictions
    freshness_metrics.json             - accuracy numbers (shown in the app)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.freshness_model import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "fresh_rotten"
MODELS_DIR = PROJECT_DIR / "models"
BATCH_SIZE = 32

HOW_TO_GET_DATA = f"""
Dataset not found at: {DATA_DIR}

How to get it:
  1. Open https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
     (free Kaggle account needed) and click "Download". You get a ~3.5 GB zip file.
  2. Unzip it. Inside the 'dataset' folder you will find 'train' and 'test'.
  3. Move those two folders here so it looks like this:

     foodbridge-ai/data/fresh_rotten/
     ├── train/
     │   ├── freshapples/     ├── rottenapples/
     │   ├── freshbanana/     ├── rottenbanana/
     │   └── freshoranges/    └── rottenoranges/
     └── test/
         └── (same 6 folders)

  Any folder whose name starts with "fresh" counts as Fresh, and any folder
  starting with "rotten" counts as Stale, so you can add more foods the same way.
  If you have no test/ folder, 20% of train/ is held back for testing instead.

Then run:  python train_freshness.py
"""


# ---------------------------------------------------------------------------
# Step 1: find the images and give each one a label
# ---------------------------------------------------------------------------
def find_images(split_dir: Path) -> tuple[list[str], list[int]]:
    """
    Look inside e.g. data/fresh_rotten/train/ and list every image.
    Label 0 = Fresh (folder name starts with "fresh"),
    Label 1 = Stale (folder name starts with "rotten").
    """
    paths, labels = [], []
    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if name.startswith("fresh"):
            label = 0
        elif name.startswith("rotten"):
            label = 1
        else:
            print(f"  (skipping folder '{folder.name}': name must start with 'fresh' or 'rotten')")
            continue
        for file in folder.iterdir():
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(file))
                labels.append(label)
    return paths, labels


# ---------------------------------------------------------------------------
# Step 2: turn the list of files into a TensorFlow dataset that loads
#         and resizes images in batches (so we never load all 13,000 at once)
# ---------------------------------------------------------------------------
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)  # every image becomes 224x224
    return image, label


def make_dataset(paths, labels, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)  # mix fresh and rotten together
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Step 3: build the model = frozen MobileNetV2 + a small new "head"
# ---------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    # Download MobileNetV2 already trained on ImageNet, WITHOUT its last layer
    # (that last layer predicts 1,000 ImageNet classes; we need just 1 answer).
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze it: keep what it already learned

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
        # Data augmentation: random flips/rotations/zoom during training only,
        # so the model copes with photos taken from different angles.
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        # MobileNetV2 expects pixel values from -1 to 1 (photos are 0 to 255)
        tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
        base,
        # Summarise MobileNetV2's output into one list of 1,280 numbers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),  # randomly ignore some numbers, prevents memorising
        # Our new layer: one output between 0 and 1 = probability food is STALE
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",  # standard loss for yes/no questions
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Step 5 helper: draw and save the confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Freshness classifier - test images")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the FoodBridge AI freshness classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="training rounds (default 5)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dataset folder")
    args = parser.parse_args()

    # --- Step 1: find the data (or explain how to get it) ---
    train_dir, test_dir = args.data_dir / "train", args.data_dir / "test"
    if not train_dir.is_dir():
        print(HOW_TO_GET_DATA)
        sys.exit(1)

    print("Step 1: Finding images...")
    train_paths, train_labels = find_images(train_dir)
    if test_dir.is_dir():
        test_paths, test_labels = find_images(test_dir)
    else:
        # No test folder: hold back every 5th image (20%) for testing
        test_paths, test_labels = train_paths[::5], train_labels[::5]
        held_back = set(test_paths)
        keep = [i for i, p in enumerate(train_paths) if p not in held_back]
        train_paths = [train_paths[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]

    if not train_paths or not test_paths:
        print("No images found in fresh*/rotten* folders.\n" + HOW_TO_GET_DATA)
        sys.exit(1)
    print(f"  Training images: {len(train_paths)}  "
          f"(Fresh {train_labels.count(0)}, Stale {train_labels.count(1)})")
    print(f"  Test images:     {len(test_paths)}  "
          f"(Fresh {test_labels.count(0)}, Stale {test_labels.count(1)})")

    # --- Step 2: build the data pipelines ---
    print("\nStep 2: Preparing image batches...")
    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    test_ds = make_dataset(test_paths, test_labels, shuffle=False)

    # --- Step 3: build the model ---
    print("\nStep 3: Building MobileNetV2 transfer-learning model...")
    model = build_model()

    # --- Step 4: train. Each epoch = one full pass over the training images ---
    print(f"\nStep 4: Training for {args.epochs} epochs...")
    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, verbose=2)

    print("\nTraining accuracy per epoch:")
    for epoch, (acc, val_acc) in enumerate(zip(history.history["accuracy"],
                                               history.history["val_accuracy"]), start=1):
        print(f"  Epoch {epoch}: train {acc:.1%} | test {val_acc:.1%}")

    # --- Step 5: test on images the model has never seen ---
    print("\nStep 5: Evaluating on test images...")
    prob_stale = model.predict(test_ds, verbose=0).ravel()
    predicted = (prob_stale >= 0.5).astype(int)  # 0.5 or more = Stale
    actual = np.array(test_labels)
    test_accuracy = float((predicted == actual).mean())

    # Confusion matrix: rows = the true answer, columns = what the model said.
    # Diagonal = correct. Top-right = fresh food wrongly called stale.
    # Bottom-left = STALE food wrongly called fresh (the dangerous mistake!).
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    print(f"  Test accuracy: {test_accuracy:.1%}\n")
    print("  Confusion matrix (rows = actual, columns = predicted):")
    print(f"                 pred Fresh  pred Stale")
    print(f"  actual Fresh   {matrix[0, 0]:>10}  {matrix[0, 1]:>10}")
    print(f"  actual Stale   {matrix[1, 0]:>10}  {matrix[1, 1]:>10}\n")
    print(classification_report(actual, predicted, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))

    # --- Step 6: save the model, the chart and the numbers ---
    print("Step 6: Saving...")
    MODELS_DIR.mkdir(exist_ok=True)
    # .h5 = a single, widely supported model file. Keras prints a "legacy format"
    # warning for .h5; that's expected and safe to ignore.
    model.save(MODEL_PATH)
    save_confusion_matrix(matrix, MODELS_DIR / "freshness_confusion_matrix.png")
    metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "num_train_images": len(train_paths),
        "num_test_images": len(test_paths),
        "epochs": args.epochs,
    }
    (MODELS_DIR / "freshness_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {MODEL_PATH.relative_to(PROJECT_DIR)}")
    print("Saved confusion matrix to models/freshness_confusion_matrix.png")


if __name__ == "__main__":
    main()
